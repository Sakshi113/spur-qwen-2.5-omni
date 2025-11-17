from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Union, Dict
from plotly import graph_objects as go

import numpy as np
from scipy.fft import dct, idct
import json
from scipy.ndimage.filters import uniform_filter1d

import os
from pathlib import Path
from scipy.io import loadmat
from spatpy.signal_path.ufb_backend import UFBBandingSpatpy
from ufb_banding.banding.spatial import (
    SpatialBandingCoefs,
    SpatialBandingParams,
)
import xarray as xr

from ufb_banding import UFBBanding

from spatpy.device_model import DeviceModel
from spatpy.device_model.comsol import (
    COMSOL_DEVICE_GEOMETRY,
    ComsolModel,
)
from spatpy.geometry import Point, PointCloud
from spatpy.placement import DeviceGeometry, uniform_spherical_distribution
from spatpy.formats import apply_mix, mix_matrix, string_to_channel_format
from spatpy.ambisonics import (
    AmbisonicChannelFormat,
    BasisChannelFormat,
    ChannelFormat,
)
from spatpy.hankel import HankelFunction
from spatpy.io import read_audio_file
from spatpy import eq


CICERO_DEVICE_MIC_LABELS = {
    "find_x2": ["back", "top", "bottom"],
    "find_x3_pro": ["back", "top", "bottom"],
}

from spatpy.plot import ResponsePlotter
from spatpy.io import read_wav_file

SENSIBLE_FMIN_DEFAULT = 100
SENSIBLE_FMAX_DEFAULT = 8000
SENSIBLE_NBAND_DEFAULT = 20

DEFAULT_ACID_RESPONSE_DIR = "/opt/datasets/orig/acid_response"
from types import SimpleNamespace


def read_fusion_irs(dirname):
    p = Path(dirname)
    assert p.is_dir()
    wav_files = list(p.glob("bf/laptop/ir/*.wav"))
    irs = {}
    for ir_wav in wav_files:
        # example: imp_050deg_-20dB
        _, az_deg, gain_db = ir_wav.stem.split("_")
        az_deg = float(az_deg.rstrip("deg"))
        gain_db = float(gain_db.rstrip("dB"))
        pcm, fs = read_wav_file(str(ir_wav))
        irs[az_deg] = {
            "gain_db": gain_db,
            "az_deg": az_deg,
            "pcm": pcm,
            "fs": fs,
        }

    ir_list = []
    azimuths_deg = sorted([((-360 + k) if k > 180 else k) for k in irs.keys()])
    # add 90 degree offset to azimuths
    for az in azimuths_deg:
        ir_list.append((az, irs[az if az >= 0 else 360 + az]))
    return ir_list


def format_freq(f):
    return f"{f / 1000:.1f}kHz"


def fband_logspace(
    fmin_hz: Optional[float] = None,
    fmax_hz: Optional[float] = None,
    nband: Optional[int] = None,
) -> np.ndarray:
    if fmin_hz is None:
        fmin_hz = SENSIBLE_FMIN_DEFAULT
    if fmax_hz is None:
        fmax_hz = SENSIBLE_FMAX_DEFAULT
    if nband is None:
        nband = SENSIBLE_NBAND_DEFAULT
    return np.exp(np.linspace(np.log(fmin_hz), np.log(fmax_hz), nband))


def apply_mic_gains(
    S: xr.DataArray, w_opt: xr.DataArray, target_format=None
) -> xr.DataArray:
    if target_format is None:
        target_format = string_to_channel_format(w_opt.format)

    strategy_names = None
    if "strat" in w_opt.dims:
        strategy_names = w_opt.strat.values

    band = w_opt.band.values
    freq = (
        "band",
        w_opt.freq.values,
        {"long_name": "band centre frequency", "units": "Hz"},
    )
    dims = [target_format.dim_name, "band", "source"]
    coords = {
        "band": band,
        "freq": freq,
        "source": S.source,
    }
    coords.update(target_format.coords)
    size = (target_format.nchannel, len(band), len(S.source))
    if strategy_names is not None:
        dims = ["strat"] + dims
        coords["strat"] = strategy_names
        size = (len(strategy_names),) + size

    y = xr.DataArray(
        np.zeros(size, dtype=np.complex128),
        dims=dims,
        coords=coords,
        attrs={
            "long_name": "microphone band gains",
            "format": target_format.short_name,
        },
    )
    A = mix_matrix(src=w_opt.format, dst=target_format)
    if strategy_names is not None:
        for strat in strategy_names:
            for i in range(len(y.band)):
                h = A @ w_opt.loc[strat, :, i, :].values
                y.loc[strat, :, i, :] = h @ S.values[i, :, :].T
    else:
        for i in range(len(y.band)):
            h = A @ w_opt.loc[:, i, :].values
            y.loc[:, i, :] = h @ S.values[i, :, :].T

    y = y.assign_coords(source=S.source)
    return y


@dataclass
class DeviceResponse:
    mics: DeviceGeometry = field(
        default_factory=DeviceGeometry.circular_mics
    )  #: microphone geometry
    sources: DeviceGeometry = field(
        default_factory=DeviceGeometry.uniform_sphere_sources
    )  #: source positions
    body: Optional[PointCloud] = None
    model: Optional[DeviceModel] = None  #: device simulation model, optional
    descriptor: Optional[str] = None
    discrete_freqs_hz: Optional[
        np.ndarray
    ] = None  #: discrete frequencies at which this response is known, optional

    def __post_init__(self):
        if self.descriptor is None:
            self.descriptor = "unknown"

    @property
    def device_name(self) -> str:
        return self.parse_descriptor(self.descriptor).device_name

    @property
    def nsource(self) -> int:
        return self.sources.count

    @property
    def nmic(self) -> int:
        return self.mics.count

    def sample_bands(
        self,
        fband: Union[List[float], np.ndarray],
        mic_order=None,
        normalise_phase=True,
    ) -> np.ndarray:
        S = self.sample_frequency(fband)
        return self.create_sampled_response(
            S, fband, mic_order=mic_order, normalise_phase=normalise_phase
        )

    def convolve_at_angle(
        self,
        spatial_banding: SpatialBandingParams,
        mono_wavs: Union[str, List[str]],
        mic_order=None,
        azimuths_deg: Union[float, List[float]] = 0.0,
        elevation_deg=90.0,
        elevation_tolerance_deg=15.0,
    ):
        from spatpy.signal_path.analysis import PowerVector
        import torch

        if isinstance(mono_wavs, str):
            mono_wavs = [mono_wavs]
            azimuths_deg = [azimuths_deg]

        fs = None
        coefs = None
        mono_pcm = []
        nsample = []
        for fname in mono_wavs:
            pcm, cur_fs = read_wav_file(fname)
            if coefs is None:
                fs = cur_fs
                coefs = SpatialBandingCoefs(spatial_banding, fs, nch=1)
            else:
                assert fs == cur_fs
            mono_pcm.append(pcm)
            nsample.append(len(pcm))
        nsample = np.array(nsample)
        max_nsample = np.max(nsample)
        mono_pcm = [np.pad(x, (0, max_nsample - len(x))) for x in mono_pcm]
        mono_pcm = np.vstack(mono_pcm).T
        mono_bins = UFBBanding(coefs).analyse_pcm_to_bins(mono_pcm)
        mono_bins = xr.DataArray(mono_bins, dims=("source", "frame", "bin"))
        mono_bins = mono_bins.transpose("frame", "bin", "source")
        mono_bins = mono_bins.expand_dims("mic", axis=-1)

        resp = self.sample_bands(coefs.fbin).S.rename(band="bin")
        resp = resp.isel(
            source=np.abs(resp.el_deg - elevation_deg) < elevation_tolerance_deg
        )
        az_index = []
        for az in azimuths_deg:
            az_index.append(np.argmin(np.abs(resp.az_deg.values - az)))
        az_index = np.array(az_index)
        h_mic = resp[:, az_index, :]
        h_mic = h_mic.transpose("bin", "source", "mic")
        if mic_order is not None:
            h_mic = (
                h_mic.set_index(mic="miclabel")
                .sel(mic=mic_order)
                .set_index(mic="mic")
            )
        h_mic = h_mic.expand_dims("frame", axis=0)
        spatial_bins = mono_bins.values * h_mic.values

        coefs = SpatialBandingCoefs(spatial_banding, fs, nch=self.nmic)
        fwd_pv = PowerVector(coefs)

        pv = fwd_pv(
            torch.tensor(spatial_bins, names=("frame", "bin", "source", "ch"))
        )
        coords = {
            "pv": range(pv.shape[-1]),
            "band": range(spatial_banding.nband),
            "freq": (
                "band",
                spatial_banding.fband,
                {"long_name": "band centre frequency", "unit": "Hz"},
            ),
            "source": range(pv.shape[0]),
            "nframe": ("source", nsample // coefs.block_size),
        }
        sources = resp.source[az_index]
        coords.update(sources.coords)
        pv_dims = ("source", "frame", "band", "pv")
        pv_xr = xr.DataArray(
            pv.align_to(*pv_dims).numpy(),
            name="simulated power vectors",
            dims=pv_dims,
            coords=coords,
        )
        return spatial_bins, pv_xr

    def create_sampled_response(
        self, S, fband, mic_order=None, normalise_phase=True
    ) -> xr.DataArray:
        nband = len(fband)
        if normalise_phase:
            sources = self.sources.locs
            # subtract phase for a reference source
            ref_src = np.argmin((sources - Point(x=sources.r[0], y=0, z=0)).r)
            for band in range(len(fband)):
                phase_offset = np.angle(S[band, ref_src, 0])
                for i in range(self.nmic):
                    S[band, :, i] *= np.exp(-1j * phase_offset)
        S_xr = xr.DataArray(
            S,
            name=self.descriptor,
            dims=["band", "source", "mic"],
            coords={
                "band": range(nband),
                "freq": (
                    "band",
                    fband,
                    {"long_name": "band centre frequency", "unit": "Hz"},
                ),
                "source": range(self.nsource),
                "mic": range(self.nmic),
                "micx": ("mic", self.mics.locs.x, {"unit": "metre"}),
                "micy": ("mic", self.mics.locs.y, {"unit": "metre"}),
                "micz": ("mic", self.mics.locs.z, {"unit": "metre"}),
                "miclabel": (
                    "mic",
                    self.mic_labels,
                    {"long_name": "microphone label"},
                ),
            },
        )
        source_coords = self.sources.locs.points_xr.coords
        for (name, val) in source_coords.items():
            if name != "point" and name != "tag":
                S_xr = S_xr.assign_coords({name: ("source", val.data)})

        if mic_order is not None:
            S_xr = (
                S_xr.set_index(mic="miclabel")
                .sel(mic=mic_order)
                .set_index(mic="mic")
            )
        return SampledResponse(S_xr, discrete_freqs_hz=self.discrete_freqs_hz)

    @property
    def mic_labels(self):
        return [
            t if t else f"mic{i}" for (i, t) in enumerate(self.mics.locs.tags)
        ]

    def get_power_vectors(
        self,
        banding: SpatialBandingParams,
        block_size=None,
        mic_order=None,
        fs=48000.0,
    ) -> xr.DataArray:
        """Compute and return the `Power Vector <http://syd-dot.apac-eng.dolby.net/capture-docs/immersive/powervector/>`_ for each source"""

        pvs = []

        coefs = SpatialBandingCoefs(
            banding,
            block_size=block_size,
            fs=fs,
            nch=self.nmic,
        )
        S_bin = self.sample_bands(coefs.fbin, mic_order=mic_order).S.rename(
            band="bin"
        )
        ufb_engine = UFBBandingSpatpy(
            coefs,
            stateful=False,
            align_pv=("frame", "pv", "band"),
            align_bins=("source", "frame", "bin", "ch"),
        )
        S_bin = S_bin.transpose("source", "bin", "mic").expand_dims(
            frame=1, axis=1
        )
        pv = ufb_engine.analyse_bins_to_pv(S_bin.to_numpy()).squeeze()
        pv_xr = xr.DataArray(
            pv,
            name=S_bin.name,
            dims=("source", "pv", "band"),
            coords=dict(S_bin.source.coords),
        )
        pv_xr = pv_xr.assign_coords(
            band=range(coefs.nband),
            freq=("band", list(coefs.fband)),
            pv=range(pv_xr.sizes["pv"]),
        )

        return pv_xr

    def plot_bands(
        self,
        fband=None,
        plotter=None,
        write_html=None,
        body_trace=None,
        with_2d=True,
        **kwargs,
    ) -> go.Figure:
        """Plot device response magnitude and phase"""
        if fband is None:
            fband = np.linspace(100.0, 8000.0, 50)
        band_resp = self.sample_bands(fband)
        if body_trace is None and self.model:
            body_trace = self.model.body_trace
        if body_trace is None and self.body:
            body_trace = self.body.mesh3d(
                showlegend=False,
                showscale=False,
                polar=False,
                color="gray",
                opacity=0.4,
            )

        return band_resp.plot_bands(
            body_trace=body_trace,
            plotter=plotter,
            write_html=write_html,
            title=self.descriptor,
            with_2d=with_2d,
            **kwargs,
        )

    def plot_locations(self) -> go.Figure:
        """Plot device and source geometry"""
        fig = go.Figure()

        fig.add_trace(
            self.mics.locs.scatter(name="mics", marker=dict(color="black"))
        )

        if np.max(self.mics.locs.z) > np.min(self.mics.locs.z):
            fig.update_scenes(aspectmode="data")

        # _, scale = self.mics.locs.scale_maxdim(1.0)

        points = self.sources.locs
        fig.add_trace(points.scatterpolar(name="sources"))
        if self.body is not None:
            body_trace = self.body.mesh3d(
                showlegend=False,
                showscale=False,
                polar=False,
                color="gray",
                opacity=0.4,
            )
            fig.add_trace(body_trace)

        fig.update_scenes(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"
        )
        return fig

    @staticmethod
    def parse_descriptor(descriptor):
        kind = None
        custom_geometry = None
        custom_nmic = None
        sources = None
        az_step_deg = None
        el_step_deg = None
        manufacturer_id = None
        measurement_id = None

        if ":" not in descriptor:
            device_name = descriptor
        else:
            (kind, device_name) = descriptor.split(":")
            if kind == "custom":
                custom_geometry, s = device_name.split("_")
                mic_str = s[: [x.isdigit() for x in s].index(False)]
                custom_nmic = int(mic_str)
            elif kind == "fusion":
                manufacturer_id, device_name = device_name.split("_")
            elif kind == "cshare":
                device_name, measurement_id = device_name.split("-")

            tok = None
            placement_args = None
            for t in "@#*^":
                if t in device_name:
                    device_name, placement_args = device_name.split(t)
                    tok = t
                    break

            if tok is not None:
                if tok == "@":
                    n = int(placement_args)
                    sources = DeviceGeometry.uniform_sphere_sources(n)
                elif tok == "#":
                    az_step_deg, el_step_deg = placement_args.split("/")
                    az_step_deg = float(az_step_deg)
                    el_step_deg = float(el_step_deg)
                    sources = DeviceGeometry.equiangular_sphere_sources(
                        az_step_deg=az_step_deg, el_step_deg=el_step_deg
                    )
                elif tok == "^":
                    n = int(placement_args)
                    sources = DeviceGeometry.goldberg_sources(n)
                else:
                    assert tok == "*"
                    if placement_args is not None:
                        az_step_deg = float(placement_args)
                    sources = DeviceGeometry.turntable_sources(
                        az_step_deg=az_step_deg
                    )

        return SimpleNamespace(
            kind=kind,
            device_name=device_name,
            manufacturer_id=manufacturer_id,
            measurement_id=measurement_id,
            custom_geometry=custom_geometry,
            custom_nmic=custom_nmic,
            sources=sources,
            az_step_deg=az_step_deg,
            el_step_deg=el_step_deg,
        )

    @classmethod
    def from_descriptor(
        cls,
        descriptor,
        acid_response_dir=None,
        comsol_dir=None,
        fusion_tuning_dir=None,
        capture_and_share_dir=None,
        cicero_ir_dir=None,
        malfeasance_ir_dir=None,
    ):
        if acid_response_dir is None:
            acid_response_dir = DEFAULT_ACID_RESPONSE_DIR
        if fusion_tuning_dir is None:
            fusion_tuning_dir = os.path.join(
                acid_response_dir, "measurement", "fusion_tuning"
            )
        if comsol_dir is None:
            comsol_dir = os.path.join(acid_response_dir, "simulation", "comsol")
        if cicero_ir_dir is None:
            cicero_ir_dir = os.path.join(
                acid_response_dir, "simulation", "matlab"
            )
        if malfeasance_ir_dir is None:
            malfeasance_ir_dir = os.path.join(
                acid_response_dir, "simulation", "malfeasance"
            )
        if capture_and_share_dir is None:
            capture_and_share_dir = os.path.join(
                acid_response_dir, "measurement", "cshare"
            )
        desc = cls.parse_descriptor(descriptor)
        if desc.kind == "comsol":
            response = ComsolResponse.from_simulations_dir(
                desc.device_name,
                approx_sources=desc.sources,
                simulations_dir=comsol_dir,
                descriptor=descriptor,
            )
        elif desc.kind == "fusion":
            response = ImpulseResponse.from_fusion_tuning_dir(
                fusion_tuning_dir, desc.manufacturer_id, desc.device_name
            )
        elif desc.kind == "cshare":
            response = ImpulseResponse.from_capture_and_share_dir(
                capture_and_share_dir, desc.device_name, desc.measurement_id
            )
        elif desc.kind == "cicero":
            response = ImpulseResponse.from_cicero_ir_dir(
                cicero_ir_dir, desc.device_name, approx_sources=desc.sources
            )
        elif desc.kind == "malfeasance":
            response = SampledResponse.from_malfeasance_ir_dir(
                malfeasance_ir_dir,
                desc.device_name,
                approx_sources=desc.sources,
            )
        else:
            assert desc.kind == "custom"
            assert desc.custom_geometry == "circular"
            if desc.source_placement == "uniform":
                sources = DeviceGeometry.uniform_sphere_sources(
                    n=desc.uniform_n
                )
            elif desc.source_placement == "equiangular":
                sources = DeviceGeometry.equiangular_sphere_sources(
                    az_step_deg=desc.az_step_deg, el_step_deg=desc.el_step_deg
                )
            else:
                assert desc.source_placement == "turntable"
                sources = DeviceGeometry.turntable_sources(
                    az_step_deg=desc.az_step_deg
                )

            response = SimulatedResponse(
                mics=DeviceGeometry.circular_mics(desc.custom_nmic),
                sources=sources,
            )
        response.descriptor = descriptor
        return response


@dataclass
class SimulatedResponse(DeviceResponse):
    """Simulated device response from known microphone geometry using relative delays alone"""

    def __post_init__(self):
        self.delays_s = self.mics.relative_delays(self.sources)

    def sample_frequency(self, freq_hz) -> np.ndarray:
        return np.exp(
            -2j * np.pi * np.atleast_2d(freq_hz) * np.atleast_3d(self.delays_s)
        ).transpose(2, 0, 1)


MATLAB_DEVICE_DEFAULT_VERSIONS = {
    "find_x2": "2021-11-11",
    "find_x3_pro": "2021-12-03",
}
MATLAB_DEVICE_NAMES = {"find_x2": "FindX2", "find_x3_pro": "FindX3Pro"}

MALFEASANCE_DEVICE_DEFAULT_VERSIONS = {
    "find_x3_pro": "2022-06-08",
    "L13YogaGen1": "2022-06-17",
    "find_x5_pro": "2022-07-20",
}

FUSION_DEVICE_GEOMETRY = {
    "LNV": {
        "X1YogaGen6": DeviceGeometry.centred_linear_mics_mm([36.0, 138.0, 36.0])
    }
}

FUSION_DEFAULT_VERSIONS = {"LNV": {"L13YogaGen1": "41"}}

FUSION_DEVICE_NAMES = list(FUSION_DEVICE_GEOMETRY.keys())


@dataclass
class ImpulseResponse(DeviceResponse):
    """Measured or simulated impulse response"""

    ir: Optional[np.ndarray] = None
    ir_t_offset: int = 0
    fs: Optional[float] = None
    ir_stimulus: Optional[np.ndarray] = None

    @property
    def nframe(self) -> int:
        return self.ir.shape[0]

    def sample_ir(self, ir, freq, t, t_offset=0):

        if isinstance(freq, np.ndarray):
            return (
                ir.T
                @ np.exp(
                    -2j
                    * np.pi
                    * np.atleast_2d(freq).T
                    / self.fs
                    * (t - t_offset)
                ).T
            ).T
        return ir.T @ np.exp(-2j * np.pi * freq / self.fs * (t - t_offset))

    def sample_frequency(self, freq_hz) -> np.ndarray:
        n = self.nframe
        w = self.sample_ir(
            self.ir, freq_hz, np.arange(n), t_offset=self.ir_t_offset
        )

        if self.ir_stimulus is not None:
            m = self.ir_stimulus.shape[0]
            x = self.sample_ir(
                self.ir_stimulus, freq_hz, np.arange(m), t_offset=0
            )
            if isinstance(freq_hz, np.ndarray):
                x = np.expand_dims(x, (1, 2))
            w /= x

        return w

    @classmethod
    def from_capture_and_share_dir(
        cls,
        capture_and_share_dir: str,
        device_name: str,
        measurement_id: str,
        landscape=True,
    ):
        CAPTURE_AND_SHARE_DEVICE_NAMES = dict(find_x3_pro="oppofindx3pro")
        folder_name = CAPTURE_AND_SHARE_DEVICE_NAMES.get(device_name)
        assert folder_name is not None
        d = Path(capture_and_share_dir)
        wav_dir = d / f"{folder_name}_{measurement_id}"
        stimulus, fs = read_audio_file(str(d / "sweep_20_20k_5s_48k.wav"))
        ref_unaligned, ref_fs = read_audio_file(str(wav_dir / "ref_mic.wav"))
        ref_unaligned -= np.mean(ref_unaligned, 0)
        # _, ref, offset = align_and_truncate(stimulus, ref_unaligned)
        ir_window_len_s = 0.1
        ir_pad_s = 0.5
        ir_start_threshold_db = 30
        power_db = 20 * np.log10(np.abs(ref_unaligned) + 1e-6)
        y = uniform_filter1d(power_db[:, 0], int(ir_pad_s * fs))

        tstart = np.argmax(y > (np.min(y) + ir_start_threshold_db))
        tstart -= int(ir_pad_s * fs)
        tstart = max(tstart, 0)
        tend = min(ref_unaligned.shape[0], tstart + stimulus.shape[0])

        # window the start and end of the ir
        n = int(ir_window_len_s * fs)
        window = np.atleast_2d(
            0.5 * np.cos(np.pi * np.arange(n) / (n - 1)) + 0.5
        ).T
        ref = ref_unaligned[tstart:tend, :]
        ref[:n] *= window[::-1]
        ref[-n:] *= window

        assert ref_fs == fs

        source_locs = []
        mic_geometry = COMSOL_DEVICE_GEOMETRY.get(device_name, None).mics

        if landscape:
            sweeps = list(wav_dir.glob("*deg_landscape.wav"))
        else:
            sweeps = list(wav_dir.glob("*deg_portrait.wav"))

        # ref_mic = wav_dir / 'ref_mic.wav'

        nsource = len(sweeps)
        irs = []
        for (i, sweep) in enumerate(sweeps):
            filename = str(sweep)
            direction, angle = sweep.stem.split("_")[1:3]
            angle = int(angle.rstrip("deg"))
            if direction == "left":
                angle *= -1

            pcm, sweep_fs = read_audio_file(filename)
            # this measurement has the channels swapped for some reason
            if measurement_id == "BCN_repeat" and direction == "right":
                pcm = pcm[:, [1, 2, 0]]
            assert sweep_fs == fs
            source_locs.append(
                Point.from_spherical(
                    r=1.0,
                    az=angle,
                    el=0,
                    deg=True,
                )
            )
            # remove DC offset from mems mics
            pcm -= np.mean(pcm, 0)
            power_db = 20 * np.log10(np.abs(pcm) + 1e-6)
            y = uniform_filter1d(power_db[:, 0], int(ir_pad_s * fs))
            tstart = np.argmax(y > (np.min(y) + ir_start_threshold_db))
            tstart -= int(ir_pad_s * fs)
            tstart = max(tstart, 0)
            tend = min(pcm.shape[0], tstart + ref.shape[0])
            pcm = pcm[tstart:tend, :]
            # ramp up
            pcm[:n] *= window[::-1]
            # ramp down
            pcm[-n:] *= window
            irs.append(pcm)

        max_t = max([ir.shape[0] for ir in irs])
        all_irs = np.zeros((max_t, nsource, irs[0].shape[1]))

        for (i, ir) in enumerate(irs):
            # reverse the order of the mics to match the model
            all_irs[: ir.shape[0], i, :] = ir[:, ::-1]

        sources = DeviceGeometry(PointCloud(source_locs))
        return cls(
            mics=mic_geometry,
            sources=sources,
            ir=all_irs,
            ir_stimulus=ref.flatten(),
            # ir_t_offset=ir_t_offset,
            fs=fs,
            descriptor=f"cshare:{device_name}-{measurement_id}",
        )

    @classmethod
    def from_fusion_tuning_dir(
        cls,
        tuning_dir: str,
        manufacturer_id: str,
        device_name: str,
        ir_version: Optional[str] = None,
        source_dist_m: float = 1.0,
        source_elevation_jitter_deg=1.0,
    ):
        """Read recorded impulse responses from Fusion tuning directory //depot/aus/Advanced_Development/pal/fusion/main/tuning"""
        if ir_version is None:
            ir_version = FUSION_DEFAULT_VERSIONS.get(
                manufacturer_id, dict()
            ).get(device_name, None)
        assert ir_version
        ir_dir = os.path.join(
            tuning_dir, manufacturer_id, device_name, ir_version
        )
        irlist = read_fusion_irs(ir_dir)
        nsource = len(irlist)
        fs = None
        irs = None
        source_locs = []

        for (i, (az_deg, ir)) in enumerate(irlist):
            el_deg = np.random.uniform() * source_elevation_jitter_deg
            source_locs.append(
                Point.from_spherical(
                    r=source_dist_m,
                    az=az_deg,
                    el=el_deg,
                    deg=True,
                )
            )
            pcm = ir["pcm"]
            if fs is not None:
                assert ir["fs"] == fs
            else:
                fs = ir["fs"]

            if irs is None:
                nframe = pcm.shape[0]
                nmic = pcm.shape[1]
                irs = np.zeros((nframe, nsource, nmic))
            irs[: pcm.shape[0], i, :] = pcm * 10 ** (-ir["gain_db"] / 20)

        mic_geometry = None
        if manufacturer_id in FUSION_DEVICE_GEOMETRY:
            if device_name in FUSION_DEVICE_GEOMETRY[manufacturer_id]:
                mic_geometry = FUSION_DEVICE_GEOMETRY[manufacturer_id][
                    device_name
                ]
        if mic_geometry is None:
            spacing_mm = [50 for _ in range(nmic - 1)]
            mic_geometry = DeviceGeometry.centred_linear_mics_mm(spacing_mm)
        sources = DeviceGeometry(PointCloud(source_locs))
        bulk_delay = np.min(np.argmax(irs**2, 0), 0)
        ir_t_offset = np.min(bulk_delay)
        return cls(
            mics=mic_geometry,
            sources=sources,
            ir=irs,
            ir_t_offset=ir_t_offset,
            fs=fs,
            descriptor=f"fusion:{device_name}",
        )

    @staticmethod
    def _fem_result_to_geometry_and_irs(
        impulse_resp: Dict,
        mic_labels: Optional[List[str]] = None,
        transpose_ir: Optional[Tuple[int]] = None,
        source_distance_m: Optional[float] = None,
    ):
        locs = impulse_resp["Locs"]
        if "MicLabel" in impulse_resp:
            mic_labels = [m.rstrip() for m in impulse_resp["MicLabel"]]
        sources = PointCloud.from_cartesian(
            x=locs[0, :], y=locs[1, :], z=locs[2, :]
        )
        if source_distance_m is not None:
            sources = sources * (
                source_distance_m / np.max(sources.points_xr.r.values)
            )
        mic_pos = impulse_resp["MicPos"]
        mics = []
        for ind in range(mic_pos.shape[0]):
            mic = Point(
                x=mic_pos[ind, 0],
                y=mic_pos[ind, 1],
                z=mic_pos[ind, 2],
                tag=mic_labels[ind] if mic_labels is not None else f"mic{ind}",
            )
            mics.append(mic)
        vox = np.argwhere(impulse_resp["VOXGrid"] > 0)
        for i in range(vox.shape[1]):
            vox[:, i] -= impulse_resp["VOXGrid"].shape[i] // 2
        scale = impulse_resp["VOX_Scale"]
        bx, by, bz = (
            vox[:, 0] * scale,
            vox[:, 1] * scale,
            vox[:, 2] * scale,
        )
        fs = impulse_resp.get("FSample", 48000.0)
        ir = impulse_resp["IRs"]
        if transpose_ir:
            ir = np.transpose(ir, transpose_ir)

        stimulus = impulse_resp.get("InSig", None)

        ir_start = np.min(np.argwhere(np.logical_not(np.isclose(ir, 0)))[:, 0])
        ir = ir[ir_start:, :, :]

        return (bx, by, bz), mics, sources, stimulus, ir, fs

    @staticmethod
    def _mat_files_to_geometry_and_irs(
        ir_mat_files: List[str],
        mic_labels: List[str],
        source_distance_m: float,
        add_delay: bool,
    ):
        sigs = []
        mics = []
        nframe = []
        bulk_delays = []
        mic_ids = []
        for f in ir_mat_files:
            mic_id = int(f.stem.split("_")[-1]) - 1
            mic_ids.append(mic_id)
            ir = loadmat(str(f), squeeze_me=True)
            # put sources at arbitrary distance
            locs = ir["Locs"] / np.max(ir["Locs"]) * source_distance_m
            sources = PointCloud.from_cartesian(
                x=locs[0, :], y=locs[1, :], z=locs[2, :]
            )
            mx, my, mz = ir["MicPos"]
            mic = Point(
                x=mx,
                y=my,
                z=mz,
                tag=mic_labels[mic_id] if mic_labels else f"mic{mic_id}",
            )
            mics.append(mic)
            vox = np.argwhere(ir["VOXGrid"] > 0)
            for i in range(vox.shape[1]):
                vox[:, i] -= ir["VOXGrid"].shape[i] // 2
            s = ir["VOX_Scale"]
            bx, by, bz = (vox[:, 0] * s, vox[:, 1] * s, vox[:, 2] * s)
            fs = ir["FSample"]
            sig = ir["OutSig"]
            stimulus = ir["InSig"]

            bulk_delay = np.min(np.argmax(sig**2, 0)) - np.argmax(
                stimulus**2
            )
            bulk_delays.append(bulk_delay)
            nframe.append(sig.shape[0])
            sigs.append(sig)

        nsource = len(sources)
        nmic = len(mics)
        source_geom = DeviceGeometry(sources)
        mic_geom = DeviceGeometry(PointCloud(mics))
        if add_delay:
            delays_samples = np.ceil(
                fs * mic_geom.relative_delays(source_geom)
            ).astype(np.int32)
            delays_samples -= np.min(delays_samples, 0)
            nframe = np.max(
                np.array(nframe) + delays_samples - np.array(bulk_delays)
            )
        else:
            nframe = max(nframe)

        ir = np.zeros((nframe, nsource, nmic))
        if add_delay:
            for (i, sig) in enumerate(sigs):
                delay = delays_samples[:, i]
                for (j, t) in enumerate(delay):
                    n = sig.shape[0] + t - bulk_delays[i]
                    ir[:n, j, mic_ids[i]] = sig[(bulk_delays[i] - t) :, j]
        else:
            for (i, sig) in enumerate(sigs):
                ir[: sig.shape[0], :, mic_ids[i]] = sig

        return (bx, by, bz), mics, sources, stimulus, ir, fs

    @classmethod
    def _from_geometry_and_irs(
        cls, geometry_and_irs, approx_sources, include_z, **kwargs
    ):
        body, mics, sources, stimulus, ir, fs = geometry_and_irs
        source_geom = DeviceGeometry(sources)
        mic_geom = DeviceGeometry(PointCloud(mics))
        bx, by, bz = body
        body_corners = [
            Point(np.min(bx), np.min(by), np.min(bz)),
            Point(np.min(bx), np.min(by), np.max(bz)),
            Point(np.min(bx), np.max(by), np.min(bz)),
            Point(np.min(bx), np.max(by), np.max(bz)),
            Point(np.max(bx), np.min(by), np.min(bz)),
            Point(np.max(bx), np.min(by), np.max(bz)),
            Point(np.max(bx), np.max(by), np.min(bz)),
            Point(np.max(bx), np.max(by), np.max(bz)),
        ]
        body = PointCloud(body_corners)

        if approx_sources is not None:
            source_geom, subset = source_geom.sample_approx(approx_sources.locs)
            ir = ir[:, subset, :]

        # only take irs near the xy plane
        if not include_z:
            locs = sources.locs
            # get the index
            outline = sources.projection_outline(
                angularaxis="az", min_points=360
            )
            sources = PointCloud.from_cartesian(
                x=locs[0, outline], y=locs[1, outline], z=locs[2, outline]
            )
            source_geom = DeviceGeometry(sources)
            ir = ir[:, outline, :]

        bulk_delay = np.min(np.argmax(ir**2, 0), 0) - (
            0 if stimulus is None else np.argmax(np.squeeze(stimulus) ** 2)
        )
        return cls(
            mics=mic_geom,
            sources=source_geom,
            body=body,
            ir=ir,
            fs=fs,
            ir_t_offset=np.min(bulk_delay),
            ir_stimulus=None if stimulus is None else np.squeeze(stimulus),
            **kwargs,
        )

    @classmethod
    def from_malfeasance_result(
        cls,
        result: Dict,
        approx_sources=None,
        include_z=True,
        source_distance_m=10.0
    ):
        geometry_and_irs = cls._fem_result_to_geometry_and_irs(
            result, source_distance_m=source_distance_m
        )
        spec = json.loads(result["DeviceSpec"])
        model = DeviceModel.from_spec(spec)
        return cls._from_geometry_and_irs(
            geometry_and_irs,
            approx_sources,
            include_z,
            model=model,
            descriptor=f"malfeasance:{spec['name']}",
        )

    @classmethod
    def from_cicero_ir_dir(
        cls,
        ir_dir: str,
        device_name: str,
        device_version=None,
        add_delay=True,
        approx_sources=None,
        include_z=True,
        source_distance_m=10.0,
        descriptor=None,
    ):
        """Read impulse responses from Cicero MATLAB simulations"""
        ir_path = Path(ir_dir)

        # inconsistent naming
        cicero_device_name = MATLAB_DEVICE_NAMES.get(device_name, device_name)
        if device_version is None:
            device_version = MATLAB_DEVICE_DEFAULT_VERSIONS.get(device_name)

        mic_labels = CICERO_DEVICE_MIC_LABELS.get(device_name, None)
        ir_mat_files = list(ir_path.glob(f"{cicero_device_name}_IR_*.mat"))

        # all IRS in one file
        if len(ir_mat_files) == 0:
            stimulus = None
            filename = (
                ir_path
                / device_name
                / device_version
                / f"{device_name}_FEM_IRs.mat"
            )
            if not filename.exists():
                raise Exception(
                    "Unable to load Cicero impulse responses for"
                    f" {device_name} from {ir_dir}"
                )

            impulse_resp = loadmat(str(filename), squeeze_me=True)
            geometry_and_irs = cls._fem_result_to_geometry_and_irs(
                impulse_resp,
                mic_labels,
                transpose_ir=(
                    2,
                    1,
                    0,
                ),  # (nmic, nsource, nframe) -> (nframe, nsource, nmic)
                source_distance_m=source_distance_m,
            )
        else:
            geometry_and_irs = cls._mat_files_to_geometry_and_irs(
                ir_mat_files, mic_labels, source_distance_m, add_delay
            )

        return cls._from_geometry_and_irs(
            geometry_and_irs,
            approx_sources,
            include_z,
            descriptor=f"cicero:{device_name}"
            if not descriptor
            else descriptor,
        )


@dataclass
class ComsolResponse(DeviceResponse):
    """Device response at set of discrete frequencies, simulated using Comsol"""

    model: Any = None
    all_sources: Any = None
    source_mask: Optional[np.array] = None

    def sample_bands(self, fband, mic_order=None):
        df = self.model.mics[0].df
        nband = len(fband)
        S = np.zeros((nband, self.nsource, self.nmic), dtype=np.complex128)
        source_list, source_ids = np.unique(self.all_sources, return_index=True)
        for i in range(self.nmic):
            df = self.model.mics[i].df
            for (band, freq) in enumerate(fband):
                quantised_freq = df.freq[np.argmin(np.abs(df.freq - freq))]
                freq_df = df[df.freq == quantised_freq].iloc[
                    source_ids[self.source_mask]
                ]
                points, pressures = zip(
                    *[
                        (Point(x=x, y=y, z=z), p)
                        for ((x, y, z, p),) in zip(
                            freq_df.loc[:, ["x", "y", "z", "pressure"]].values
                        )
                    ]
                )
                assert np.array_equiv(points, source_list[self.source_mask]), (
                    "This code assumes the points for each frequency are the"
                    " same and in the same order"
                )
                S[band, :, i] = np.array(pressures)

        return self.create_sampled_response(S, fband, mic_order=mic_order)

    @classmethod
    def from_simulations_dir(
        cls,
        device_name,
        simulations_dir=None,
        device_version=None,
        approx_sources=None,
        descriptor=None,
    ):
        """Read from Comsol simulation output directory"""

        # normalise matlab device name
        orig_device_name = device_name
        if device_name == "FindX2":
            device_name = "find_x2"
        model = ComsolModel.from_simulations_dir(
            device_name,
            simulations_dir=simulations_dir,
            device_version=device_version,
        )
        device = COMSOL_DEVICE_GEOMETRY.get(device_name)
        mics = device.mics
        df = model.mics[0].df
        sim_freqs = np.array(sorted(list(set(df.freq))))
        all_sources = [
            Point(x=x, y=y, z=z)
            for ((x, y, z),) in zip(
                df.loc[df.freq == sim_freqs[0], ["x", "y", "z"]].values
            )
        ]
        source_list, source_ids = np.unique(all_sources, return_index=True)

        sim_sources = PointCloud(source_list)
        if approx_sources:
            subset = sim_sources.sample_approx(approx_sources.locs)
            sim_sources = PointCloud(source_list[subset])
        else:
            subset = np.ones(len(source_list), dtype=np.bool8)

        sources = DeviceGeometry(sim_sources)
        return cls(
            mics,
            sources,
            body=device.body,
            all_sources=all_sources,
            discrete_freqs_hz=sim_freqs,
            source_mask=subset,
            model=model,
            descriptor=f"comsol:{orig_device_name}"
            if not descriptor
            else descriptor,
        )


@dataclass
class SampledResponse:
    S: xr.DataArray
    discrete_freqs_hz: Optional[np.ndarray] = None
    model: Optional[DeviceModel] = None
    device_eq: Optional[xr.DataArray] = None

    def __post_init__(self):
        if "label" in self.S.dims:
            self.S = self.S.isel(label=0, drop=True)

        # this normalises out the average energy so that S has a total power of 1
        # so that relative costs are well-defined for the optimisation step
        # (TODO: investigate whether this is relevant)
        self.S /= np.sqrt(np.mean(np.abs(self.S) ** 2))
        # self.S /= self.nsource

    @property
    def source_locs(self):
        return PointCloud(points_xr=self.S.source)

    @property
    def mic_locs(self):
        return PointCloud.from_cartesian(
            x=self.S.micx, y=self.S.micy, z=self.S.micz
        )

    def av_spatial_response(self, band, smooth=False, ndct=3) -> xr.DataArray:
        h_avg = np.abs(self.S.sel(band=band)).mean(dim="mic").values
        if smooth:
            av_dct = dct(h_avg, norm="ortho")
            av_dct[ndct:] = 0
            h_avg = idct(av_dct, norm="ortho")
        return h_avg

    def av_mic_response(self, band) -> xr.DataArray:
        h_avg = np.abs(self.S.sel(band=band)).mean(dim="mic")
        return h_avg

    def band_is_above_simulation(self, band):
        return self.discrete_freqs_hz is not None and self.S.freq[
            band
        ] > np.max(self.discrete_freqs_hz)

    def get_band_target(
        self,
        target_format: ChannelFormat,
        band: int,
        modulate=False,
    ) -> xr.DataArray:
        """Compute the idealised FOA for the specified system response."""
        az = self.S.az.values
        el = self.S.el.values
        target = target_format.azel_to_pan(az, el)
        if modulate:
            modulated = target.copy()
            A = np.abs(self.S[band, :, :].values)
            b = target.values.T
            W = np.linalg.lstsq(A, b, rcond=None)[0]
            modulated = (A @ W).T
            modulated /= np.max(np.abs(modulated), axis=1, keepdims=True)
            target = modulated
        return target

    def get_target(
        self,
        target_format: ChannelFormat,
        modulate=False,
    ) -> xr.DataArray:
        y = self.get_response(target_format)
        for band in range(self.nband):
            target = self.get_band_target(
                target_format,
                band=band,
                modulate=modulate,
            )
            y.loc[:, band] = target
        return y

    def get_response(
        self,
        target_format: Optional[ChannelFormat] = None,
        strategy_names=None,
        w_opt=None,
    ) -> xr.DataArray:
        if w_opt is not None:
            if target_format is None:
                target_format = string_to_channel_format(w_opt.format)
            strategy_names = w_opt.strat.values
            band = w_opt.band.values
            freq = (
                "band",
                w_opt.freq.values,
                {"long_name": "band centre frequency", "units": "Hz"},
            )
        else:
            band = range(self.nband)
            freq = (
                "band",
                self.fband,
                {"long_name": "band centre frequency", "units": "Hz"},
            )

        dims = [target_format.dim_name, "band", "source"]
        coords = {
            "band": band,
            "freq": freq,
            "source": self.S.source,
        }
        coords.update(target_format.coords)

        nch = target_format.nchannel
        size = (target_format.nchannel, len(band), self.nsource)
        if strategy_names is not None:
            dims = ["strat"] + dims
            coords["strat"] = strategy_names
            size = (len(strategy_names), nch, len(band), self.nsource)

        y = xr.DataArray(
            np.zeros(size, dtype=np.complex128),
            dims=dims,
            coords=coords,
            attrs={
                "long_name": "microphone band gains",
                "format": target_format.short_name,
            },
        )
        if w_opt is not None:
            A = mix_matrix(src=w_opt.format, dst=target_format)
            for strat in strategy_names:
                for i in range(len(y.band)):
                    h = A @ w_opt.loc[strat, :, i, :].values
                    y.loc[strat, :, i, :] = h @ self.S.values[i, :, :].T

        y = y.assign_coords(source=self.S.source)
        return y

    @property
    def nband(self):
        return len(self.S.band)

    @property
    def nsource(self):
        return len(self.S.source)

    @property
    def fband(self):
        return self.S.freq.values

    @property
    def nmic(self):
        return len(self.S.miclabel)

    def plot_bands(
        self,
        fband=None,
        body_trace=None,
        plotter=None,
        write_html=None,
        title=None,
        with_2d=True,
        **kwargs,
    ) -> go.Figure:
        """Plot device response magnitude and phase"""
        S = self.S if fband is None else self.sample_bands(fband).S
        if plotter is None:
            plotter = ResponsePlotter(**kwargs)

        # divide by the RMS of the lowest frequency band
        S0 = S.sel({"band": 0})
        scale = np.sqrt(np.mean(np.abs(S0) ** 2))
        S /= scale

        if "strat" in S.dims:
            S = S.isel(strat=0, drop=True)

        if body_trace is None:
            body_trace = (
                self.model.body_trace if self.model is not None else None
            )
        return plotter.plot_response(
            S,
            body_trace=body_trace,
            label_coord="miclabel",
            write_html=write_html,
            with_2d=with_2d,
        )

    def to_hoa(
        self,
        hoa_format_name: Optional[str] = None,
        c_m_per_s=343.3,
        fs=48000.0,
        convert_to_farfield=False,
    ):
        """Derived from //depot/aus/Advanced_Development/Cicero/device_modelling/Build_3_DirectionalResponse_from_FEM_IRs.m"""

        if hoa_format_name is None:
            hoa_format_name = "BF9"

        S = self.S
        mic_responses = []
        fband = S.freq.values
        nband = len(fband)
        nmic = len(S.mic)

        BF = AmbisonicChannelFormat(hoa_format_name)
        sources = S.source
        DistLoc = sources.r.values
        DistMid = np.median(DistLoc)
        C = c_m_per_s
        HankelMid = HankelFunction(2 * np.pi * DistMid * fband / C, 0)
        HankelLoc = np.stack(
            [
                HankelFunction(2 * np.pi * DistLoc * f / C, range(BF.order))
                for f in fband
            ]
        )
        HankelAdjust = HankelLoc / np.expand_dims(HankelMid, -1)
        Enc1 = BF.azel_to_pan(sources.az.values, sources.el.values)
        Dec1 = np.linalg.pinv(Enc1.values)
        Order = Enc1.l
        for mic in range(nmic):
            FR = S[:, :, mic].copy()
            FR_BF = np.zeros((nband, len(BF.components)), dtype=np.complex128)
            for (n, f) in enumerate(fband):
                # For each freq bin, we compute the Hankel-based Spherical Harmonic
                for o in range(BF.order):
                    FR_1 = FR[n : (n + 1), :].values
                    if convert_to_farfield:
                        FR_1 *= HankelAdjust[n, :, o]
                    FR_BF[n : (n + 1), Order == o] = FR_1 @ Dec1[:, Order == o]

            mic_responses.append(FR_BF)

        S_hoa = xr.DataArray(
            np.stack(mic_responses, axis=-1),
            name=S.name,
            attrs=dict(fs=fs, format=hoa_format_name),
            dims=["band", BF.dim_name, "mic"],
            coords={
                "band": range(nband),
                "freq": (
                    "band",
                    fband,
                    {"long_name": "band centre frequency", "unit": "Hz"},
                ),
                "mic": range(nmic),
                "micx": ("mic", S.micx.values, {"unit": "metre"}),
                "micy": ("mic", S.micy.values, {"unit": "metre"}),
                "micz": ("mic", S.micz.values, {"unit": "metre"}),
                "miclabel": (
                    "mic",
                    S.miclabel.values,
                    {"long_name": "microphone label"},
                ),
            },
        )
        S_hoa = S_hoa.assign_coords(BF.coords)
        return S_hoa

    @classmethod
    def from_hoa(
        cls,
        S_hoa: xr.DataArray,
        mic_geom: DeviceGeometry,
        sources: PointCloud,
        add_delay=True,
        model=None,
    ):
        bf = AmbisonicChannelFormat(S_hoa.format)
        src = sources.points_xr.rename(point="source")
        basis = bf.azel_to_pan(src.az.values, src.el.values)
        S_sampled = S_hoa @ basis
        S_sampled = S_sampled.assign_coords(src.coords)
        S_sampled = S_sampled.transpose("band", "source", "mic")
        if add_delay:
            source_geom = DeviceGeometry(sources)
            delay = mic_geom.relative_delays(source_geom)
            delay -= np.min(delay, 0)
            S_sampled *= np.exp(
                -2j
                * np.pi
                * np.expand_dims(delay, 0)
                * np.expand_dims(S_sampled.freq, (1, 2))
            )

        """
        # subtract phase for a reference source
        ref_src = np.argmin((sources - Point(x=sources.r[0], y=0, z=0)).r)
        for band in range(len(S_sampled.freq)):
            phase_offset = np.angle(S_sampled[band, ref_src, 0])
            for i in range(S_sampled.shape[-1]):
                S_sampled[band, :, i] *= np.exp(-1j * phase_offset)
        # get the phase to line up with measurements
        # S_sampled = S_sampled.conj()
        """
        return cls(S_sampled, model=model)

    @classmethod
    def from_near_field_simulation(
        cls,
        near_field_result,
        fs: float = 48000.0,
        hoa_format_name=None,
        near_plot_filename=None,
        far_plot_filename=None,
        approx_sources=None,
        source_distance_m=10.0,
        fband=None
    ):
        imp = ImpulseResponse.from_malfeasance_result(
            near_field_result, source_distance_m=source_distance_m
        )
        if fband is None:
            oversample = imp.fs / fs
            nband = int(
                max(256, 2 ** np.ceil(np.log2(imp.ir.shape[-1] / oversample)))
            )
            fband = (np.arange(nband) + 0.5) / nband * fs / 2
        resp = imp.sample_bands(fband, normalise_phase=False)

        hoa_resp = resp.to_hoa(
            hoa_format_name=hoa_format_name, convert_to_farfield=True, fs=fs
        )
        if approx_sources:
            sources = approx_sources.locs * (
                source_distance_m
                / np.max(approx_sources.locs.points_xr.r.values)
            )
        else:
            sources = imp.sources.locs
        farfield_resp = cls.from_hoa(
            hoa_resp, imp.mics, sources, add_delay=True, model=imp.model
        )
        if far_plot_filename is not None:
            fig = farfield_resp.plot_bands(
                with_2d=False,
                title="Far-field compensated response",
            )
            fig.write_html(far_plot_filename)

        if near_plot_filename is not None:
            fig = resp.plot_bands(with_2d=False, title="Near-field response")
            fig.write_html(near_plot_filename)

        return farfield_resp

    @classmethod
    def from_malfeasance_ir_dir(
        cls,
        ir_dir: str,
        device_name: str,
        device_version=None,
        approx_sources=None,
        source_distance_m=10.,
        fband=None
    ):
        """Read impulse responses from Cicero MATLAB simulations"""
        ir_path = Path(ir_dir)

        if device_version is None:
            device_version = MALFEASANCE_DEVICE_DEFAULT_VERSIONS.get(
                device_name
            )

        filename = (
            ir_path
            / device_name
            / device_version
            / f"{device_name}_FEM_IRs.mat"
        )
        if not filename.exists():
            raise Exception(
                "Unable to load malfeasance impulse responses for"
                f" {device_name} from {ir_dir}"
            )

        near_field_result = loadmat(str(filename), squeeze_me=True)
        return cls.from_near_field_simulation(
            near_field_result, approx_sources=approx_sources, source_distance_m=source_distance_m, fband=fband
        )

    @property
    def device_name(self) -> str:
        return self.parse_descriptor(self.descriptor).device_name

    def sample_bands(self, fband, mic_order=None):
        inds = np.argmin(
            np.abs(np.atleast_2d(self.fband) - np.atleast_2d(fband).T), 1
        )
        S = self.S[inds, :, :]
        S = S.assign_coords(dict(band=np.arange(len(fband))))
        if mic_order is not None:
            S = (
                S.set_index(mic="miclabel")
                .sel(mic=mic_order)
                .set_index(mic="mic")
            )
        return SampledResponse(S)

    def flatten_using_eq_measurement(
        self, measurement: DeviceResponse, az_deg_min=-120, az_deg_max=120
    ):
        if isinstance(measurement, ImpulseResponse):
            measurement = measurement.sample_bands(self.fband)

        assert np.all(measurement.fband == self.fband)

        mic_eqs = eq.mic_corrections_from_measurement_and_simulation(
            measurement.S, self.S, az_deg_min=az_deg_min, az_deg_max=az_deg_max
        )

        mic_dims = [d for d in self.S.dims if d != "source"]
        device_eq = xr.DataArray(
            np.zeros(self.S.isel(source=0).values.shape, dtype=np.float32),
            name="per-microphone device EQ compensation",
            dims=mic_dims,
        )
        device_eq = device_eq.assign_coords(
            {d: self.S.coords[d] for d in mic_dims}
        )
        for (i, mic) in enumerate(device_eq.miclabel.values):
            mic_eq = device_eq.isel(mic=i)
            mic_eq[:] = mic_eqs[mic]
        self.device_eq = device_eq

    def get_power_vectors(
        self,
        banding: SpatialBandingParams,
        block_size=None,
        mic_order=None,
        fs=48000.0,
    ) -> xr.DataArray:
        """Compute and return the `Power Vector <http://syd-dot.apac-eng.dolby.net/capture-docs/immersive/powervector/>`_ for each source"""

        pvs = []
        coefs = SpatialBandingCoefs(
            banding,
            block_size=block_size,
            fs=fs,
            nch=self.nmic,
        )
        S_bin = self.sample_bands(coefs.fbin, mic_order=mic_order).S.rename(
            band="bin"
        )
        ufb_engine = UFBBandingSpatpy(
            coefs,
            stateful=False,
            align_pv=("frame", "pv", "band"),
            align_bins=("source", "frame", "bin", "ch"),
        )
        S_bin = S_bin.transpose("source", "bin", "mic").expand_dims(
            frame=1, axis=1
        )
        pv = ufb_engine.analyse_bins_to_pv(S_bin.to_numpy()).squeeze()
        pv_xr = xr.DataArray(
            pv,
            name=S_bin.name,
            dims=("source", "pv", "band"),
            coords=dict(S_bin.source.coords),
        )
        pv_xr = pv_xr.assign_coords(
            band=range(coefs.nband),
            freq=("band", list(coefs.fband)),
            pv=range(pv_xr.sizes["pv"]),
        )

        return pv_xr


if __name__ == "__main__":
    pass
