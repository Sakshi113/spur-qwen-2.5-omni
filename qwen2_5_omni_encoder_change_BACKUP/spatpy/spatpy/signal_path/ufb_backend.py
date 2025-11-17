from dataclasses import dataclass, field, InitVar
from typing import Any, Union, Optional, Tuple
from types import SimpleNamespace

from ufb_banding.banding.spatial import SpatialBandingCoefs
import torch
import torch.fft
import numpy as np
from ufb_banding.ufb import TransformCoefs, TransformParams
from ufb_banding.banding import (
    BandingCoefs,
    BandingParams,
    LowerBandMode,
    UpperBandMode,
    BandingShape,
)

import sys
import argparse
from scipy.io import savemat

from spatpy.signal_path.synthesis import InverseTransform
from spatpy.signal_path.analysis import (
    ForwardBanding,
    ForwardTransform,
    PowerVector,
)
from spatpy.signal_path.primitives import Reblocker
from spatpy.signal_path.io import read_wav_file


@dataclass
class UFBBandingSpatpy:
    coefs: InitVar[
        Union[TransformCoefs, BandingCoefs, SpatialBandingCoefs]
    ] = None
    stateful: bool = False
    align_bins: Optional[Tuple] = None
    align_bands: Optional[Tuple] = None
    align_pv: Optional[Tuple] = None

    def __post_init__(self, coefs):
        self.set_coefs(coefs)
        self.align_bins = (
            ("ch", "frame", "bin")
            if self.align_bins is None
            else self.align_bins
        )
        self.align_bands = (
            ("ch", "frame", "band")
            if self.align_bands is None
            else self.align_bands
        )
        self.align_pv = (
            ("band", "frame", "pv") if self.align_pv is None else self.align_pv
        )
        self.fwd_xfm = None
        self.fwd_banding = None
        self.fwd_pv = None
        self.inv_xfm = None

    def set_coefs(self, coefs):
        self.transform_coefs = None
        self.banding_coefs = None
        self.spatial_banding_coefs = None
        if coefs:
            if isinstance(coefs, TransformCoefs):
                self.transform_coefs = coefs

            if isinstance(coefs, SpatialBandingCoefs):
                self.spatial_banding_coefs = coefs

            if isinstance(coefs, BandingCoefs):
                self.transform_coefs = coefs.transform_coefs
                self.banding_coefs = coefs

        if hasattr(coefs, "fs"):
            self.fs = coefs.fs
        self.block_size = coefs.block_size

    def synthesise_bins_to_pcm(self, bins, interleaved=True, reset_state=False):
        reinit = not self.stateful or reset_state
        if self.inv_xfm is None or reset_state:
            self.inv_xfm = InverseTransform(self.transform_coefs)
        bins = (
            torch.from_numpy(bins)
            .refine_names(*self.align_bins)
            .to(torch.complex64)
        )
        pcm = self.inv_xfm(bins)
        pcm = (
            pcm.align_to("frame", "sample", "ch")
            .rename(None)
            .flatten(0, 1)
            .numpy()
        )
        return pcm

    def analyse_bins_to_pv(self, bins, reset_state=False):
        return self.analyse(bins=bins, reset_state=reset_state).pv

    def analyse_pcm_to_bins(self, pcm, reset_state=False):
        return self.analyse(
            pcm=pcm, reset_state=reset_state, compute_pv=False
        ).bins

    def analyse_bins_to_bands(self, bins):
        return self.analyse(bins=bins, compute_pv=False).bands

    def analyse_pcm_to_bands(self, pcm, reset_state=False):
        return self.analyse(
            pcm=pcm, reset_state=reset_state, compute_pv=False
        ).bands

    def analyse_pcm_to_pv(self, pcm, reset_state=False):
        return self.analyse(pcm=pcm, reset_state=reset_state).pv

    def synthesise_band_gains_to_bin_gains(self, band_gains):
        assert False, "Not implemented"

    def analyse(self, pcm=None, bins=None, compute_pv=True, reset_state=False):
        if pcm is not None:
            x = pcm
        else:
            x = bins

        if x.ndim == 1:
            x = np.expand_dims(x, 1)

        reinit = not self.stateful or reset_state
        if self.fwd_xfm is None or reinit:
            self.fwd_xfm = ForwardTransform(self.transform_coefs)

        if self.banding_coefs:
            if self.fwd_banding is None or reinit:
                self.fwd_banding = ForwardBanding(self.banding_coefs)

        if self.spatial_banding_coefs:
            if self.fwd_pv is None or reinit:
                self.fwd_pv = PowerVector(self.spatial_banding_coefs)

        if pcm is not None:
            if isinstance(pcm, np.ndarray):
                pcm = (
                    torch.from_numpy(x)
                    .refine_names("sample", "ch")
                    .to(torch.float32)
                )
            reblock = Reblocker(block_size=self.block_size)
            frames = reblock(pcm)
            bins = self.fwd_xfm(frames)

        else:
            if x.ndim == 2:
                x = np.expand_dims(x, 1)
            bins = (
                torch.from_numpy(x)
                .refine_names(*self.align_bins)
                .to(torch.complex64)
            )

        bands = None
        if self.fwd_banding is not None:
            bands = self.fwd_banding(bins)

        pv = None
        if compute_pv and self.fwd_pv is not None:
            pv = self.fwd_pv(bins)

        if self.align_bins is not None:
            bins = bins.align_to(..., *self.align_bins)

        bins = bins.numpy()

        if bands is not None:
            if self.align_bands is not None:
                bands = bands.align_to(..., *self.align_bands)
            bands = bands.numpy()

        if pv is not None:
            if self.align_pv is not None:
                pv = pv.align_to(..., *self.align_pv)
            pv = pv.numpy()

        return SimpleNamespace(bins=bins, bands=bands, pv=pv)

    @property
    def info(self):
        mode = "Transform"
        if not self.banding_coefs:
            mode += "-ONLY"
        else:
            mode += " and "
            if self.spatial_banding_coefs:
                mode += "Spatial "
            mode += "Banding"

        return "\n".join(
            [
                f"Mode: {mode}",
                "UFBBanding backend: Spatpy",
                f"Stateful: {self.stateful}",
            ]
        )


def add_banding_options(
    parser,
    fs=None,
    with_block_size=True,
    default_block_size_ms=10.0,
    default_round_to_nearest=1,
    default_fmin=50.0,
    default_fmax=16000.0,
    default_band_count=40,
    default_spacing=None,
    default_shape=None,
    default_transform=None,
    default_mode_lower=None,
    default_mode_upper=None,
):
    if default_spacing is None:
        default_spacing = "log"
    if default_shape is None:
        default_shape = "soft"
    if default_transform is None:
        default_transform = "RaisedSine"
    if default_mode_lower is None:
        default_mode_lower = "lpf"
    if default_mode_upper is None:
        default_mode_upper = "zeros_to_nyq"

    tfm = parser.add_argument_group("Banding options")
    if with_block_size:
        tfm.add_argument(
            "--block-size-ms",
            help="processing block size in milliseconds",
            type=float,
            default=default_block_size_ms,
        )
        tfm.add_argument(
            "--round-to-nearest",
            help="round processing block size to nearest power of 2",
            type=int,
            default=default_round_to_nearest,
        )
    tfm.add_argument(
        "--band-fmin",
        help="centre frequency of bottom band in Hz",
        type=float,
        default=default_fmin,
    )
    if fs is not None:
        default_fmax = fs / 2
    tfm.add_argument(
        "--band-fmax",
        help="centre frequency of top band in Hz",
        type=float,
        default=default_fmax,
    )
    tfm.add_argument(
        "--band-count",
        help="number of bands",
        type=int,
        default=default_band_count,
    )

    tfm.add_argument(
        "--band-spacing",
        help="spacing of band centre frequencies",
        choices=["mel", "log", "linear"],
        default=default_spacing,
    )
    tfm.add_argument(
        "--band-shape",
        help="band shape",
        choices=["hard", "soft", "triangular"],
        default=default_shape,
    )
    tfm.add_argument(
        "--band-mode-lower",
        help="lower band extension mode",
        choices=["lpf", "vsv_hpf", "hpf"],
        default=default_mode_lower,
    )
    tfm.add_argument(
        "--band-mode-upper",
        help="upper band extension mode",
        choices=["zeros_to_nyq", "ones_to_nyq"],
        default=default_mode_upper,
    )
    tfm.add_argument(
        "--transform",
        help="transform prototype window",
        choices=[
            "CLDFB",
            "CoreTransformCLDFB",
            "CoreTransformMDXT",
            "CoreTransformP4_319SC",
            "CoreTransformP64ATM301",
            "DolbyIntrinsicsGeneticFull",
            "DolbyIntrinsicsGeneticMini",
            "DolbyIntrinsicsGeneticMiniAST",
            "DolbyIntrinsicsHEAACFull",
            "DolbyIntrinsicsHEAACMini",
            "DolbyIntrinsicsHEAACMiniAST",
            "DolbyIntrinsicsP64ATM301Full",
            "DolbyIntrinsicsP64ATM301Mini",
            "DolbyIntrinsicsP64ATM301MiniAST",
            "RaisedSine",
        ],
        default=default_transform,
    )


def parse_banding_options(
    options, b_round_to_nearest_power_of_2=None, dt_ms=None
):
    if b_round_to_nearest_power_of_2 is None:
        b_round_to_nearest_power_of_2 = False
        if hasattr(options, "round_to_nearest"):
            b_round_to_nearest_power_of_2 = bool(options.round_to_nearest)

    if dt_ms is None:
        dt_ms = options.block_size_ms

    transform = getattr(TransformParams, options.transform)(
        b_round_to_nearest_power_of_2=b_round_to_nearest_power_of_2
    )
    if options.band_spacing == "log":
        fband = np.exp(
            np.linspace(
                np.log(options.band_fmin),
                np.log(options.band_fmax),
                options.band_count,
                endpoint=True,
            )
        )
    elif options.band_spacing == "mel":
        fband = BandingParams.melspace(
            options.band_fmin, options.band_fmax, options.band_count
        )
    else:
        assert options.band_spacing == "linear"
        fband = np.linspace(
            options.band_fmin,
            options.band_fmax,
            options.band_count,
            endpoint=True,
        )
    if options.band_shape == "triangular":
        shape = BandingShape.TRI
    elif options.band_shape == "soft":
        shape = BandingShape.SOFT
    else:
        assert options.band_shape == "hard"
        shape = BandingShape.HARD
    params = BandingParams(
        dt_ms,
        fband,
        shape,
        transform,
        lower_band_mode=LowerBandMode[options.band_mode_lower.upper()],
        upper_band_mode=UpperBandMode[options.band_mode_upper.upper()],
    )
    return params


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Produce a .mat file containing power vectors from a .wav file"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wavfile", help="input .wav file")
    parser.add_argument("--matfile", help="output .mat file", default="pv.mat")

    add_banding_options(parser)

    pv = parser.add_argument_group("Power vector options")
    pv.add_argument(
        "--hz_s_per_band",
        type=float,
        default=5.0,
        help="pv smoothing time constant",
    )
    pv.add_argument(
        "--flavour",
        choices=("short", "extended", "orthogonal"),
        default="extended",
        help="pv flavour",
    )
    return parser


def wav2pv(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_parser()
    options = parser.parse_args(args)

    params = parse_banding_options(options)

    pcm, fs = read_wav_file(options.wavfile)

    coefs = SpatialBandingCoefs(
        params,
        fs,
        nch=pcm.shape[1],
        hz_s_per_band=options.hz_s_per_band,
        gpv_flavour=options.flavour,
    )
    pv = UFBBandingSpatpy(coefs).analyse_pcm_to_pv(pcm)

    savemat(options.matfile, {"pv": pv})
