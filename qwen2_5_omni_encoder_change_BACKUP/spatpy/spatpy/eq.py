import argparse
from dataclasses import dataclass
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from ufb_banding.ufb import TransformParams, TransformCoefs
from ufb_banding.banding import (
    BandingParams,
    BandingCoefs,
    BandingShape,
    LowerBandMode,
    UpperBandMode,
)
from ufb_banding import UFBBanding
from spatpy.io import read_audio_file, write_audio_file
from typing import Optional, Any, List
import torch
from pathlib import Path
import sys
from scipy.fft import dct, idct


def to_db(x, as_powers=False):
    y = np.zeros_like(x)
    y[~np.isnan(x)] = (10 if as_powers else 20) * np.log10(x[~np.isnan(x)])
    y[np.isnan(x)] = np.NAN
    return y


def from_db(x, to_powers=False):
    return 10 ** (x / (10 if to_powers else 20))


def smooth_ir(
    y,
    freq=None,
    sharp_peak=False,
    ndct=32,
    resonance_fmax=10000,
    peak_width=1500,
    peak_tolerance_db=6,
):
    # smoothed response
    y_dct = dct(y, norm="ortho")
    y_dct[ndct:] = 0
    y_smooth = idct(y_dct, norm="ortho")

    if not sharp_peak:
        return y_smooth

    # index of resonant peak
    p = np.argmax(y[freq < resonance_fmax])

    # make the "smooth" version sharper around the peak
    # by fitting a cubic spline
    fpeak = freq[p]
    # initialise start and end of peak to something vaguely close
    peak_start = np.searchsorted(freq, (fpeak - peak_width / 2))
    peak_end = np.searchsorted(freq, (fpeak + peak_width / 2))
    # search backwards for the start
    while (
        peak_start > 0
        and np.abs(y[peak_start] - y_smooth[peak_start]) > peak_tolerance_db
    ):
        peak_start -= 1
    # search forwards for the end
    while (
        peak_end < len(y) - 1
        and np.abs(y[peak_start] - y_smooth[peak_start]) > peak_tolerance_db
    ):
        peak_end += 1
    y_smooth[peak_start:p] = np.interp(
        freq[peak_start:p], freq[[peak_start, p]], [y_smooth[peak_start], y[p]]
    )
    y_smooth[p:peak_end] = np.interp(
        freq[p:peak_end], freq[[p, peak_end]], [y[p], y_smooth[peak_end]]
    )

    return y_smooth


def mic_response(S, mic_name, az_deg):
    az_index = np.argmin((S.az_deg.values - az_deg + 360) % 360)
    x = S.freq.values
    y = to_db(
        np.abs(S[:, az_index, S.miclabel == mic_name].values.flatten()),
        as_powers=True,
    )
    return x, y


def mic_corrections_from_measurement_and_simulation(
    S_measured,
    S_sim,
    mic_reference_azimuths=None,
    az_deg_min=-120,
    az_deg_max=120,
    level_fmin=200,
    level_fmax=2000,
    eq_fmax=18000,
):
    if mic_reference_azimuths is None:
        mic_reference_azimuths = dict()
    freq = S_sim.freq.values
    mics = S_measured.miclabel.values
    mean_level = lambda x: np.mean(
        x[(freq >= level_fmin) & (freq < level_fmax)]
    )
    mic_eqs = dict()
    for mic in mics:
        # get the measured device response at the reference azimuth
        ref_az = mic_reference_azimuths.get(mic, 0)
        _, yref = mic_response(S_measured, mic, ref_az)

        # apply smoothing and subtract mean level in the given range
        yref_smooth = smooth_ir(yref, freq, sharp_peak=True)
        yref_smooth -= mean_level(yref_smooth)

        # get the simulated response at the reference azimuth
        _, sim_ref = mic_response(S_sim, mic, ref_az)
        sim_ref -= mean_level(sim_ref)

        # figure out the difference from the simulated reference measurement.
        # this is the gain correction that needs to be applied across all azimuths.
        correction = yref_smooth - sim_ref

        # average simulated response across all azimuths in the desired range
        sim_avg = np.zeros_like(yref)
        azimuths = S_sim.az_deg.values
        azimuths = azimuths[(azimuths >= az_deg_min) & (azimuths <= az_deg_max)]
        for az in azimuths:
            _, predicted = mic_response(S_sim, mic, az)
            predicted -= mean_level(predicted)
            sim_avg += predicted / len(azimuths)

        # to flatten the response, need to subtract the average prediction across all azimuths
        # plus the correction at the reference azimuth
        mic_eqs[mic] = from_db(-(sim_avg + correction), to_powers=True)

        # apply a constant gain above eq_fmax
        fmax_ind = np.searchsorted(freq, eq_fmax)
        if fmax_ind < len(mic_eqs[mic]):
            mic_eqs[mic][freq > eq_fmax] = mic_eqs[mic][fmax_ind]
    return mic_eqs


def match_files(
    in_filename,
    ref_filename,
    out_filename=None,
    format=None,
    post_gain_db=0.0,
    eq_out_filename=None,
    eq_in_filename=None,
    **kwargs,
):
    in_pcm, fs = read_audio_file(in_filename)
    ref_pcm, ref_fs = read_audio_file(ref_filename)
    assert fs == ref_fs
    matched_pcm, bin_eq, bulk_gain_db = match_eq(
        in_pcm,
        ref_pcm,
        fs,
        eq_in_filename=eq_in_filename,
        eq_out_filename=eq_out_filename,
        **kwargs)
    if out_filename is not None:
        matched_pcm *= from_db(post_gain_db)
        write_audio_file(out_filename, matched_pcm, fs, format=format)
    return matched_pcm, bin_eq, bulk_gain_db


def estimate_signal_spectrum(
    in_bands,
    mask_with_vad=False,
    vad_threshold=0.5,
    vad_min_snr_db=6.0,
    vad_plot_filename=None,
):
    if mask_with_vad:
        in_vad_info = calculate_vad(
            in_bands,
            vad_min_snr_db,
            vad_threshold,
            vad_plot_filename=vad_plot_filename,
        )
        mask = in_vad_info.voice_frames
    else:
        mask = np.ones(shape=in_bands.shape[1:], dtype=bool)

    spect_est = np.mean(in_bands[0, ...], axis=0, where=mask)
    spect_est_db = to_db(spect_est, as_powers=True)

    spect_mask_nframes = np.sum(mask, axis=0)

    return spect_est_db, spect_mask_nframes


@dataclass
class VadInfo:
    dist: Any
    voice_frames: np.ndarray
    speech_level: np.ndarray


def calculate_vad(
    band_powers, min_snr_db, vad_threshold, vad_plot_filename=None
):
    from utennsil.ospeech.speechnoise import BatchSpeechNoiseDist
    from spatpy.signal_path.io import stacked_heatmap

    bands_db = torch.tensor(to_db(band_powers, as_powers=True))

    dist = BatchSpeechNoiseDist(
        torch.mean(bands_db, axis=0, keepdim=True),
        min_snr_db=min_snr_db,
    )

    voice_frames = dist.p_speech.numpy().squeeze(0) >= vad_threshold

    speech_level = np.sqrt(np.mean(band_powers[:, voice_frames]))

    in_vad_info = VadInfo(
        dist=dist, voice_frames=voice_frames, speech_level=speech_level
    )
    if vad_plot_filename is not None:
        fig = stacked_heatmap(
            to_db(band_powers, as_powers=True),
            in_vad_info.dist.p_speech,
            in_vad_info.voice_frames.astype(int),
            title1="in bands",
            cmap1="viridis",
            title2="in p_speech",
            title3="mask",
        )
        fig.write_html(vad_plot_filename)
    return in_vad_info


def match_eq(
    in_pcm,
    ref_pcm,
    fs,
    eq_reference_channels: Optional[List[int]] = None,
    transform: Optional[TransformParams] = None,
    banding: Optional[BandingParams] = None,
    dt_ms: Optional[float] = 10.0,
    eq_fmin: Optional[float] = 0.0,
    eq_fmax: Optional[float] = None,
    mask_with_vad: Optional[bool] = False,
    vad_threshold: Optional[float] = 0.5,
    vad_min_snr_db: Optional[float] = 6.0,
    shape: BandingShape = BandingShape.SOFT,
    lower_band_mode: LowerBandMode = LowerBandMode.LPF,
    upper_band_mode: UpperBandMode = UpperBandMode.ONES_TO_NYQ,
    tilt: bool = False,
    in_vad_plot_filename: Optional[str] = None,
    ref_vad_plot_filename: Optional[str] = None,
    eq_plot_filename: Optional[str] = None,
    eq_in_filename: Optional[str] = None,
    eq_out_filename: Optional[str] = None
):
    """
    Modulate the EQ of ``in_pcm`` to match that of ``ref_pcm``, to make
    listening comparisons easier.

    Args:
        in_pcm: multichannel PCM to be modified
        ref_pcm: EQ reference multichannel PCM
        fs: sample rate (must be common to both ``in_pcm`` and ``ref_pcm``)
        eq_reference_channels: which channels to use as the EQ reference, if not supplied defaults to all
        transform: transform domain in which to apply EQ
        banding: banding domain in which to apply EQ
        dt_ms: transform length (if transform is not supplied)
        eq_fmin: only EQ from this frequency
        eq_fmax: only EQ up to this frequency
        mask_with_vad: use VAD for calculating EQ (requires ``ospeech``)
        vad_threshold: VAD confidence threshold, see ``ospeech`` for more info
        vad_min_snr_db: VAD SNR setpoint, see ``ospeech`` for more info
        shape: banding configuration (if banding not supplied)
        lower_band_mode: banding configuration (if banding not supplied)
        upper_band_mode: banding configuration (if banding not supplied)
        tilt: banding configuration (if banding not supplied)
        in_vad_plot_filename: plot input VAD result to this file
        ref_vad_plot_filename: plot reference VAD result to this file
        eq_plot_filename: plot applied EQ to this file

    Returns:
        A tuple of ``(matched_pcm, bin_eq, bulk_gain_db)``. ``bin_eq`` is the applied bin EQ, and
        ``bulk_gain_db`` is the net gain applied.
    """
    if eq_reference_channels is None:
        eq_reference_channels = [i for i in range(ref_pcm.shape[1])]
    if banding is None:
        if transform is None:
            transform = TransformParams.RaisedSine()
        banding = BandingParams.Log(
            dt_ms,
            fs,
            shape,
            transform,
            lower_band_mode=lower_band_mode,
            upper_band_mode=upper_band_mode,
            tilt=tilt,
        )
    coefs = BandingCoefs(banding, fs)
    xfm = UFBBanding(coefs)
    in_pcm -= np.mean(in_pcm, axis=0, keepdims=True)
    ref_pcm -= np.mean(ref_pcm, axis=0, keepdims=True)
    x_in = xfm.analyse(in_pcm)
    
    in_spec, in_mask_nframes = estimate_signal_spectrum(
        np.mean(x_in.bands, axis=0, keepdims=True),
        mask_with_vad=mask_with_vad,
        vad_threshold=vad_threshold,
        vad_min_snr_db=vad_min_snr_db,
        vad_plot_filename=in_vad_plot_filename,
    )

    if eq_in_filename is None:
        x_ref = xfm.analyse(ref_pcm)

        ref_spec, ref_mask_nframes = estimate_signal_spectrum(
            np.mean(x_ref.bands[eq_reference_channels, ...], axis=0, keepdims=True),
            mask_with_vad=mask_with_vad,
            vad_threshold=vad_threshold,
            vad_min_snr_db=vad_min_snr_db,
            vad_plot_filename=ref_vad_plot_filename,
        )

        if eq_fmax is not None:
            band_eq = np.zeros(shape=in_spec.shape)
            eq_fmax_idx = np.searchsorted(xfm.coefs.fband, eq_fmax)
            band_eq[:eq_fmax_idx] = (ref_spec - in_spec)[:eq_fmax_idx]
            band_eq[eq_fmax_idx:] = band_eq[
                eq_fmax_idx - 1
            ]  # extend the final gain up to Nyquist
        else:
            band_eq = ref_spec - in_spec

        bulk_gain_db = np.mean(band_eq)
        band_eq_linear = from_db(band_eq)
        bin_eq = xfm.synthesise_band_gains_to_bin_gains(
            np.expand_dims(np.expand_dims(band_eq_linear, 0), 0)
        )
        if eq_fmin is not None:
            eq_fmin_idx = np.searchsorted(xfm.coefs.fbin, eq_fmin)
            bin_eq[:, :, :eq_fmin_idx] = from_db(bulk_gain_db)
        
        if eq_out_filename:
            np.savez(
                eq_out_filename,
                bin_eq=bin_eq,
                bulk_gain_db=bulk_gain_db,
                band_eq=band_eq,
                ref_spec=ref_spec,
                allow_pickle=False)
    else:
        loaded_data = np.load(eq_in_filename, allow_pickle=False)
        bin_eq = loaded_data['bin_eq']
        bulk_gain_db = loaded_data['bulk_gain_db']
        band_eq = loaded_data['band_eq']
        ref_spec = loaded_data['ref_spec']

    in_level = np.mean(np.linalg.norm(in_pcm, axis=0) ** 2)
    ref_level = np.mean(
        np.linalg.norm(ref_pcm[:, eq_reference_channels], axis=0) ** 2
    )

    if eq_plot_filename is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=[0] + xfm.coefs.fband.tolist(),
                y=[to_db(in_level, as_powers=True)] + in_spec.tolist(),
                name=f"Original spectrum",
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0] + xfm.coefs.fband.tolist(),
                y=[to_db(ref_level, as_powers=True)] + ref_spec.tolist(),
                name=f"Reference spectrum",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xfm.coefs.fband,
                y=band_eq,
                name=f"Applied EQ",
                line=dict(dash="dot"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=xfm.coefs.fband,
                y=in_spec + band_eq,
                name=f"After EQ",
            )
        )

        fig.update_layout(
            xaxis_type="log",
            xaxis_title="Freq. (Hz)",
            yaxis_title="Mag. (dB)",
            yaxis2_title="Applied EQ (dB)",
        )

        fig.add_trace(
            go.Scatter(
                x=xfm.coefs.fbin,
                y=to_db(np.squeeze(bin_eq)),
                name=f"Applied EQ by bin",
                line=dict(dash="dot"),
            ),
            secondary_y=True,
        )
        fig.write_html(eq_plot_filename)
    

    matched_pcm = xfm.synthesise_bins_to_pcm(
        x_in.bins * bin_eq, interleaved=False
    )

    return matched_pcm, bin_eq.flatten(), bulk_gain_db


def make_parser():
    parser = argparse.ArgumentParser(
        description="Automatic EQ matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target_file", help="EQ target audio file (autodetect format)"
    )
    parser.add_argument(
        "in_file", help="input audio file(s) (autodetect format)", nargs="+"
    )
    parser.add_argument(
        "--suffix",
        help="suffix appended to input filenames when writing output files",
        default="_autoeq",
    )
    parser.add_argument(
        "--fmin", type=float, help="don't apply EQ below this frequency"
    )
    parser.add_argument(
        "--fmax", type=float, help="don't apply EQ above this frequency"
    )
    parser.add_argument(
        "--ext",
        help=(
            "output file extension. if not specified, defaults to input file"
            " extension."
        ),
    )
    parser.add_argument(
        "--format",
        help=(
            "output audio format, passed to pydub. if not specified, defaults"
            " to input file format."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="write debugging plots next to output files",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Use a rudimentary VAD to EQ only active content",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "path to output directory. if not specified, defaults to current"
            " working directory."
        ),
    )
    parser.add_argument(
        "--eq_out_filename",
        help=(
            "If specified the eq design result will be saved so it can be applied to other input files"
        ),
    )
    parser.add_argument(
        "--eq_in_filename",
        help=(
            "If specified the eq from the provided file will be used and the 'target_file' argument will be ignored"
        ),
    )
    return parser


def batch_match_eq(
    target_file,
    input_files,
    vad=False,
    plot=False,
    out_dir=None,
    ext=None,
    suffix=None,
    format=None,
    eq_out_filename=None,
    eq_in_filename=None,
    **kwargs,
):
    if out_dir is None:
        out_dir = "."
    if suffix is None:
        suffix = "_autoeq"

    d = Path(out_dir)
    eq_plot_filename = None
    in_vad_plot_filename = None
    ref_vad_plot_filename = None
    for f in input_files:
        ipath = Path(f)
        oext = ipath.suffix if ext is None else ext
        opath = d / (ipath.stem + suffix + oext)
        if plot:
            eq_plot_filename = d / (ipath.stem + "_eq.html")
            in_vad_plot_filename = d / (ipath.stem + "_in_vad.html")
            ref_vad_plot_filename = d / (ipath.stem + "_ref_vad.html")
        match_files(
            str(ipath),
            target_file,
            out_filename=str(opath),
            format=format,
            mask_with_vad=vad,
            eq_plot_filename=eq_plot_filename,
            in_vad_plot_filename=in_vad_plot_filename,
            ref_vad_plot_filename=ref_vad_plot_filename,
            eq_out_filename=eq_out_filename,
            eq_in_filename=eq_in_filename,
            **kwargs,
        )


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = make_parser()
    options = parser.parse_args(args)
    batch_match_eq(
        options.target_file,
        options.in_file,
        vad=options.vad,
        plot=options.plot,
        ext=options.ext,
        format=options.format,
        suffix=options.suffix,
        out_dir=options.out_dir,
        eq_fmin=options.fmin,
        eq_fmax=options.fmax,
        eq_out_filename=options.eq_out_filename,
        eq_in_filename=options.eq_in_filename
    )


if __name__ == "__main__":
    main()
