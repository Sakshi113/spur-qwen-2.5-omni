from scipy.signal import (
    fftconvolve,
    correlate,
    correlation_lags,
    lfilter,
    butter,
)
import numpy as np
from typing import Any, Optional, Tuple
from spatpy.io import read_audio_file, write_audio_file
import sys
import argparse
from pathlib import Path
import xarray as xr

BLOCK_SIZE = 2**19
""" The maximum block size allowed by blockwise functions"""


def _blockwise_1d(
    scipy_fn: Any, a: np.ndarray, b: np.ndarray, mode: Optional[str] = None
) -> np.ndarray:
    if mode is None:
        mode = "full"

    a = a.squeeze()
    b = b.squeeze()
    assert len(a.shape) == 1, "Inputs must be 1-D"
    assert len(b.shape) == 1, "Inputs must be 1-D"
    if mode == "same":
        samelen = np.max(a.shape)
        otherlen = np.max(b.shape)
    alen = np.max(a.shape)
    blen = np.max(b.shape)
    if alen < blen:
        a, b = b, a
        alen, blen = blen, alen
    result = np.zeros((alen + blen - 1))
    for i in range(0, alen, BLOCK_SIZE):
        result[i : i + BLOCK_SIZE + blen - 1] += scipy_fn(
            a[i : i + BLOCK_SIZE], b, mode="full"
        )
    if mode == "same":
        result = result[otherlen // 2 - 1 : samelen + otherlen // 2 - 1]
    elif mode == "valid":
        result = result[blen - 1 : alen]
    return result


def xcorr_1d(a: np.ndarray, b: np.ndarray, mode=None) -> np.ndarray:
    """Correlates a and b
    Applies some trickery so it is always as fast as possible and can handle large inputs
    """
    return _blockwise_1d(correlate, a, b, mode=mode)


def convolve_1d(a: np.ndarray, b: np.ndarray, mode=None) -> np.ndarray:
    """Convolves a with b
    Applies some trickery so it is always as fast as possible and can handle large inputs
    """
    return _blockwise_1d(fftconvolve, a, b, mode=mode)


def multichannel_convolve(pcm: np.ndarray, ir: np.ndarray) -> np.ndarray:
    block_size, n_out, n_in = ir.shape
    assert n_in == pcm.shape[1]
    y = None
    for i in range(n_in):
        outputs = []
        for j in range(n_out):
            outputs.append(convolve_1d(pcm[:, i], ir[:, n_out - 1 - j, i]))
        yi = np.vstack(outputs).T
        if y is None:
            y = np.zeros_like(yi)
        y += yi
    return y


def align_and_truncate(
    a: np.ndarray, b: np.ndarray, align_mode=None, channel_axis=-1
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Align two multichannel signals and truncate if necessary.

    Parameters
    ----------
     align_mode : str {'first', 'mean'}, optional
        ``first``
            align using the first channel
        ``mean``
            align using the mean of all channels

    Returns: a tuple of (``a_trunc``, ``b_trunc``, ``offset``)
    """
    if align_mode is None:
        align_mode = "first"

    # edge case
    sample_axis = (channel_axis + 1) % 2
    if a.shape[sample_axis] <= 1 or b.shape[sample_axis] <= 1:
        a_trunc = a.take(indices=[], axis=sample_axis)
        b_trunc = b.take(indices=[], axis=sample_axis)
        return a_trunc, b_trunc, None

    if align_mode == "first":
        a0 = np.take(a, 0, axis=channel_axis)
        b0 = np.take(b, 0, axis=channel_axis)
    else:
        assert align_mode == "mean"
        a0 = np.mean(a, axis=channel_axis)
        b0 = np.mean(b, axis=channel_axis)

    lags = correlation_lags(len(a0), len(b0))
    offset = lags[np.argmax(xcorr_1d(a0, b0))]

    if offset < 0:
        offset = abs(offset)
        result_len = max(0, min(len(a0), len(b0) - offset))
        aind = range(result_len)
        bind = range(offset, offset + result_len)
    else:
        result_len = max(0, min(len(a0) - offset, len(b0)))
        aind = range(offset, offset + result_len)
        bind = range(result_len)

    a_trunc = a.take(indices=aind, axis=sample_axis)
    b_trunc = b.take(indices=bind, axis=sample_axis)
    assert a_trunc.shape[sample_axis] == b_trunc.shape[sample_axis]

    return a_trunc, b_trunc, offset


def align_and_truncate_multiple(*args):
    truncated = [a.copy() for a in args]
    for i in range(len(truncated)):
        for j in range(len(truncated)):
            if i == j:
                continue
            truncated[i], truncated[j], _ = align_and_truncate(
                truncated[i], truncated[j]
            )
    min_len = min([t.shape[0] for t in truncated])
    for i in range(len(truncated)):
        truncated[i] = truncated[i][:min_len, :]

    assert all(
        [
            truncated[i].shape[0] == truncated[j].shape[0]
            for i in range(len(args))
            for j in range(len(args))
        ]
    )
    return tuple(truncated)


def align_and_truncate_files(*args):
    signals = []
    fs = None
    for filename in args:
        pcm, _fs = read_audio_file(filename)
        if fs is None:
            fs = _fs
        assert fs == _fs
        signals.append(pcm)
    return align_and_truncate_multiple(*signals), fs


def make_autoalign_parser():
    parser = argparse.ArgumentParser(
        description="Automatic alignment and truncation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "in_file", help="input audio files (autodetect format)", nargs="+"
    )
    parser.add_argument(
        "--suffix",
        help="suffix appended to input filenames when writing output files",
        default="_autoalign",
    )
    return parser


def autoalign_main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = make_autoalign_parser()
    options = parser.parse_args(args)
    signals, fs = align_and_truncate_files(*options.in_file)
    for (sig, f) in zip(signals, options.in_file):
        p = Path(f)
        out_file = str(p.parent / (p.stem + options.suffix + p.suffix))
        write_audio_file(out_file, sig, fs)


def block_dc(x, fs, f_cutoff_hz=50.0, sample_axis=0):
    a = np.array([1.0, 1.0 / np.exp(2.0 * np.pi * f_cutoff_hz / fs)])
    b = np.array([1.0])
    return lfilter(b, a, x, axis=sample_axis)


def low_pass(x, fs, f_cutoff_hz, sample_axis=0, order=12):
    b, a = butter(order, f_cutoff_hz, btype="low", fs=fs)
    return lfilter(b, a, x, axis=sample_axis)


def rfft(x, n=None, time_dimension='time', frequency_dimension='frequency'):
    """ Perform a real FFT on the input DataArray and return a new one where the
        `time_dimension` has been replaced with a new `frequency_dimension`. It
        infers the sample rate and therefore frequencies from the time step of
        the input """
    assert time_dimension in x.dims
    if n is None:
        n = len(x.time)
    H = xr.apply_ufunc(
        np.fft.rfft,
        x,
        kwargs={'n': n},
        input_core_dims=[[time_dimension]],
        output_core_dims=[[frequency_dimension]],
        dask='parallelized',
        output_dtypes=['float64'],
        dask_gufunc_kwargs=dict(output_sizes={frequency_dimension: n // 2 + 1})
    )
    fs = (1 / (x.coords[time_dimension][1] - x.coords[time_dimension][0])).values
    f = np.arange(n // 2 + 1) / n * fs
    H = H.assign_coords(**{frequency_dimension: f})
    H.frequency.attrs['units'] = 'Hz'
    return H


def irfft(H, n=None, frequency_dimension='frequency', time_dimension='time'):
    """ Perform a real IFFT on the input DataArray and return a new one where the
        `frequency_dimension` has been replaced with a new `time_dimension`. It
        infers the sample rate and therefore times from the frequency step of the
        input """
    assert frequency_dimension in H.dims
    if n is None:
        # Infer whether the length of the input to the RFFT was even or odd
        # based on whether the top bin is purely real (even) or not (odd)
        # note: this can fail so set the `n` input for explicit behaviour
        if np.allclose(H.isel({frequency_dimension: -1}).imag, 0):
            n = (len(H.coords[frequency_dimension]) - 1) * 2
        else:
            n = len(H.coords[frequency_dimension]) * 2 - 1
    x = xr.apply_ufunc(
        np.fft.irfft,
        H,
        kwargs={'n': n},
        input_core_dims=[[frequency_dimension]],
        output_core_dims=[[time_dimension]],
        dask='parallelized',
        output_dtypes=['float64'],
        dask_gufunc_kwargs=dict(output_sizes={time_dimension: n}))
    fs = (H.coords[frequency_dimension][1] - H.coords[frequency_dimension][0]).values * n
    t = np.arange(n) / fs
    x = x.assign_coords(**{time_dimension: t})
    x.time.attrs['units'] = 's'
    return x


def octave_smooth(H, nth_octave=3, frequency_dimension='frequency'):
    """ Smooth the response across the `frequency_dimension` by calculating
        a weighted average of the surrounding frequency points. The weighted
        average is based on a hanning function with a width corresponding to
        the `nth_octave` parameter. As an example `nth_octave`==3 is 3rd
        octave smoothing.

        The shape of the output is the same as the shape of H but it will be
        real as it is the average power that is calculated.
    """
    smoothed = (H * 0).real
    
    freqs = H.coords[frequency_dimension]
    for freq in freqs:
        lo = 2**(np.log2(freq) - 1/nth_octave)
        hi = 2**(np.log2(freq) + 1/nth_octave)
        to_avg = H.where((freqs >= lo) & (freqs <= hi), drop=True)
        win = xr.DataArray(
            np.hanning(len(to_avg.frequency) + 2)[1:-1],
            coords={frequency_dimension: to_avg.coords[frequency_dimension]})
        win /= win.sum(frequency_dimension)
        smoothed.loc[{frequency_dimension: freq}] = \
            (to_avg * np.conj(to_avg) * win).real.sum(frequency_dimension)**0.5

    return smoothed


if __name__ == "__main__":
    autoalign_main(
        [
            "/Users/rzkate/luna_park_acid_jazz/luna_park_bluejeans.wav",
            "/Users/rzkate/luna_park_acid_jazz/luna_park_local_mic.wav",
        ]
    )
