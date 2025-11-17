import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from ufb_banding.banding import BandingParams, BandingCoefs, BandingShape
from ufb_banding import UFBBanding
from spatpy.io import read_audio_file, write_audio_file
from spatpy.signal_path.io import heatmap, stacked_heatmap
from spatpy.signal_path.ufb_backend import (
    add_banding_options,
    parse_banding_options,
)
import numpy as np


@dataclass
class WindNoiseSuppressor:
    banding: BandingCoefs
    nmic: int = 3
    wind_fmax_hz: float = 3000
    bin_ratio_fmax_hz: float = 300
    wind_slope_min: float = 0
    wind_slope_scale: float = 2
    wind_level_decay_alpha: float = 0.95
    apply_aggressive_lf_suppression: bool = True
    aggressive_fmin_hz: float = 300
    aggressive_fmax_hz: float = 800
    burst_fmax_hz: float = 3000

    hpf_depth_db: float = 18.0

    # balanced settings
    suppression_duck_db: float = -9.0
    suppression_backfill_db: float = -1.5

    @property
    def top_wind_band(self):
        return np.searchsorted(self.banding.fband, self.wind_fmax_hz)

    @property
    def top_burst_band(self):
        return np.searchsorted(self.banding.fband, self.burst_fmax_hz)

    @property
    def wind_band_indices(self):
        return np.arange(self.top_wind_band)

    def estimate_wind_levels(self, mic_bins, mic_bands):
        # Spectral Tilt
        # =============
        eps = 1e-12
        nmic = mic_bins.shape[0]
        nframe = mic_bins.shape[1]
        spectral_tilt_bands = np.arange(1, self.top_wind_band)
        X = np.vstack(
            (
                np.ones(len(spectral_tilt_bands)),
                np.log10(self.banding.fband[spectral_tilt_bands]),
            )
        ).T
        logWindBands = 10 * np.log10(mic_bands[:, :, spectral_tilt_bands] + eps)

        wind_level_spec_tilt = np.zeros((nmic, nframe))
        for t in range(nframe):
            w = np.linalg.lstsq(X, logWindBands[:, t, :].T, rcond=None)[0]
            wind_level_spec_tilt[:, t] = w[1, :]

        # Sum & Diff Ratio
        # =================
        mics = [1, 0, 0] if self.nmic == 3 else [1, 0]
        s = mic_bins + mic_bins[mics, :, :]
        d = mic_bins - mic_bins[mics, :, :]
        bsum = self.banding.band_matrix @ np.moveaxis(s * np.conj(s), 2, 1).real
        bdif = self.banding.band_matrix @ np.moveaxis(d * np.conj(d), 2, 1).real

        sum_diff_ratio = bdif / (bsum + bdif + eps)
        weights = 0.7105 * np.expand_dims(
            self.wind_band_indices / len(self.wind_band_indices), (0, 2)
        )
        sum_diff_ratio = np.sum(
            sum_diff_ratio[:, self.wind_band_indices, :] * weights, axis=1
        )

        # Big Wind Ratio
        # ==============
        burst = np.sum(mic_bands, axis=2)
        if self.nmic == 3:
            a = (burst[0, :] / (burst[1, :] + eps)) + (
                burst[0, :] / (burst[2, :] + eps)
            )
            a = np.minimum(1, np.maximum(0, a - 5) / 100)
            b = (burst[1, :] / (burst[0, :] + eps)) + (
                burst[1, :] / (burst[2, :] + eps)
            )
            b = np.minimum(1, np.maximum(0, b - 5) / 100)
            c = (burst[2, :] / (burst[1, :] + eps)) + (
                burst[2, :] / (burst[0, :] + eps)
            )
            c = np.minimum(1, np.maximum(0, c - 5) / 100)
            burst_level = np.stack((a, b, c))
        else:
            a = burst[0, :] / (burst[1, :] + eps)
            a = np.minimum(1, np.maximum(0, a - 5) / 100)
            b = burst[1, :] / (burst[0, :] + eps)
            b = np.minimum(1, np.maximum(0, b - 5) / 100)
            burst_level = np.stack((a, b))

        level = np.maximum(
            0,
            (np.maximum(0, sum_diff_ratio - 0.15) / (1 - 0.15))
            * np.minimum(
                1,
                np.maximum(0, -wind_level_spec_tilt - self.wind_slope_min)
                / self.wind_slope_scale,
            ),
        )
        if self.nmic == 2:
            level /= 6
        else:
            level *= 2.8
        wind_level = np.minimum(1, level)

        wind_band_level = mic_bands[:, :, self.wind_band_indices] / np.sum(
            mic_bands[:, :, self.wind_band_indices], axis=(1, 2), keepdims=True
        )

        # decay wind noise signals in time
        for t in range(1, nframe):
            next_wl = np.maximum(wind_level[:, t - 1] * 0.9, wind_level[:, t])
            wind_level[:, t] = (
                1.0 - self.wind_level_decay_alpha
            ) * next_wl + self.wind_level_decay_alpha * wind_level[:, t - 1]
            burst_level[:, t] = np.maximum(
                burst_level[:, t - 1] * 0.9, burst_level[:, t]
            )
        return (wind_level, wind_band_level, burst_level)

    @property
    def backfill(self):
        return 10 ** (self.suppression_backfill_db / 10)

    def compute_aggressive_lf_gains(self, mic_bins, fmax):
        ratio_gain = np.ones_like(mic_bins, dtype=np.float32)

        mics1 = [1, 0, 0] if self.nmic == 3 else [1, 0]
        mics2 = [2, 2, 1] if self.nmic == 3 else [0, 1]

        bins1 = mic_bins[mics1, :, :]
        bins2 = mic_bins[mics2, :, :]

        S = bins1 + bins2
        D = bins1 - bins2

        SS = (S * np.conj(S)).real
        DD = (D * np.conj(D)).real

        r = np.maximum(
            10.0 ** (2 * self.suppression_duck_db / 10.0),
            SS / (1e-16 + SS + DD),
        )
        r = np.minimum(1.0, r)

        num_ratio_bins = np.searchsorted(self.banding.fbin, fmax)
        for (t, n) in enumerate(num_ratio_bins):
            ratio_gain[:, t, :n] = r[mics1, t, :n] * r[mics2, t, :n]
        return ratio_gain

    def compute_band_gains(self, mic_bins, mic_bands=None):
        if mic_bands is None:
            mic_bands = UFBBanding(self.banding).analyse_bins_to_bands(mic_bins)
        (wind_level, wind_band_level, burst_level) = self.estimate_wind_levels(
            mic_bins, mic_bands
        )
        s = np.tile(
            np.expand_dims(np.minimum(1, wind_level), 2),
            (1, 1, self.top_wind_band),
        )
        s[0, :, :] = np.max(wind_level, axis=0, keepdims=True).T
        HPFMix = np.ones_like(mic_bands)
        x = 10 ** (((np.arange(self.top_wind_band) / self.top_wind_band * s) * self.hpf_depth_db - self.hpf_depth_db) / 10) * \
            10 ** (((np.arange(self.top_wind_band) / self.top_wind_band * (1 - s) * self.hpf_depth_db) / 10))
        HPFMix[:, :, self.wind_band_indices] = x
        aggressive_fmax = self.aggressive_fmin_hz + np.max(burst_level, 0) * (
            self.aggressive_fmax_hz - self.aggressive_fmin_hz
        )
        nframe = wind_band_level.shape[1]
        nb = self.banding.nband

        #
        # Mono Mix
        # =====
        n = self.top_burst_band
        alpha = burst_level[0]
        beta = burst_level[1]
        gamma = None
        if self.nmic == 3:
            gamma = burst_level[2]

        # Supp M1
        # ==========
        x, y = self.compute_xy(beta, gamma if self.nmic == 3 else alpha)
        GbandsA = np.zeros((self.nmic, nframe, nb))
        GbandsA[:, :, :n] = np.expand_dims(
            np.stack(
                (
                    self.approx_db_gain(alpha),
                    alpha * x * self.backfill,
                    alpha * y * self.backfill,
                )[: self.nmic]
            ),
            2,
        )
        GbandsA[0, :, n:] = 1.0
        GbandsA[0, :, :] *= HPFMix[0, :, :]

        # Supp M2
        # ==========
        x, y = self.compute_xy(alpha, gamma if self.nmic == 3 else beta)
        GbandsB = np.zeros((self.nmic, nframe, nb))
        GbandsB[:, :, :n] = np.expand_dims(
            np.stack(
                (
                    beta * x * self.backfill,
                    self.approx_db_gain(beta),
                    beta * y * self.backfill,
                )[: self.nmic]
            ),
            2,
        )
        GbandsB[1, :, n:] = 1.0
        GbandsB[1, :, :] *= HPFMix[1, :, :]

        if self.nmic == 3:
            # Supp M3
            # ==========
            x, y = self.compute_xy(alpha, beta)
            GbandsC = np.zeros((self.nmic, nframe, nb))
            GbandsC[:, :, :n] = np.expand_dims(
                np.stack(
                    (
                        gamma * x * self.backfill,
                        gamma * y * self.backfill,
                        self.approx_db_gain(gamma),
                    )
                ),
                2,
            )
            GbandsC[2, :, n:] = 1.0
            # GbandsB[2, :, :] *= HPFMix[2, :, :]
            Gout = np.stack((GbandsA, GbandsB, GbandsC))
        else:
            # don't copy opposite channels
            GbandsA[1, :, :] = 0.0
            GbandsB[0, :, :] = 0.0
            Gout = np.stack((GbandsA, GbandsB))
        # Gout /= np.linalg.norm(Gout, axis=(0, 1))
        return Gout, aggressive_fmax

    @staticmethod
    def compute_xy(p, a):
        xy = 2 - (p + a)
        x = 1 - p
        y = 1 - a
        ind = (xy < 1) & (xy > 0)
        x[ind] /= xy[ind]
        y[ind] /= xy[ind]
        return x, y

    def approx_db_gain(self, x):
        # approx 10**(x*duckdB/10) by ..
        y = np.maximum(
            10 ** (self.suppression_duck_db / 10),
            1 + x * (10 ** (self.suppression_duck_db / 10) - 1.5),
        )
        return y

    def compute_bin_gains(self, mic_bins, mic_bands=None):
        ufb = UFBBanding(self.banding)
        band_gains, fmax = self.compute_band_gains(mic_bins)
        bin_gains = ufb.synthesise_band_gains_to_bin_gains(band_gains)
        if self.apply_aggressive_lf_suppression:
            aggressive_lf_gains = self.compute_aggressive_lf_gains(mic_bins, fmax)
            bin_gains *= aggressive_lf_gains[:, None, :, :]
        return bin_gains

    def apply_bin_gains(self, mic_bins, bin_gains):
        nframe = mic_bins.shape[1]
        nbin = mic_bins.shape[2]
        y = np.zeros_like(mic_bins)
        for t in range(nframe):
            for b in range(nbin):
                y[:, t, b] = bin_gains[:, :, t, b] @ mic_bins[:, t, b]
        return y


def suppress_wind_noise(banding: BandingParams, pcm, fs: float):
    coefs = BandingCoefs(banding, fs)
    xfm = UFBBanding(coefs)
    pcm -= np.mean(pcm, axis=0, keepdims=True)
    mics = xfm.analyse(pcm)
    wns = WindNoiseSuppressor(coefs, nmic=pcm.shape[-1])
    sup_gains = wns.compute_bin_gains(mics.bins, mics.bands)
    sup_bins = wns.apply_bin_gains(mics.bins, sup_gains)
    sup_pcm = xfm.synthesise_bins_to_pcm(sup_bins, interleaved=False)
    return sup_pcm, sup_gains


def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "in_wav", help="2 or 3 channel input wave file to be suppressed"
    )
    parser.add_argument(
        "--out_wav",
        help=(
            "wind suppressed output PCM file (default:"
            " [in_wav_name]_wind_suppressed.wav)"
        ),
    )
    parser.add_argument(
        "--out_gains",
        help=(
            "suppression gains numpy array file (default:"
            " [in_wav_name]_wind_suppression_gains.np)"
        ),
    )
    parser.add_argument(
        "--out_banding",
        help=(
            "banding params JSON file (default:"
            " [in_wav_name]_wind_suppression_banding.json)"
        ),
    )
    add_banding_options(
        parser,
        default_band_count=40,
        default_block_size_ms=20,
        default_mode_upper="ones_to_nyq",
        default_spacing="log",
        default_shape="soft",
    )
    return parser


def get_inputs(args):
    parser = make_parser()
    options = parser.parse_args(args)
    banding = parse_banding_options(options)
    pcm, fs = read_audio_file(options.in_wav)
    return options, banding, pcm, fs


def save_outputs(options, banding: BandingParams, sup_pcm, fs, sup_gains):
    p = Path(options.in_wav)
    out_pcm_file = (
        p.parent / (p.stem + "_wind_suppressed" + p.suffix)
        if options.out_wav is None
        else options.out_wav
    )
    out_gains_file = (
        p.parent / (p.stem + "_wind_suppression_gains.npy")
        if options.out_gains is None
        else options.out_gains
    )
    out_banding_file = (
        p.parent / (p.stem + "_wind_suppression_banding.json")
        if options.out_banding is None
        else options.out_banding
    )
    with open(out_banding_file, "wt") as fobj:
        fobj.write(banding.to_json())
    write_audio_file(str(out_pcm_file), sup_pcm, fs)
    np.save(out_gains_file, sup_gains)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    options, banding, pcm, fs = get_inputs(args)
    sup_pcm, sup_gains = suppress_wind_noise(banding, pcm, fs)
    save_outputs(options, banding, sup_pcm, fs, sup_gains)


if __name__ == "__main__":
    main()
