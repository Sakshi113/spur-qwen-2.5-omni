from scipy.io import loadmat
from dataclasses import dataclass, field
import os
import sys
from spatpy.beehive import BeehiveChannelFormat
from spatpy.io import (
    read_audio_file,
    write_audio_file,
    packaged_binary,
    write_damf_file,
)
from typing import Optional, Dict
import argparse

from spatpy.ambisonics import (
    AmbisonicChannelFormat,
    ChannelFormat,
)
from spatpy.formats import string_to_channel_format
from spatpy.dsp import multichannel_convolve
from tempfile import mkstemp
import platform
import subprocess
from pathlib import Path
import numpy as np
from spatpy.speakers import SpeakerChannelFormat


DEFAULT_HRTF_FILENAME = os.path.join(
    os.path.dirname(__file__), "data", "hrtf", "ITA14_Coupled7_HOA1S.mat"
)

DEFAULT_ROSELLA_BINARY = packaged_binary(
    "rosella", "rosella_generic_float32_release"
)
DEFAULT_IMS_BINARY = packaged_binary(
    "ims_renderer", "imsren_test_generic_float32_std_release"
)


@dataclass
class MatlabHRTF:
    fs: float = 48000.0
    flip_lr: bool = False

    filename: Optional[str] = None
    channel_format: Optional[
        AmbisonicChannelFormat
    ] = AmbisonicChannelFormat.HOA1S

    def __post_init__(self):
        if self.filename is None:
            self.filename = DEFAULT_HRTF_FILENAME

        idx = {48000: 0, 32000: 1, 16000: 2, 8000: 3}[int(self.fs)]
        self.hrtf_lib = loadmat(self.filename)

        self.ir = self.hrtf_lib["ir"][0][idx].astype("float32")

        if self.flip_lr:
            self.ir = self.ir[:, [1, 0], :]
        self.latency_s = self.hrtf_lib["latency_s"][0][idx]

    @property
    def ir_len(self):
        return self.ir.shape[0]

    @property
    def n_in(self):
        return self.ir.shape[2]

    @property
    def n_out(self):
        return self.ir.shape[1]


@dataclass
class Binauraliser:
    include_z: bool = False

    @classmethod
    def binauralise_file(
        cls,
        in_filename: str,
        in_format: AmbisonicChannelFormat,
        bulk_gain_db=0.0,
        out_filename=None,
        **kwargs,
    ):
        pcm, fs = read_audio_file(in_filename)
        pcm *= 10.0 ** (bulk_gain_db / 20.0)
        state = cls(**kwargs)
        return state.binauralise(
            pcm, in_format, out_filename=out_filename, fs=fs
        )


@dataclass
class IVASBinauraliser(Binauraliser):
    flip_lr: bool = False
    hrtf_filename: Optional[str] = None
    truncate_result: bool = True
    hrtf_by_sample_rate: Dict = field(default_factory=dict)

    def binauralise(
        self,
        foa,
        foa_format: AmbisonicChannelFormat,
        fs: float = 48000.0,
        out_filename=None,
        out_format=None,
    ):
        if fs not in self.hrtf_by_sample_rate:
            self.hrtf_by_sample_rate[fs] = MatlabHRTF(
                fs=fs, filename=self.hrtf_filename, flip_lr=self.flip_lr
            )
        hrtf = self.hrtf_by_sample_rate[fs]
        pcm = hrtf.channel_format.reformat_from(foa, foa_format)
        include_z = self.include_z and foa_format.nchannel == 4
        ir = hrtf.ir
        if not include_z:
            pcm = pcm[:, :3]
            ir = ir[:, :, :3]
        out_pcm = multichannel_convolve(pcm, ir)
        if self.truncate_result:
            out_pcm = out_pcm[: pcm.shape[0], :]
        if out_filename is not None:
            write_audio_file(out_filename, out_pcm, fs, format=out_format)
        return out_pcm


@dataclass
class RosellaBinauraliser(Binauraliser):
    flip_lr: bool = False
    with_reverb: bool = False
    rosella_binary: str = field(default_factory=lambda: DEFAULT_ROSELLA_BINARY)
    binaural_render_mode: str = field(default_factory=lambda: "off")

    def binauralise(
        self,
        pcm,
        channel_format: ChannelFormat,
        fs: float = 48000.0,
        out_filename=None,
        out_format=None,
        binaural_render_mode=None,
    ):

        delete_out_file = False
        if out_filename is None:
            _, out_filename = mkstemp(suffix=".wav")
            delete_out_file = True

        if binaural_render_mode is None:
            binaural_render_mode = self.binaural_render_mode

        if (
            not self.include_z
            and isinstance(channel_format, AmbisonicChannelFormat)
            and channel_format.vertical_degree > 0
        ):
            new_channel_format = AmbisonicChannelFormat("BF1H")
            pcm = (channel_format.ambisonic_mix_matrix(new_channel_format) @ pcm.T).T
            channel_format = new_channel_format

        tmp_files = []
        try:
            tmp_files = write_damf_file(
                channel_format,
                pcm,
                fs,
                binaural_render_mode=binaural_render_mode,
            )
            subprocess.check_call(
                [
                    self.rosella_binary,
                    f"--init=rosella_mode={4 if self.with_reverb else 2}",
                    tmp_files[0],
                    f"--out={out_filename}",
                ]
            )
            out_pcm, fs = read_audio_file(out_filename)
            out_pcm = out_pcm[:, :2]
            if self.flip_lr:
                out_pcm = np.vstack((out_pcm[:, 1], out_pcm[:, 0])).T
            if not delete_out_file:
                write_audio_file(out_filename, out_pcm, fs, format=out_format)
        finally:
            for f in tmp_files:
                if f is not None:
                    os.remove(f)
            if delete_out_file:
                os.remove(out_filename)
        return out_pcm


def make_rosellify_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "in_channel_format",
        help=(
            'input channel format string (case insensitive), e.g. "HOA1S",'
            ' "AmbiX", "stereo", "7.1.4". Only first order ambisonics are'
            " currently supported."
        ),
    )
    parser.add_argument("in_filename", help="audio file to be binauralised")
    parser.add_argument(
        "--out-filename",
        help=(
            "output filename (default:"
            " '[in_name]_rosella_binaural.[in_format]')"
        ),
    )
    parser.add_argument(
        "--add-reverb",
        help="add reverb (full rosella, otherwise will be direct only)",
        action="store_true",
    )
    parser.add_argument(
        "--binaural-render-mode",
        help="binaural rendering mode",
        choices=("off", "near", "middle", "far"),
        default="off",
    )
    parser.add_argument(
        "--flip-lr",
        help="flip left and right channels in output",
        action="store_true",
    )
    parser.add_argument(
        "--pre-gain-db",
        default=0.0,
        type=float,
        help="bulk gain to be applied in dB before binauralising",
    )
    return parser


def rosellify(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_rosellify_parser()
    options = parser.parse_args(args)
    channel_format = string_to_channel_format(options.in_format)
    binauraliser = RosellaBinauraliser(
        include_z=True, flip_lr=options.flip_lr, with_reverb=options.add_reverb
    )
    pcm, fs = read_audio_file(options.in_filename)
    p = Path(options.in_filename)
    out_filename = (
        p.parent / (p.stem + "_rosella_binaural" + p.suffix)
        if options.out_filename is None
        else options.out_filename
    )
    binauraliser.binauralise(
        pcm,
        channel_format,
        fs=fs,
        out_filename=str(out_filename),
        binaural_render_mode=options.binaural_render_mode,
    )


@dataclass
class IMSRenderer:
    ims_binary: str = field(default_factory=lambda: DEFAULT_IMS_BINARY)

    def render(
        self,
        foa,
        foa_format: AmbisonicChannelFormat,
        fs: float = 48000.0,
        out_filename=None,
        out_format=None,
        include_z=True,
    ):

        delete_out_file = False
        if out_filename is None:
            _, out_filename = mkstemp(suffix=".wav")
            delete_out_file = True

        try:
            (tmp_damf, tmp_caf, tmp_meta) = write_damf_file(foa_format, foa, fs)
            subprocess.check_call(
                [self.ims_binary, "-i", tmp_damf, "-o", out_filename]
            )
            out_pcm, fs = read_audio_file(out_filename)
            if not delete_out_file:
                write_audio_file(out_filename, out_pcm, fs, format=out_format)
        finally:
            for f in (tmp_caf, tmp_damf, tmp_meta):
                os.remove(f)
            if delete_out_file:
                os.remove(out_filename)
        # [Lo, Ro, La, Ra, FDN, Lo_D, Ro_D, La_D, Ra_D]
        return out_pcm


@dataclass
class IMSBinauraliser(Binauraliser):
    flip_lr: bool = True
    ims_binary: str = field(default_factory=lambda: DEFAULT_IMS_BINARY)

    def __post_init__(self):
        self._renderer = IMSRenderer(ims_binary=self.ims_binary)

    def binauralise(
        self,
        foa,
        foa_format: AmbisonicChannelFormat,
        fs: float = 48000.0,
        out_filename=None,
        out_format=None,
        ims_out_format=None,
        ims_out_filename=None,
    ):
        ims_pcm = self._renderer.render(
            foa,
            foa_format,
            fs,
            out_filename=ims_out_filename,
            out_format=ims_out_format,
            include_z=self.include_z,
        )
        # grab the La Ra signals
        out_pcm = ims_pcm[:, 2:4]
        if self.flip_lr:
            out_pcm = np.vstack((out_pcm[:, 1], out_pcm[:, 0])).T
        if out_filename is not None:
            write_audio_file(out_filename, out_pcm, fs, format=out_format)
        return out_pcm, ims_pcm
