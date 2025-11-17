from scipy.io import wavfile, savemat
from typing import Tuple, Optional
import numpy as np
from tempfile import mkstemp
import os
import platform

import numpy as np
import json
from spatpy.ambisonics import AmbisonicChannelFormat
from spatpy.beehive import BeehiveChannelFormat
from spatpy.formats import ChannelFormat, string_to_channel_format, mix_matrix
from pathlib import Path
import argparse
import sys


def packaged_binary(dirname, exe_name):
    return os.path.join(
        os.path.dirname(__file__),
        "data",
        dirname,
        platform.system(),
        platform.machine(),
        f'{exe_name}{".exe" if platform.system() == "Windows" else ""}',
    )


def read_wav_file(
    filename: str, channel_axis: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    fs, raw = wavfile.read(filename)
    if raw.dtype.name.startswith("int"):
        pcm = raw * (1.0 / ((1 << (raw.itemsize * 8 - 1)) - 1))
    else:
        assert raw.dtype.name.startswith("float")
        pcm = raw

    if channel_axis is None:
        channel_axis = -1
    if channel_axis != -1:
        pcm = np.moveaxis(pcm, -1, channel_axis)
    return (pcm, fs)


def write_wav_file(
    filename: str,
    pcm: np.ndarray,
    fs: float,
    channel_axis: Optional[int] = None,
):
    """
    Write a 16 bit fixed-point wave file to disk.

    Args:
        filename: path to file
        pcm: samples (two-dimensional)
        fs: sample rate
        channel_axis (int, optional): dimension of pcm which corresponds to channels
    Returns:
        Result of :obj:`scipy.wavefile.write`
    """
    if channel_axis is None:
        channel_axis = -1
    if channel_axis != -1:
        pcm = np.moveaxis(pcm, channel_axis, -1)
    pcm_int16 = (pcm * ((2**15) - 1)).astype(np.int16)
    return wavfile.write(filename, int(fs), pcm_int16)


def read_audio_file(filename: str, channel_axis: Optional[int] = None):
    if not filename.endswith(".wav"):
        from pydub import AudioSegment

        _, tmp_wav = mkstemp(suffix=".wav")
        try:
            AudioSegment.from_file(filename).export(tmp_wav, format="wav")
            pcm, fs = read_wav_file(tmp_wav, channel_axis=channel_axis)
        finally:
            os.remove(tmp_wav)
    else:
        pcm, fs = read_wav_file(filename, channel_axis=channel_axis)
    if pcm.ndim == 1:
        channel_axis = 1 if channel_axis is None else channel_axis
        pcm = np.expand_dims(pcm, channel_axis)
    return pcm, fs


def write_audio_file(
    filename: str,
    pcm: np.ndarray,
    fs: float,
    channel_axis: Optional[int] = None,
    format=None,
):
    from pydub import AudioSegment

    wav = False
    if format is None:
        wav = filename.endswith(".wav")

    temp_file = None
    if not wav:
        _, temp_file = mkstemp(suffix=".wav")

    if channel_axis is None:
        channel_axis = -1
    if channel_axis != -1:
        pcm = np.moveaxis(pcm, channel_axis, -1)
    try:
        wav_file = filename if wav else temp_file
        write_wav_file(wav_file, pcm, fs)
        if not wav:
            kwargs = dict()
            if format is not None:
                kwargs["format"] = format
            AudioSegment.from_file(temp_file).export(filename, **kwargs)
    finally:
        if temp_file:
            os.remove(temp_file)


def damf_boilerplate(
    channel_format: ChannelFormat,
    caf_filename,
    metadata_filename,
    damf_version=None,
    presentation_type=None,
    bed_distribution=False,
    offset=0.0,
):
    if damf_version is None:
        damf_version = "0.5.4"
    if presentation_type is None:
        presentation_type = "home"
    return dict(
        version=damf_version,
        presentations=[
            dict(
                type=presentation_type,
                offset=offset,
                bedDistribution=bed_distribution,
                audio=caf_filename,
                metadata=metadata_filename,
                simplified=channel_format.damf_simplified,
                bedInstances=channel_format.damf_bed_instances,
                objects=channel_format.damf_objects,
            )
        ],
    )


def write_damf_file(
    pcm_format: ChannelFormat,
    pcm: np.ndarray,
    fs: float,
    damf_filename=None,
    metadata_filename=None,
    caf_filename=None,
    presentation_type=None,
    bed_distribution=False,
    binaural_render_mode=None,
    head_track_mode=None,
):

    # if it's ambisonics, first convert to beehive 3.1.0.0 (3.0.0.0 not supported by rosella)
    if isinstance(pcm_format, AmbisonicChannelFormat):
        assert pcm_format.order == 1, "Only FOA is currently supported"
        beehive_format = BeehiveChannelFormat("BH3.1.0.0")
        A = pcm_format.beehive_mix_matrix(beehive_format)
        bh_pcm = pcm @ A
        pcm = bh_pcm
        pcm_format = beehive_format

    if damf_filename is None:
        _, damf_filename = mkstemp(suffix=".damf")
    if metadata_filename is None:
        _, metadata_filename = mkstemp(suffix=".metadata")
    if caf_filename is None:
        _, caf_filename = mkstemp(suffix=".caf")
    write_audio_file(caf_filename, pcm, fs, format="caf")
    with open(metadata_filename, "wt") as fobj:
        json.dump(
            pcm_format.damf_metadata(
                fs,
                binaural_render_mode=binaural_render_mode,
                head_track_mode=head_track_mode,
            ),
            fobj,
            indent=4,
        )

    metadata_relative_filename = str(
        Path(metadata_filename).relative_to(Path(damf_filename).parent)
    )
    caf_relative_filename = str(
        Path(caf_filename).relative_to(Path(damf_filename).parent)
    )
    damf = damf_boilerplate(
        pcm_format,
        caf_relative_filename,
        metadata_relative_filename,
        presentation_type=presentation_type,
        bed_distribution=bed_distribution,
    )

    with open(damf_filename, "wt") as fobj:
        json.dump(damf, fobj, indent=4)

    return damf_filename, caf_filename, metadata_filename


def make_damfeezle_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "in_channel_format",
        help=(
            'input channel format string (case insensitive), e.g. "HOA1S",'
            ' "AmbiX", "5.1", "BH9.5.0.1"'
        ),
    )
    parser.add_argument(
        "in_filename", help="input audio file (most file formats supported)"
    )
    parser.add_argument(
        "--damf", help="output damf file name (default: '[in_filename].damf)'"
    )
    parser.add_argument(
        "--caf", help="output caf file name (default: '[in_filename].caf)'"
    )
    parser.add_argument(
        "--metadata",
        help="output metadata file name (default: '[in_filename].metadata)'",
    )
    parser.add_argument(
        "--presentation-type",
        choices=("cinema", "home"),
        default="home",
        help=(
            'Can be either "cinema" or "home", indicating whether master has'
            " been approved for encoding to a cinema DCP."
        ),
    )
    parser.add_argument(
        "--bed-distribution",
        action="store_true",
        help=(
            "Whether bed channel distribution (aka array processing) should be"
            " applied by the renderer if possible."
        ),
    )
    parser.add_argument(
        "--binaural-render-mode",
        choices=("off", "near", "middle", "far"),
        default="off",
        help="""Headphone virtualization mode for all objects. Possible values are:
"off" - normal stereo rendering with no headphone virtualization.
"near" - headphone virtualization with little to no room simulation.
"far" - headphone virtualization with room simulation.
"middle" - headphone virtualization with intermediate amount of room simulation between "near" and "far".
""",
    )
    parser.add_argument(
        "--head-track-relative-to",
        choices=("scene", "head"),
        default='head',
        help="""Rotation compensation for all objects when rendered binaurally.
"scene" - object position is relative to the world outside the listener (head rotation data used)
"head" - object position is relative to the listener's head (head rotation data not used)""",
    )
    return parser


def damfeezle(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_damfeezle_parser()
    options = parser.parse_args(args)
    p = Path(options.in_filename)
    damf_filename = (
        p.parent / (p.stem + ".damf") if options.damf is None else options.damf
    )
    metadata_filename = (
        p.parent / (p.stem + ".metadata")
        if options.metadata is None
        else options.metadata
    )
    caf_filename = (
        p.parent / (p.stem + ".caf") if options.caf is None else options.caf
    )
    pcm, fs = read_audio_file(options.in_filename)
    write_damf_file(
        string_to_channel_format(options.in_channel_format),
        pcm,
        fs,
        damf_filename=damf_filename,
        metadata_filename=metadata_filename,
        caf_filename=caf_filename,
        presentation_type=options.presentation_type,
        binaural_render_mode=options.binaural_render_mode,
        bed_distribution=options.bed_distribution,
        head_track_mode=options.head_track_relative_to + ' relative',
    )


def make_mixmate_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "source_format",
        help=(
            'source channel format string (case insensitive), e.g. "HOA3F",'
            ' "AmbiX", "5.1", "BH9.5.0.1"'
        ),
    )
    parser.add_argument(
        "destination_format", help="destination channel format string"
    )
    parser.add_argument(
        "in_filename",
        nargs="?",
        help=(
            "input audio file (source). If not specified, the mix matrix will"
            " be printed to stdout in CSV format."
        ),
    )
    parser.add_argument(
        "out_filename",
        nargs="?",
        help=(
            "If input file is specified, perform the mix and write the output"
            " to this filename (default:"
            " '[in_filename]_[destination_format].[in_ext]')"
        ),
    )
    parser.add_argument(
        "--mat-filename",
        help="Write the mix matrix to this .mat file",
    )
    return parser


def mixmate(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_mixmate_parser()
    options = parser.parse_args(args)
    A = mix_matrix(options.source_format, options.destination_format)
    if options.mat_filename:
        savemat(
            options.mat_filename,
            {f"{options.source_format}_to_{options.destination_format}": A},
        )
    if not options.in_filename:
        np.savetxt(sys.stdout, A, delimiter=",", newline="\r\n")
        return

    pcm, fs = read_audio_file(options.in_filename)
    p = Path(options.in_filename)
    out_filename = (
        p.parent / (p.stem + "_" + options.destination_format + p.suffix)
        if options.out_filename is None
        else options.out_filename
    )
    out_pcm = pcm @ A
    write_audio_file(out_filename, out_pcm, fs)


if __name__ == "__main__":
    damfeezle(
        [
            "binaural",
            "/Users/rzkate/proj/acid_immersive_sanity/acid_bf/acid_bf/device=find_x3_pro-nband=9-shape=soft-alpha=0.5/audition/2021-11-23_1_Boots/naive_binaural.flac",
        ]
    )
