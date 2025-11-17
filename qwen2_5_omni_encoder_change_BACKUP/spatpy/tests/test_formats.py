import numpy as np
import pytest
from spatpy.ambisonics import AmbisonicChannelFormat
from spatpy.speakers import SpeakerChannelFormat
from spatpy.formats import parse_channel_format
import re


@pytest.mark.parametrize(
    "format_name", list(AmbisonicChannelFormat.all_named_formats().keys())
)
def test_ambisonic_simple(format_name):
    az = np.zeros(10)
    el = np.zeros_like(az)
    fmt = AmbisonicChannelFormat(format_name)
    signals = fmt.azel_to_pan(az, el)
    w = signals[signals.acn == 0].squeeze()
    assert np.all(np.isclose(w, np.ones_like(w) * w.scale.item()))


@pytest.mark.parametrize(
    "format_name",
    ("HOA19F", "AMBIX", "BF4H", "WXYZ", "WXY", "2H|SN3D|FUMA", "3|1|DXM|ACN"),
)
def test_ambisonic_format_names(format_name):
    fmt = AmbisonicChannelFormat(format_name)
    name = dict(AMBIX="HOA1S", WXYZ="BF1", WXY="BF1H").get(
        format_name, format_name
    )
    assert fmt.canonical_name == name


@pytest.mark.parametrize(
    "fmt",
    [
        ("HOA19F", "ambisonic"),
        ("AMBIX", "ambisonic"),
        ("2H|SN3D|FUMA", "ambisonic"),
        ("BH3.1.0.0", "beehive"),
        ("BH9.5.0.1@30", "beehive"),
        ("5.1", "speaker"),
        ("22.2N", "speaker"),
        ("5/5/8c/3/1", "speaker"),
    ],
)
def test_format_parsing(fmt):
    s, kind = fmt
    parsed_kind, kwargs = parse_channel_format(s)
    assert parsed_kind.lower() == kind


@pytest.mark.parametrize(
    "format_def",
    (
        (("M", "LFE", "U"), "7.1.4"),
        (("M", "LFE"), "5.1"),
        (("M",), "2"),
        (("F", "B"), "3/2"),
        (("F", "B", "LFE"), "3/2.1"),
        (("F", "B", "U"), "3/4/4"),
        (("F", "B", "U", "LFE"), "3/4/4.1"),
        (("F", "B", "U", "L"), "3/4/4/3"),
        (("F", "B", "U", "L", "LFE"), "3/4/4/3.1"),
        (("F", "B", "Uc"), "3/4/4c"),
        (("F", "B", "Uc", "LFE"), "3/4/4c.1"),
        (("F", "B", "Uc", "L"), "3/4/4c/3"),
        (("F", "B", "Uc", "L", "LFE"), "3/4/4c/3.1"),
        (("F", "B", "Uc", "Lc"), "3/4/4c/3c"),
        (("F", "B", "Uc", "Lc", "LFE"), "3/4/4c/3c.1"),
        (("F", "B", "U", "L", "Z"), "3/4/4/3/1"),
        (("F", "B", "U", "L", "Z", "LFE"), "3/4/4/3/1.1"),
        (("F", "B", "Uc", "L"), "3/4/4c/1"),
        (("F", "B", "Uc", "L", "LFE"), "3/4/4c/1.1"),
        (("F", "B", "Uc", "L", "Z"), "3/4/4c/3/1"),
        (("F", "B", "Uc", "L", "Z", "LFE"), "3/4/4c/3/1.1"),
        (("F", "B", "Uc", "Lc", "Z"), "3/4/4c/3c/1"),
        (("F", "B", "Uc", "Lc", "Z", "LFE"), "3/4/4c/3c/1.1"),
    ),
)
def test_speaker_format(format_def):
    names, fmt = format_def
    vals = [int(v) for v in re.split(r"[^\d]", fmt.replace("c", ""))]
    d = SpeakerChannelFormat.parse(fmt, short_names=True)
    for (n, v) in zip(names, vals):
        assert d.get(n, 0) == v


def test_speaker_mapping():
    S1 = SpeakerChannelFormat("5.1", ordering="wiring")
    assert np.all(S1.dlb_name == ["L", "R", "C", "LFE", "L110", "R110"])
    S2 = SpeakerChannelFormat("4", ordering="wiring")
    assert np.all(S2.dlb_name == ["L", "R", "C", "BC"])
    A = S1.get_map_from(S2)
    assert np.all(
        A
        == np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0.5],
                [0, 0, 0, 0.5],
            ]
        )
    )
    S1 = SpeakerChannelFormat("5.1L")
    assert np.all(S1.dlb_name == ["L", "C", "R", "L110", "R110", "LFE"])


@pytest.mark.parametrize(
    "descriptor", list(SpeakerChannelFormat.KNOWN_FORMAT_DESCRIPTORS.keys())
)
def test_speaker_format_roundtrip(descriptor):
    spk = SpeakerChannelFormat(descriptor)
    spk.short_name == descriptor
    SpeakerChannelFormat(
        spk.canonical_name
    ).canonical_name == spk.canonical_name


if __name__ == "__main__":
    test_ambisonic_format_names("AMBIX")
    test_ambisonic_simple("HOA1F")
    test_speaker_format((("M", "LFE", "U"), "7.1.4"))
