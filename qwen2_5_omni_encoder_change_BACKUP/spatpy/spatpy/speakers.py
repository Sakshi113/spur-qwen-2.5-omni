from dataclasses import dataclass, InitVar, field
import numpy as np
from typing import List, Optional, Any, Dict, NamedTuple
from enum import Enum
from functools import cached_property
import re
import xarray as xr
from spatpy.geometry import PointCloud, Point, source_coords, sph2cart


@dataclass(eq=True, frozen=True)
class NamedChannel:
    pass


@dataclass
class ChannelFormat:
    def damf_metadata(
        self, fs, binaural_render_mode=None, head_track_mode=None
    ):
        common_props = dict(
            samplePos=0,
            active=True,
            headTrackMode="head relative"
            if head_track_mode is None
            else head_track_mode,
        )
        if binaural_render_mode == "middle":
            common_props["binauralRenderMode"] = "undefined"
            common_props["binauralRenderMode2"] = "middle"
        else:
            common_props["binauralRenderMode"] = (
                "off" if binaural_render_mode is None else binaural_render_mode
            )

        events = [dict(ID=ch, **common_props) for ch in range(self.nchannel)]

        return dict(sampleRate=int(fs), events=events)

    @property
    def damf_simplified(self):
        return True

    @property
    def damf_bed_instances(self):
        return []

    @property
    def damf_objects(self):
        return []


@dataclass
class SampledChannelFormat(ChannelFormat):
    pass


class ChannelOrdering(Enum):
    """Channel ordering conventions"""

    def __init__(self, value):
        self.field_name = value

    def argsort(self, channels: List[NamedChannel]):
        """Return the permutation required to sort the given components according to this ordering"""
        cur = [getattr(c, self.field_name) for c in channels]
        return np.argsort(cur)

    def reorder(self, channels: List[NamedChannel]):
        """Reorder the given components according to this ordering"""
        channels.sort(key=lambda c: getattr(c, self.field_name))


class SpeakerChannelOrdering(ChannelOrdering):
    CHANNEL_FRONT_BACK = "cfb"  #: Channel front back ordering
    WIRING = "wiring"  #: Wiring order (suggested wave file ordering)
    STOCKHOLM = "stockholm"  #: Stockholm ordering
    NHK_22_2 = "nhk"  #: NHK 22.2 ordering
    MPEGH_22_0 = "mpegh22"  #: MPEGH22.0 ordering
    DOLBY_LEGACY = "legacy"  #: Dolby legacy channel order (L,C,R,...)
    BINAURAL = "ear"  #: Binaural channel order (L,R)


@dataclass(frozen=True, eq=True)
class SpeakerChannel(NamedChannel):
    wiring: int  #: wiring channel number
    cfb: int  #: channel front back number
    az: float  #: speaker azimuth (radians)
    el: float  #: speaker elevation (radians)
    dlb_name: str  #: name given by SpeakerFormatDescription MATLAB class

    bed_name: Optional[str] = None  #: DAMF bed name (optional)
    mpegh22: Optional[int] = None  #: MPEGH 22.0 channel number (optional)
    nhk: Optional[int] = None  #: NHK 22.2 channel number (optional)
    stockholm: Optional[int] = None  #: Stockholm channel number (optional)
    legacy: Optional[int] = None  #: Dolby legacy channel number (optional)
    ear: Optional[int] = None  #: Binaural channel number (optional)

    @property
    def lfe(self):
        # any of these are used to designate LFE
        if np.isclose(self.el, -np.pi / 2):
            return True
        return self.el in (None, np.nan)


class ISFParams(NamedTuple):
    NF: int
    NS: int
    NB: int
    NT: int
    NL: int
    NZ: int


def interp_angle(angles, x):
    perm = np.argsort(angles)
    invperm = np.argsort(perm)
    n = len(angles)
    angles = angles[perm]
    xp = np.concatenate((angles - 2 * np.pi, angles, angles + 2 * np.pi))
    weight = np.zeros_like(angles)
    for i in range(n):
        fp = (np.arange(3 * n) % n == i) * 1.0
        weight[i] = np.interp([x], xp, fp)
    return weight[invperm]


@dataclass
class SpeakerChannelFormat(SampledChannelFormat):
    # Default to 5.1 mode
    name: InitVar[str] = None
    n_mid_front: int = 3
    n_mid_back: int = 2
    n_upper: int = 0
    n_lower: int = 0
    n_zenith: int = 0
    n_lfe: int = 1

    ordering: Optional[str] = None

    rear_center_upper: bool = False  #: Is there a rear-center speaker overhead?
    rear_center_lower: bool = (
        False  #: Is there a rear-center speaker on the floor?
    )
    lr_angle_deg: float = 30.0
    mid_el_deg: float = 0.0
    upper_el_deg: float = 45.0
    lower_el_deg: float = -45.0

    extra_orderings: Dict = field(default_factory=dict)

    def __post_init__(self, name):
        self._short_name = name
        self.is_binaural = name == "binaural"
        if name is not None:
            fields = self.parse(name)
            for k, v in fields.items():
                # an explicitly different ordering has been asked for in the constructor,
                # so don't clobber this field
                if k == "ordering" and self.ordering is not None:
                    continue
                if k != "name":
                    self.__setattr__(k, v)

        if self.ordering is None:
            self.ordering = "cfb"

        if isinstance(self.ordering, str):
            self.ordering = SpeakerChannelOrdering(self.ordering)

        if self.extra_orderings is None:
            self.extra_orderings = dict()

    @property
    def damf_bed_instances(self):
        return [
            dict(
                groupName="main",
                channels=[
                    dict(ID=i, channel=name)
                    for (i, name) in enumerate(self.bed_name)
                ],
            )
        ]

    @property
    def bed_name(self):
        names = self.bed_name_cfb
        if names is None:
            return None
        return [names[i] for i in self.from_cfb_order]

    @property
    def bed_name_cfb(self):
        if self.rear_center_upper or self.rear_center_lower:
            return None
        if not self.n_upper in (0, 2, 4):
            return None
        if not self.n_mid in (1, 2, 5, 7, 9):
            return None

        if self.n_upper == 2:
            upper_bed_names = ["Ltm", "Rtm"]
        else:
            upper_bed_names = ["Ltf", "Rtf", "Ltr", "Rtr"]

        if self.n_mid == 1:
            mid_bed_names = ["C"]
        elif self.n_mid == 2:
            mid_bed_names = ["L", "R"]
        elif self.n_mid == 9:
            # 9.x
            mid_bed_names = [
                "C",
                "L",
                "R",
                "Lw",
                "Rw",
                "Ls",
                "Rs",
                "Lrs",
                "Rrs",
            ]
        else:
            # 5.x, 7.x
            mid_bed_names = ["C", "L", "R", "Ls", "Rs", "Lrs", "Rrs"]

        names = []
        for cfb_index in range(self.nchannel):
            if cfb_index >= (self.n_mid + self.n_upper):
                name = "LFE"
                n = cfb_index - (self.n_mid + self.n_upper)
                if n > 0:
                    name += str(n + 1)
            elif cfb_index >= self.n_mid:
                name = upper_bed_names[cfb_index - self.n_mid]
            else:
                name = mid_bed_names[cfb_index]
            names.append(name)
        return names

    @staticmethod
    def speaker_name(az, el):
        """Names as per SpeakerFormatDescription.m MATLAB class"""
        if az == 0:
            name = "C"
            if el != 0:
                name = f"F{name}"
        elif az == 180:
            name = "BC"
        elif az == 30 and el == 0:
            name = "L"
        elif az == -30 and el == 0:
            name = "R"
        elif az < 0:
            name = f"R{-az:.0f}"
        else:
            name = f"L{az:.0f}"

        if el in (-90, None):
            name = "LFE"
        elif el == 90:
            name = "Z"
        elif el > 0:
            name = "Tp" + name
        elif el < 0:
            name = "Bt" + name
        return name

    @property
    def short_name(self):
        if self._short_name is not None:
            return self._short_name
        return self.canonical_name

    @cached_property
    def canonical_name(self):
        # binaural, special case
        if self.is_binaural:
            return self.short_name
        f = str(self.n_mid_front)
        b = str(self.n_mid_back)
        u = str(self.n_upper) + ("c" if self.rear_center_upper else "")
        l = (
            ("/" + str(self.n_lower) + ("c" if self.rear_center_lower else ""))
            if (self.n_lower > 0 or self.n_zenith > 0)
            else ""
        )
        z = ("/" + str(self.n_zenith)) if self.n_zenith > 0 else ""
        lfe = ("." + str(self.n_lfe)) if self.n_lfe > 0 else ""

        if self.n_zenith > 0 or self.n_lower > 0 or self.n_upper > 0:
            return f"{f}/{b}/{u}{l}{z}{lfe}"
        elif self.n_mid_back > 0 or self.n_mid_front > 3:
            return f"{f}/{b}{lfe}"
        else:
            return f"{f}{lfe}"

    @classmethod
    def parse(cls, name: str, short_names=False):
        known_fmt = cls.KNOWN_FORMAT_DESCRIPTORS.get(name, None)
        ordering = None
        extra_orderings = None
        descriptor = name
        if known_fmt:
            descriptor = (
                known_fmt["descriptor"]
                if known_fmt["descriptor"]
                else descriptor
            )
            ordering = known_fmt["ordering"]
            extra_orderings = known_fmt["extra_orderings"]
        # e.g. 7.1.4
        matched = False
        d = None
        m = re.fullmatch(
            r"(?P<M>\d+)?\.?(?P<LFE>\d+)?\.?(?P<U>\d+)?\.?(?P<L>\d+)?",
            descriptor,
        )
        if m:
            matched = True
            d = m.groupdict(default="0")

        m = re.fullmatch(
            r"(?P<F>\d+)/(?P<B>\d+)(/((?P<U>\d+)|(?P<Uc>\d+)c))?(/((?P<L>\d+)|(?P<Lc>\d+)c))?(/(?P<Z>\d+))?(\.(?P<LFE>\d+))?",
            descriptor,
        )
        if m:
            matched = True
            d = m.groupdict(default="0")

        if not matched:
            return None

        n_mid_front = 0
        n_mid_back = 0
        n_upper = 0
        n_lower = 0
        n_zenith = 0
        n_lfe = 0
        rear_center_upper = False
        rear_center_lower = False

        for k, v in d.items():
            v = int(v)
            d[k] = v
            if k == "M":
                assert (
                    v >= 1
                ), "Number of mid-level channels should be 1 or more"
                if v == 2:
                    n_mid_front = 2
                else:
                    n_mid_front = 1 + 2 * ((v + 1) // 5)
                n_mid_back = v - n_mid_front
            elif k == "F":
                assert v >= 2, "Number of Front channels should be 2 or more"
                n_mid_front = v
            elif k == "B":
                n_mid_back = v
            elif k == "Z":
                assert v in (0, 1), "Number of Zenith channels should be 0 or 1"
                n_zenith = v
            elif k == "LFE":
                assert v in (0, 1, 2), "Number of LFE channels should be 0..2"
                n_lfe = v
            elif k in ("U", "Uc") and v > 0:
                n_upper = v
                if k == "Uc":
                    rear_center_upper = True
            elif k in ("L", "Lc") and v > 0:
                n_lower = v
                if k == "Lc":
                    rear_center_lower = True
        if short_names:
            return d

        return dict(
            name=name,
            n_mid_front=n_mid_front,
            n_mid_back=n_mid_back,
            n_upper=n_upper,
            n_lower=n_lower,
            n_zenith=n_zenith,
            n_lfe=n_lfe,
            rear_center_upper=rear_center_upper,
            rear_center_lower=rear_center_lower,
            ordering=ordering,
            extra_orderings=extra_orderings,
        )

    @property
    def n_mid(self):
        return self.n_mid_front + self.n_mid_back

    @property
    def is_lfe(self):
        return self.el_deg == -90

    @property
    def is_center(self):
        return np.mod(self.az_deg, 180) == 0

    @property
    def az_deg_cfb(self):
        """
        Speaker azimuths in CFB order

        ChannelFrontBackOrder (the simplest way to define speaker order):
        -----------------------------------------------------------------
        This format orders all the channels in a logical ordering, as follows:

        The speakers are ordered overall in planes (or rings) (wth LFE last):

        Format <=  MidPlane, [UpperPlane], [LowerPlane], [Zenith], [LFEs]

        Next, within each plane, they are ordered from front to back, with L/R
        pairs grouped together:

        MidPlane  <= [Ctr], SymPair, [SymPair], ..., [RearCtr]
        UpperPlane <= [Ctr], SymPair, [SymPair], ..., [RearCtr]
        LowerPlane <= [Ctr], SymPair, [SymPair], ..., [RearCtr]
        LFEs <= LFE1, [LFE2], ...
        """
        l_az = self.lr_angle_deg
        # Middle Ring:
        # Start with Front
        if self.n_mid_front == 1:
            az = np.zeros(1)
        else:
            az = (
                2 * np.arange(self.n_mid_front) / (self.n_mid_front - 1) - 1
            ) * l_az
        # Add the back
        if self.n_mid_back == 2:
            az = np.concatenate((az, [110, 250]))
        elif self.n_mid_back == 4:
            az = np.concatenate((az, [90, 270, 135, 225]))
        else:
            az = np.concatenate(
                (
                    az,
                    180
                    + (
                        2
                        * np.arange(1, self.n_mid_back + 1)
                        / (self.n_mid_back + 1)
                        - 1
                    )
                    * (180 - l_az),
                )
            )

        assert len(az) == (self.n_mid_front + self.n_mid_back)

        def normalize_az(az):
            az = np.round(az * 4096) // 4096
            az = np.mod(az, 360)
            az = np.sort(az)
            az = az[az <= 180]
            az = np.vstack((az, -az)).T.flatten()
            if az[0] == 0:
                az = az[1:]
            if az[-1] == -180:
                az = az[:-1]
            return az

        az = normalize_az(az)

        # Now, the upper ring
        if self.n_upper > 0:
            upper_az = normalize_az(
                self.rear_center_upper * 180 / (self.n_upper)
                + (
                    ((2 * (0.5 + np.arange(self.n_upper)) / self.n_upper) - 1)
                    * 180
                )
            )
            assert len(upper_az) == self.n_upper
            az = np.concatenate((az, upper_az))

        # Now, the lower ring
        if self.n_lower > 0:
            lower_az = normalize_az(
                self.rear_center_lower * 180 / (self.n_lower)
                + (
                    (2 * (0.5 + np.arange(self.n_lower) / self.n_lower) - 1)
                    * 180
                )
            )
            assert len(lower_az) == self.n_lower
            az = np.concatenate((az, lower_az))

        # Now, the Zenith
        if self.n_zenith:
            az = np.concatenate((az, [0]))

        # Now, the LFE
        az = np.concatenate((az, np.zeros(self.n_lfe)))
        az = np.mod(az + 179, 360) - 179
        assert len(az) == self.nchannel
        return az

    @property
    def el_deg_cfb(self):
        """Speaker elevations in CFB order"""
        return np.concatenate(
            (
                np.zeros(self.n_mid) + self.mid_el_deg,
                np.zeros(self.n_upper) + self.upper_el_deg,
                np.zeros(self.n_lower) + self.lower_el_deg,
                np.zeros(self.n_zenith) + 90,
                np.zeros(self.n_lfe) - 90,
            )
        )

    @property
    def from_cfb_order(self):
        """Permutation required to convert from CFB to this format"""
        cfb = list(range(self.nchannel))
        orderings = dict(
            cfb=cfb,
            wiring=self.wiring_order,
        )
        for k, v in self.extra_orderings.items():
            orderings[k] = [i - 1 for i in v]
        return orderings[self.ordering.value]

    @property
    def cfb_order(self):
        """Permutation required to convert from this format to CFB"""
        return np.argsort(self.from_cfb_order)

    @property
    def az_deg(self):
        return self.az_deg_cfb[self.from_cfb_order]

    @property
    def az(self):
        """Speaker azimuths (radians)"""
        return np.deg2rad(self.az_deg)

    @property
    def az_cfb(self):
        return np.deg2rad(self.az_deg_cfb)

    @property
    def el_deg(self):
        return self.el_deg_cfb[self.from_cfb_order]

    @property
    def el(self):
        """Speaker elevations (radians)"""
        return np.deg2rad(self.el_deg)

    @property
    def el_cfb(self):
        return np.deg2rad(self.el_deg_cfb)

    @property
    def nchannel(self):
        return (
            self.n_mid
            + self.n_upper
            + self.n_lower
            + self.n_zenith
            + self.n_lfe
        )

    @cached_property
    def dlb_name(self):
        """Speaker names as per SpeakerFormatDescription.m"""
        if self.is_binaural:
            return ["Lb", "Rb"]
        names = []
        lfe_count = 0
        for (az, el) in zip(self.az_deg, self.el_deg):
            name = self.speaker_name(az, el)
            if el == -90:
                lfe_count += 1
                if lfe_count > 1:
                    name += str(lfe_count)
            names.append(name)
        return names

    def __len__(self):
        return self.nchannel

    def is_superset_of(self, subset, strict=False):
        if self.is_binaural:
            return not strict and subset.is_binaural
        if len(subset) > len(self):
            return False
        if strict and len(subset) == len(self):
            return False
        A = self.get_map_from(subset, cfb=True)
        return (
            np.all(np.logical_or(A == 0, A == 1))
            and np.all(np.sum(A, 0) == 1)
            and np.all(np.sum(A, 1) <= 1)
        )

    def __repr__(self):
        if self._short_name is not None:
            return f"SpeakerChannelFormat('{self._short_name}')"
        return (
            f"SpeakerChannelFormat('{self.canonical_name}',"
            f" ordering='{self.ordering.value}')"
        )

    def azel_to_pan(self, az, el, compute_lr=True):
        """
        Source azimuth and elevation to speaker panning gain.
        Shamelessly stolen from `Pan_Spkr.m <https://swarm.dolby.net/files/depot/rd/r/MatlabCommon/Pan_Spkr.m>`_.

        .. plotly::
            :include-source:

            from spatpy import placement
            from spatpy.plot import PolarPlotter
            from spatpy.speakers import SpeakerChannelFormat

            sources = placement.circular_array(1000)
            y = SpeakerChannelFormat('5.0').azel_to_pan(sources.az, sources.el)
            PolarPlotter(trace_dim="bed_name").plot(np.abs(y))
        """
        semiPwrBoost = lambda x: x + 0.5 * x * (1 - x)
        Zpan = np.zeros(self.nchannel)
        Npan = np.zeros(self.nchannel)
        if compute_lr:
            PanTopLR = self.azel_to_pan(
                az=np.deg2rad([90, -90]),
                el=np.deg2rad([self.upper_el_deg, self.upper_el_deg]),
                compute_lr=False,
            )
            PanBotLR = self.azel_to_pan(
                az=np.deg2rad([90, -90]),
                el=np.deg2rad([self.lower_el_deg, self.lower_el_deg]),
                compute_lr=False,
            )
            Zpan = np.sum(PanTopLR, 1)
            Zpan /= np.sum(np.abs(Zpan) ** np.sqrt(2)) ** np.sqrt(0.5)
            Npan = np.sum(PanBotLR, 1)
            Npan /= np.sum(np.abs(Npan) ** np.sqrt(2)) ** np.sqrt(0.5)

        x, y, z = sph2cart(az=az, el=el)
        z2 = z * abs(z)
        gainZ = (z2 > 0.5) * (2 * z2 - 1) ** 2
        gainN = (z2 < -0.5) * (-2 * z2 - 1) ** 2
        gainU = np.minimum(1 - gainZ, np.maximum(0, np.sqrt(2) * z))
        gainL = np.minimum(1 - gainN, np.maximum(0, -np.sqrt(2) * z))
        gainM = np.maximum(0, 1 - np.sqrt(2) * np.abs(z))

        G = np.expand_dims(Zpan, 1) * np.atleast_2d(
            (gainZ - 0.25 * gainZ * (1 - gainZ))
        )
        G += np.expand_dims(Npan, 1) * np.atleast_2d(
            (gainN - 0.25 * gainN * (1 - gainN))
        )

        i = 0
        Mchans = np.arange(i + self.n_mid)
        i += self.n_mid
        Uchans = np.arange(i, i + self.n_upper)
        i += self.n_upper
        Lchans = np.arange(i, i + self.n_lower)
        i += self.n_lower

        if self.n_upper == 0:
            gainM += gainU
        else:
            Uangs = self.az_cfb[Uchans]
            for (i, a) in enumerate(az):
                G[Uchans, i] += semiPwrBoost(
                    interp_angle(Uangs, a)
                ) * semiPwrBoost(gainU[i])

        if self.n_lower == 0:
            gainM += gainL
        else:
            Langs = self.az_cfb[Lchans]
            for (i, a) in enumerate(az):
                G[Lchans, i] += semiPwrBoost(
                    interp_angle(Langs, a)
                ) * semiPwrBoost(gainL[i])

        Mangs = self.az_cfb[Mchans]
        for (i, a) in enumerate(az):
            G[Mchans, i] += semiPwrBoost(interp_angle(Mangs, a)) * semiPwrBoost(
                gainM[i]
            )
        coords = self.coords
        coords.update(source_coords(az, el))
        pan = xr.DataArray(
            G,
            dims=(self.dim_name, "source"),
            coords=coords,
            attrs=dict(format=self.short_name),
        )
        return pan

    def azel_to_hoa_decode(
        self,
        order,
        az,
        el,
        z_priority: Optional[float] = None,
        corner_priority: bool = True,
        power_law: Optional[float] = None,
    ):
        """Return speaker panning target suited for approximating with ambisonics.

        Shamelessly stolen from `Make_HOA_Decode.m <https://swarm.dolby.net/files/depot/rd/r/MatlabCommon/Make_HOA_Decode.m>`_.

        See also :obj:`spatpy.ambisonics.AmbisonicChannelFormat.speaker_mix_matrix`.

        .. plotly::
            :include-source:

            from spatpy import placement
            from spatpy.plot import PolarPlotter
            from spatpy.speakers import SpeakerChannelFormat

            sources = placement.circular_array(1000)
            y = SpeakerChannelFormat('5.0').azel_to_hoa_decode(3, sources.az, sources.el)
            PolarPlotter(trace_dim="bed_name").plot(np.abs(y))
        """
        if z_priority is None:
            z_priority = 1.0
        if power_law is None:
            power_law = np.sqrt(2)

        speakers = PointCloud.from_spherical(
            az=self.az[np.logical_not(self.is_lfe)],
            el=self.el[np.logical_not(self.is_lfe)],
        )
        points = speakers.points_xr
        Spkr_Unit_Vecs = np.vstack((points.x, points.y, points.z))

        # Some speakers will get higher 'priority' in the rendering:
        # Start by setting all priorities to zero
        SpeakerPriority = np.zeros(self.nchannel - self.n_lfe)
        if corner_priority:
            # Look through each plane:
            for spkel in [-45, 0, 45]:
                # Now, if there is a speaker near each corner, we give it higher priority
                for spkaz in [45, -45, 135, -135]:
                    # find the speaker closest to this position
                    pos = Point.from_spherical(az=spkaz, el=spkel, deg=True)
                    dotprod = [pos.dot(spk) for spk in speakers]
                    closest_speaker = np.argmax(dotprod)
                    # if the angular error is less than 30 degrees, give this speaker priority
                    if np.rad2deg(np.arccos(dotprod[closest_speaker])) < 30:
                        SpeakerPriority[closest_speaker] = 4

            SpeakerPriority[Spkr_Unit_Vecs[0, :] > 0.9999] = 2

        p = PointCloud.from_spherical(az=az, el=el)
        SpreadPoints = np.stack((p.x, p.y, p.z))
        nSP = SpreadPoints.shape[1]

        BinarySpkrPan = np.zeros((self.nchannel - self.n_lfe, nSP))
        DistanceResolution = 0.125 * 0.5 * np.pi / (1 * order + 3)
        for s in range(nSP):
            VectorsToSpeakers = SpreadPoints[:, s : (s + 1)] - Spkr_Unit_Vecs

            DistanceToSpeakers = (
                np.sqrt(np.sum(VectorsToSpeakers**2, 0)) / DistanceResolution
            )
            DistanceSuppress = np.minimum(
                1, DistanceToSpeakers**SpeakerPriority
            )
            BiasedDist = DistanceSuppress * np.sqrt(
                sum(
                    (np.diag([1, 1, z_priority]) @ VectorsToSpeakers) ** 2,
                    0,
                )
            )
            MinDist = min(BiasedDist)

            # Algorithm tweeked - 2018-03-16
            # Old method was a bit inaccurate:
            # tmp = max(0, 0.001 + MinDist - BiasedDist);
            tmp = MinDist + 0.001 > BiasedDist

            BinarySpkrPan[:, s] = tmp.T / np.sum(tmp)

        SmoothPanPoints = SpreadPoints
        TargetSmoothPan = np.zeros((self.nchannel - self.n_lfe, nSP))
        for s in range(nSP):
            DistanceToSpeakers = np.sqrt(
                np.sum(
                    (SmoothPanPoints[:, s : (s + 1)] - Spkr_Unit_Vecs) ** 2, 0
                )
            )
            DistanceToSpreadPoints = np.sqrt(
                np.sum(
                    (SmoothPanPoints[:, s : (s + 1)] - SmoothPanPoints) ** 2, 0
                )
            )
            PointSpreadDist = min(0.125 * DistanceToSpeakers)
            PointSpreadDist = max(DistanceResolution, PointSpreadDist)
            # SpreadPointsChoose = max(0,min(1,0.5+1000*(PointSpreadDist-DistanceToSpreadPoints)));
            SpreadPointsChoose = np.cos(
                np.pi
                / 2
                * np.minimum(1, DistanceToSpreadPoints / PointSpreadDist)
            )
            TargetSmoothPan[:, s] = (
                BinarySpkrPan
                @ SpreadPointsChoose.T
                / np.sum(SpreadPointsChoose)
            )
        TargetSmoothPan = TargetSmoothPan ** (1 / power_law)
        TargetSmoothPan_LFE = np.zeros((self.nchannel, nSP))
        TargetSmoothPan_LFE[np.logical_not(self.is_lfe), :] = TargetSmoothPan
        coords = self.coords
        coords.update(source_coords(az, el))

        pan = xr.DataArray(
            TargetSmoothPan_LFE,
            dims=[self.dim_name, "source"],
            coords=coords,
            attrs=dict(format=self.short_name),
        )
        return pan

    def get_map_from(self, subset, cfb=False):
        """
        Get a matrix to convert between channel formats.

        Args:
            subset (SpeakerChannelFormat): channel format to be mapped.

        Returns:
            A mix matrix which maps channels of ``subset`` to this format.

        .. code-block:: python

            from spatpy.speakers import SpeakerChannelFormat

            s1 = SpeakerChannelFormat('5.1')
            # ['C', 'L', 'R', 'L110', 'R110', 'LFE']
            print(s1.dlb_name)

            s2 = SpeakerChannelFormat('4')
            # ['C', 'L', 'R', 'BC']
            print(s2.dlb_name)

            print(s1.get_map_from(s2))
            # [[1.  0.  0.  0. ]
            # [0.  1.  0.  0. ]
            # [0.  0.  1.  0. ]
            # [0.  0.  0.  0.5]
            # [0.  0.  0.  0.5]
            # [0.  0.  0.  0. ]]

        """
        if cfb:
            az, el = self.az_deg_cfb, self.el_deg_cfb
            ss_az, ss_el = subset.az_deg_cfb, subset.el_deg_cfb
        else:
            az, el = self.az_deg, self.el_deg
            ss_az, ss_el = subset.az_deg, subset.el_deg
        n = self.nchannel
        m = subset.nchannel

        A = np.zeros((n, m))

        # step through each channel of <subS>, and find the nearest channel
        # in <this>
        for c in range(m):
            (same_plane,) = np.nonzero(el == ss_el[c])
            if len(same_plane) == 0:
                continue
            dist = np.abs(np.mod(180 + az - ss_az[c], 360) - 180)
            min_dist = np.min(dist[same_plane])
            (ch,) = np.nonzero(np.isclose(dist, min_dist))
            ch = [i for i in ch if i in same_plane]
            assert len(ch) in (1, 2), "logical error"
            if el[ch[0]] == -90:
                # it's an LFE chan. Match to only one:
                ch = [ch[0]]

            if len(ch) == 1:
                # Good. There is only one channel in <this> that is closest.
                A[ch, c] = 1
            elif np.mod(ss_az[c], 180) == 0:
                # oops, we found a ctr chan with a pair of matching buddies
                A[ch, c] = 0.5  # select both off-center chans
            else:
                # we have a non-ctr chan in <subS> that is close to two
                # channels in <this>
                closest_to_90 = np.argmin(np.abs(az[ch] - 90))
                A[
                    ch[closest_to_90], c
                ] = 1  # select one chan, closest to 90 deg
        return A

    @cached_property
    def wiring_order(self):
        """
        WaveFileOrder (suggested ordering for WaveFiles and for wiring speaker arrays).
        A reordering can be made to WaveFileFormat, which is defined according to
        the following process:

            #.  A list of channels, in ChannelFrontBackOrder (CFBO) is first constructed
            #.  A new (initially empty) list is made, to hold the WaveFileOrder.
            #.  The L/R channels are 'taken' from the CFBO list, and appended to
                the WaveFileOrder list
            #.  Now, we take channels from the front of the CFBO list, and append
                them to the WaveFileOrder list, using the following rules (rules
                chosen to attempt to keep all channels in pairs):

                #. If the first 2 items in the CFBO list are a Left/Right
                   symmetric pair, then these are transfered as a pair to the
                   end of the WaveFileOrder list
                #. If the first item in the CFBO listis a 'lone' channel (a
                   center-front, center-back, or LFE channel), then we take this
                   channel, as well as another 'lone' channel from the CFBO list.
                   If there are any LFE channels in the CFBO list, We take the
                   first LFE channel. Otherwise, we take the first 'lone' channel
                   of any kind (it will be a center-front or center-back
                   channel). If no other 'lone' channels are left to pair with
                   the one form the front of the CFBO list, then we have leave
                   that guy un-paired"""
        az = self.az_deg_cfb
        el = self.el_deg_cfb
        is_center = np.mod(az, 180) == 0
        is_lfe = el == -90
        ch_num = np.arange(self.nchannel)
        cfbo = [ch for ch in range(self.nchannel)]
        wfo = []

        # What speaker-arrangements are the Common-formats that should be used
        # as the basis of larger formats (for example, if a speaker array is a
        # super-set of 7.1, then it should contain 7.1, with correct 7.1
        # channel ordering, in it's first 8 channels)
        common_subsets = (
            SpeakerChannelFormat("3/4/4.1"),
            SpeakerChannelFormat("3/4.1"),
            SpeakerChannelFormat("3/2.1"),
            SpeakerChannelFormat("2"),
        )
        ss = [s for s in common_subsets if self.is_superset_of(s, strict=True)]
        if ss:
            ss = ss[0]
            A = self.get_map_from(ss, cfb=True)
            A = A[:, ss.wiring_order]
            # Find the speakers, in our set, that most-closely match with the
            # speakers in the selected sub-set;
            matching_channels, ss_wfo = np.nonzero(A == 1)
            perm = np.argsort(ss_wfo)
            wfo = matching_channels[perm].tolist()
            cfbo = [c for c in cfbo if c not in wfo]

        # Now go through in order:
        while len(cfbo) > 0:
            ch = cfbo[0]
            inds = [ch]
            not_assigned = np.array([c not in wfo for c in ch_num])
            if is_center[ch]:
                # We have a ctr speaker
                # Look for an LFE channel
                tmp = None
                other_lfe = not_assigned & is_lfe & (ch_num != ch)
                other_center = not_assigned & is_center & (ch_num != ch)
                if np.any(other_lfe):
                    tmp = ch_num[other_lfe][0]
                elif np.any(other_center):
                    # We didn't find an LFE, so look for any ctr speaker
                    tmp = ch_num[other_center][0]

                # OK, we have another 'lone' speaker, so pair it up with the
                # first one:
                if tmp is not None:
                    inds.append(tmp)
                else:
                    # We didn't find any other 'lone' speaker.
                    # If this is the last speaker, we will use it. Otherwise,
                    # send it to the back of the line:
                    if len(cfbo) > 1:
                        # This is not the last speaker, so put it on the end of the
                        # list, and try again
                        cfbo = cfbo[1:] + [cfbo[0]]
                        continue

            assert all([i not in wfo for i in inds])
            wfo.extend(inds)
            cfbo = [c for c in cfbo if c not in inds]

        assert set(wfo) == set(range(self.nchannel))

        return np.array(wfo)

    @property
    def channels(self):
        return self.components

    @cached_property
    def components(self) -> List[SpeakerChannel]:
        ch = []
        wfo = self.wiring_order
        for i in range(self.nchannel):
            kwargs = dict(
                wiring=wfo[i],
                cfb=i,
                az=self.az[i],
                el=self.el[i],
                dlb_name=self.dlb_name[i],
            )
            for k, v in self.extra_orderings.items():
                kwargs[k] = v[i] - 1
            ch.append(SpeakerChannel(**kwargs))
        self.ordering.reorder(ch)
        return ch

    @property
    def dim_name(self):
        return self.short_name

    @property
    def isf_params(self) -> ISFParams:
        # Turn this speaker format into { NF,NS,NB,NT,NL,NZ } syntax for use with ISF
        NS = min(self.n_mid_back // 2, (self.n_mid_back * 2 + 3) // 5)
        return ISFParams(
            NF=self.n_mid_front - 2,
            NS=NS,
            NB=self.n_mid_back - 2 * NS,
            NT=-self.n_upper / 2
            if self.rear_center_upper
            else self.n_upper / 2,
            NL=self.n_lower,
            NZ=self.n_zenith,
        )

    @property
    def coords(self):
        """Coordinates of this layout suitable for using with xarrays"""
        cs = {
            self.dim_name: range(self.nchannel),
            "dlb_name": (self.dim_name, self.dlb_name),
            "spkaz": (self.dim_name, self.az),
            "spkel": (self.dim_name, self.el),
        }
        cols = ["wiring", "cfb", "lfe"] + list(self.extra_orderings.keys())
        components = self.components
        for col in cols:
            cs[col] = (self.dim_name, [getattr(c, col) for c in components])

        if self.bed_name:
            cs["bed_name"] = (self.dim_name, self.bed_name)

        return cs

    @property
    def components_xr(self):
        return xr.DataArray(
            np.arange(self.nchannel), dims=[self.dim_name], coords=self.coords
        )


SpeakerChannelFormat.KNOWN_FORMAT_DESCRIPTORS = dict()
for (enum_name, short_name, desc, extra_ordering) in [
    # Sand-box with 16 + 8 + 1 + LFE
    (
        "STOCKHOLM_23_1",
        "23.1S",
        "5/11/6c/0/1.1",
        dict(
            stockholm=[
                4,
                2,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                16,
                14,
                12,
                10,
                8,
                6,
                18,
                17,
                19,
                21,
                22,
                20,
                23,
                24,
            ],
        ),
    ),
    (
        "MPEGH_22_0",
        "22M",
        "5/5/8c/3/1",
        dict(
            mpegh22=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                22,
                19,
                20,
                21,
            ]
        ),
    ),
    (
        "NHK_22_2",
        "22.2N",
        "5/5/8c/3/1.2",
        dict(
            nhk=[
                4,
                5,
                1,
                23,
                8,
                9,
                2,
                3,
                10,
                24,
                6,
                7,
                12,
                13,
                11,
                22,
                16,
                17,
                14,
                15,
                18,
                19,
                20,
                21,
            ]
        ),
    ),
    (
        "STOCKHOLM_22_2",
        "22.2S",
        "3/7/8c/3/1.2",
        dict(
            stockholm=[
                2,
                3,
                1,
                23,
                6,
                7,
                8,
                9,
                12,
                13,
                16,
                17,
                14,
                15,
                11,
                18,
                22,
                24,
                20,
                21,
                19,
                10,
                4,
                5,
            ]
        ),
    ),
    ("SURROUND_22_2", "22.2", "5/5/8c/3/1.2", None),
    ("SURROUND_22_2_M", "22.2M", "3/7/8c/3/1.2", None),
    ("SURROUND_13_1", "13.1", "5/4/4.1", None),
    ("SURROUND_9_1_4", "9.1.4", None, None),
    ("SURROUND_9_1_2", "9.1.2", None, None),
    ("SURROUND_7_1_4", "7.1.4", None, None),
    ("SURROUND_7_1_2", "7.1.2", None, None),
    ("SURROUND_7_1_0", "7.1", None, None),
    ("SURROUND_5_1_4", "5.1.4", None, None),
    ("SURROUND_5_1_2", "5.1.2", None, None),
    ("SURROUND_5_1_0", "5.1", None, None),
    ("LEGACY_5_1_0", "5.1L", "3/2.1", dict(legacy=[2, 1, 3, 4, 5, 6])),
    ("SURROUND_5_0_0", "5", None, None),
    ("LEGACY_5_0_0", "5L", "3/2", dict(legacy=[2, 1, 3, 4, 5])),
    ("STEREO", "stereo", "2", None),
    ("BINAURAL", "binaural", "2", dict(ear=[1, 2])),
    ("MONO", "mono", "1", None),
]:
    kwargs = dict()
    ordering = None
    if extra_ordering is not None:
        k = list(extra_ordering.keys())[0]
        ordering = k
    SpeakerChannelFormat.KNOWN_FORMAT_DESCRIPTORS[short_name] = dict(
        descriptor=desc, extra_orderings=extra_ordering, ordering=ordering
    )
    setattr(SpeakerChannelFormat, enum_name, SpeakerChannelFormat(short_name))


if __name__ == "__main__":
    fmt = SpeakerChannelFormat("5.1L")
    print(fmt.dlb_name)
