import numpy as np
import xarray as xr
from typing import Optional, Any
from dataclasses import dataclass, InitVar
from scipy.optimize import nnls
from scipy.signal import remez, freqz
from scipy.interpolate import interp1d
from spatpy.geometry import sph2cart, cart2sph, source_coords

from spatpy.ambisonics import BasisChannelFormat, AmbisonicChannelFormat
from spatpy.speakers import (
    NamedChannel,
    SpeakerChannelFormat,
)


@dataclass(eq=True, frozen=True)
class BeehiveChannel(NamedChannel):
    dlb_name: str
    az: float
    el: float


WXY_to_BH3000 = np.array(
    [
        [0.35355, 0.35355, 0.35355],
        [0.40825, -0.20412, -0.20412],
        [0.00000, 0.35355, -0.35355],
    ]
)

WXYZ_to_BH3100 = np.array(
    [
        [0.35355, 0.35355, 0.35355, 0.17678],
        [0.40825, -0.20412, -0.20412, 0.00000],
        [0.00000, 0.35355, -0.35355, 0.00000],
        [-0.14434, -0.14434, -0.14434, 0.86603],
    ]
)

# Define the X,Y,Z coords of the 'reference' speaker layout in Atmos,
# along with the desired azimuth angle (in the BeeHive ring) for each
# speaker. The 'reference' speaker layout is defined to contain enough
# speakers to fully define the mapping from Atmos coordinates to the
# BeeHive ring azimuths:
ATMOS_REF_CONFIG = np.array(
    [
        [0.5, 0.0, 0, 0 / 7],  #  C
        [0.0, 0.0, 0, 1 / 7],  #  L
        [1.0, 0.0, 0, 6 / 7],  #  R
        [0.0, 0.5, 0, 2 / 7],  #  Ls
        [1.0, 0.5, 0, 5 / 7],  #  Rs
        [0.0, 1.0, 0, 3 / 7],  #  Lsr
        [1.0, 1.0, 0, 4 / 7],  #  Rsr
        [0.2, 0.2, 1, 1 / 8],  #  Ltf
        [0.8, 0.2, 1, 7 / 8],  #  Rtf
        [0.2, 0.8, 1, 3 / 8],  #  Ltr
        [0.8, 0.8, 1, 5 / 8],  #  Rtr
        [0.0, 0.0, -1, 2 / 10],  #  Left  floor front
        [1.0, 0.0, -1, 8 / 10],  #  Right floor front
        [0.0, 1.0, -1, 4 / 10],  #  Left  floor rear
        [1.0, 1.0, -1, 6 / 10],  # Right floor rear
    ]
)


def makeIntermediateWeights(yMinMax, num, spanDegrees, order, fold):
    y = np.zeros(num)
    w = np.zeros(num)
    if num == 0:
        return y, w

    if num == 1:
        y[0] = yMinMax[1]
        w[0] = 1
        assert not fold
        return y, w

    OS = spanDegrees * (2 * order + 1) / 360
    derate = (min(0.99, OS)) ** 0.7
    derate2 = (min(1, OS / 2)) ** 0.7
    derate3 = (min(1, OS / 3)) ** 0.7
    y = yMinMax[0] + np.diff(yMinMax) * np.arange(num) / (num - 1)
    if num == 2:
        w = np.array([1, 1], dtype=np.float32)
    elif num == 3:
        w = np.array([1, derate2, 1])
    elif num == 4:
        w = np.array([1, derate3, derate3, 1])
    elif num % 2 == 1:
        # >=5, odd
        _, tmpw = makeIntermediateWeights(
            np.array([yMinMax[0], np.mean(yMinMax)]),
            (num + 1) // 2,
            spanDegrees / 2,
            order,
            0,
        )
        w = derate * np.concatenate((tmpw, tmpw[-2::-1]))
    else:
        # >=6, even
        _, tmpw1 = makeIntermediateWeights(
            y[0, 2], 3, spanDegrees * 2 / (num - 1), order, 0
        )
        _, tmpw2 = makeIntermediateWeights(
            y[2, num - 3],
            num - 4,
            spanDegrees * (num - 5) / (num - 1),
            order,
            0,
        )
        w = derate * np.concatenate((tmpw1[:-1], tmpw2, tmpw1[-2::-1]))

    w[0] = 1
    w[-1] = 1
    if fold != 0:
        m = num // 2
        if num % 2 == 1:
            inds = np.concatenate(
                ([m], np.arange(m - 1, -1, -1), np.arange(m + 1, num))
            )
        else:
            inds = np.concatenate((np.arange(m, -1, -1), np.arange(m + 1, num)))
        inds = inds[::fold]
        y = y[inds]
        w = w[inds]
    return y, w


@dataclass
class BeehiveRingFilterParams:
    order: int = 4
    pass_ripple: float = 0.05
    stop_ripple: Optional[float] = None
    pwr: float = 1.414

    def __post_init__(self):
        if self.stop_ripple is None:
            ripple_defaults = np.array(
                [0, 0.2, 0.085, 0.07, 0.05, 0.03, 0.025, 0.02]
            )
            self.stop_ripple = ripple_defaults[
                min(self.order, len(ripple_defaults))
            ]
        self.pass_ripple = np.minimum(self.pass_ripple, self.stop_ripple)

    def make_win(self, pass_angle, stop_angle):
        if pass_angle > 0:
            pass_gain = 1 - self.pass_ripple / 4
            pass_weight = 1 / self.pass_ripple
            stop_suppress = 1
        else:
            pass_gain = (1 - self.pass_ripple / 4) * (
                90 / (90 - pass_angle)
            ) ** 2
            pass_weight = 1 / self.pass_ripple * ((90 - pass_angle) / 90) ** 2
            stop_suppress = max(1, ((90 - pass_angle) / 90))
        stop_suppress *= max(1, stop_angle * (self.order * 2 + 1) / 360)
        post_stop_angle = min(179.9, stop_angle + 180 / self.order)

        filt_freqs = (
            np.array([0.0, max(pass_angle, 1), post_stop_angle, 180]) / 360
        )
        filt_gains = np.array([pass_gain, 0])
        stop_weight = 2 * stop_suppress / self.stop_ripple
        filt_weight = np.array(
            [
                pass_weight,
                stop_weight,
            ]
        )
        filt_weight /= np.max(filt_weight)

        if self.order > 1:
            proto = remez(
                numtaps=2 * self.order + 1,
                bands=filt_freqs,
                desired=filt_gains,
                weight=filt_weight,
            )
            w, z = freqz(proto, 1, fs=1)
            npass = np.sum(w <= filt_freqs[0])
            nstop = np.sum(w >= filt_freqs[1])
            ntrans = len(z) - (npass + nstop)
            desired = np.concatenate(
                (
                    np.zeros(npass) + filt_gains[0],
                    np.zeros(ntrans),
                    np.zeros(nstop) + filt_gains[1],
                )
            )
            weight = np.concatenate(
                (
                    np.zeros(npass) + filt_weight[0],
                    np.zeros(ntrans),
                    np.zeros(nstop) + filt_weight[1],
                )
            )
            err = np.max((np.abs(z) - desired) * weight)
            n = len(proto)
            win = 2 * proto[n // 2 :]
            win[0] /= 2
            assert not np.any(np.isnan(win))
        elif self.order == 1:
            tmp = np.sqrt((stop_ang - 0.9) / 360)
            tmp2 = min(tmp * 1.6, 1.06 - tmp)
            win = np.vstack([tmp, tmp2])
        else:
            win = np.ones(1)

        return win, err


class BeehiveRingFilterDesign:
    def __init__(
        self,
        params: BeehiveRingFilterParams,
        fraction,
        pass_angles,
        stop_angles,
    ):
        self.params = params
        self.fraction = fraction
        self.pass_angles = pass_angles
        self.stop_angles = stop_angles

    @classmethod
    def from_params(cls, params: BeehiveRingFilterParams):
        """
        Make_BeeHive_RingDecode - Compute decode coefs for speakers in one BH ring
        
        This class is used to compute the decoding coefficients for one BeeHive
        ring to a speaker, given the range of azimuth angles spanned by that
        speaker. A (slow) initialisation process is needed in the constructor,
        to pre-compute some useful configuration data, and this is then used to
        speed up the operation of the getDecode() method.
        """

        stop_angles = np.arange(179) + 1
        pass_angles = []
        fraction = []
        for stop_angle in stop_angles:
            if params.order == 0:
                win = stop_angle / 179
                pass_angle = 0
            else:
                pass_angle = stop_angle - 90 / params.order
                step = 10
                err = 0
                while abs(step) > 0.001 and pass_angle < stop_angle:
                    win, err = params.make_win(pass_angle, stop_angle)
                    if err > 1:
                        step = step / 4 - abs(step) * 3 / 4
                    else:
                        step = step / 4 + abs(step) * 3 / 4
                    if pass_angle < 0 and err <= 1:
                        break
                    pass_angle += step
            pass_angles.append(pass_angle)
            dec = np.real(
                np.atleast_2d(win)
                @ np.exp(
                    np.atleast_2d(np.arange(params.order + 1)).T
                    @ np.atleast_2d(1j * np.pi / 180 * np.arange(360))
                )
            )
            fraction.append(min(1.0, np.mean(np.abs(dec) ** params.pwr)))

        n = len(fraction)
        for k in range(n - 1):
            fraction[n - 2 - k] = min(
                fraction[n - 2 - k], fraction[n - 1 - k] * 0.99999
            )

        return cls(
            params=params,
            fraction=fraction,
            pass_angles=pass_angles,
            stop_angles=stop_angles,
        )

    def get_single_decode(
        self,
        start_angle: float,
        end_angle: float,
    ):
        # fprintf('Getting singleDecode for Angles %f to %f\n', StartAng, EndAng);
        frac = (end_angle - start_angle) / 360
        if frac > 0.99:
            dec = np.zeros(self.params.order + 1)
            dec[0] = 1
            return dec
        n = np.arange(len(self.fraction))
        frac_bounds = (self.fraction[0], self.fraction[-1])
        ind = interp1d(
            self.fraction,
            n,
            bounds_error=False,
            fill_value=(0, len(self.fraction) - 1),
        )(frac)
        win, _ = self.params.make_win(
            pass_angle=interp1d(
                n, self.pass_angles, bounds_error=False, fill_value=frac_bounds
            )(ind),
            stop_angle=interp1d(
                n, self.stop_angles, bounds_error=False, fill_value=frac_bounds
            )(ind),
        )
        dec = win * np.exp(
            -(end_angle + start_angle) * np.pi * 1j / 360
        ) ** np.arange(self.params.order + 1)
        return dec

    def get_ring_decode(self, azimuths, weights=None):
        # getRingDecode - compute the decode equations for entire ring
        # Azimuths = vector of azimuth angles (degrees)
        # Weights = optional vector giving weights to allow some speakers to
        #           be given precedence over others
        order = self.params.order
        if weights is None:
            weights = np.ones_like(azimuths)
        assert len(azimuths) == len(weights)
        IsAnchor = weights == np.max(weights)
        SpkrAzStart = 0 * azimuths
        SpkrAzEnd = 0 * azimuths
        for aChan in np.arange(len(weights))[IsAnchor]:
            # Enumerate all speakers in anticlockwise order from aChan:
            aAz = azimuths[aChan]
            antiClkSet = np.argsort(np.mod(azimuths - aAz - 1 / 256, 360))
            # find the set of all channels up til the next anchor channel after
            # this anchor (aChan), in an anticlockwise direction:
            n = antiClkSet[IsAnchor][0]
            chanSet = np.concatenate(([aChan], antiClkSet[:n]))
            AzWidth = (
                np.mod(azimuths[chanSet[-1]] - aAz - 1 / 256, 360) + 1 / 256
            )
            tmpWeights = weights[chanSet]
            tmpWeights[0] = tmpWeights[0] / 2
            tmpWeights[-1] = tmpWeights[-1] / 2
            fencePosts = aAz + AzWidth * np.cumsum(tmpWeights[:-1]) / np.sum(
                tmpWeights
            )
            SpkrAzEnd[chanSet[:-1]] = fencePosts
            SpkrAzStart[chanSet[1:]] = fencePosts
        SpkrAzEnd = SpkrAzEnd + 360 * (SpkrAzEnd <= SpkrAzStart)
        Dec = np.zeros((order + 1, len(azimuths)), dtype=np.complex64)
        for c in range(len(azimuths)):
            Dec[:, c] = self.get_single_decode(SpkrAzStart[c], SpkrAzEnd[c])
        # Now insert the inverse of the BFormat-to-VS matrix:
        BFh2VS = BeehiveChannelFormat.bfh2vs(2 * order + 1, order)
        Tmp = (
            np.linalg.inv(np.hstack([np.real(BFh2VS), -np.imag(BFh2VS[:, 1:])]))
            .astype(np.complex64)
            .T
        )
        VS2BFh = Tmp[: (order + 1), :]
        VS2BFh[1:, :] = VS2BFh[1:, :] + 1j * Tmp[(order + 1) :, :]
        RingDec = np.real(Dec.T @ VS2BFh)
        return RingDec


@dataclass
class BeehiveChannelFormat(BasisChannelFormat):
    name: InitVar[str] = None
    n_mid: int = 0
    n_upper: int = 0
    n_lower: int = 0
    n_zenith: int = 0
    ring_elevation_deg: int = 45

    def __post_init__(self, name=None):
        if name is not None:
            if not name.startswith("BH"):
                name = "BH" + name
            fields = self.parse(name)
            for k, v in fields.items():
                self.__setattr__(k, v)
        channels = []
        for (ring_name, n_ring, ring_el) in [
            ("L", self.n_lower, -self.ring_elevation_deg),
            ("M", self.n_mid, 0),
            ("U", self.n_upper, self.ring_elevation_deg),
        ]:
            az = np.linspace(-np.pi, np.pi, n_ring, endpoint=False)
            el = np.zeros_like(az) + np.deg2rad(ring_el)
            for i in range(n_ring):
                channels.append(
                    BeehiveChannel(
                        dlb_name=f"{ring_name}{i + 1}", az=az[i], el=el[i]
                    )
                )
        if self.n_zenith == 1:
            channels.append(BeehiveChannel(dlb_name="Z1", az=0, el=np.pi / 2))

        self.channels = channels

    @staticmethod
    def parse(name: str):
        name = name.upper()
        if not name.startswith("BH"):
            return None

        ring_elevation_deg = 45
        if "@" in name:
            toks = name.split("@")
            if len(toks) != 2 or not toks[1].isdigit():
                return None
            name = toks[0]
            ring_elevation_deg = int(toks[1])
        toks = name[2:].split(".")
        if len(toks) < 4:
            return None
        if not all([t.isdigit() for t in toks]):
            return None
        return dict(
            n_mid=int(toks[0]),
            n_upper=int(toks[1]),
            n_lower=int(toks[2]),
            n_zenith=int(toks[3]),
            ring_elevation_deg=ring_elevation_deg,
        )

    @property
    def az(self):
        return np.array([c.az for c in self.channels])

    @property
    def el(self):
        return np.array([c.el for c in self.channels])

    @property
    def dlb_name(self):
        return np.array([c.dlb_name for c in self.channels])

    @property
    def canonical_name(self):
        return f"BH{self.n_mid}.{self.n_upper}.{self.n_lower}.{self.n_zenith}"

    @property
    def short_name(self):
        return self.canonical_name

    @property
    def dim_name(self):
        return self.short_name

    @property
    def nchannel(self):
        return len(self.channels)

    @property
    def coords(self):
        """Coordinates of this layout suitable for using with xarrays"""
        cs = {
            self.dim_name: range(self.nchannel),
            "spkaz": (self.dim_name, self.az),
            "spkel": (self.dim_name, self.el),
            "dlb_name": (self.dim_name, self.dlb_name),
        }
        return cs

    @property
    def components_xr(self):
        return xr.DataArray(
            np.arange(self.nchannel), dims=[self.dim_name], coords=self.coords
        )

    @property
    def damf_objects(self):
        return [dict(ID=i, ISF=name) for (i, name) in enumerate(self.dlb_name)]

    def atmos_ref_bh_az(self, x, y, ring_z):
        # Set up some arrays for translating between Atmos Az and BH Az, for
        # M ring:
        ref = ATMOS_REF_CONFIG
        tmp = ref[ref[:, 2] == ring_z, :]
        sort_ind = np.argsort(np.mod(tmp[:, 3], 1))  # Sort ascending BH_Az
        ref_bh_az = 360 * tmp[sort_ind, 4]
        ref_at_az = np.mod(
            np.rad2deg(
                np.atan2(
                    np.deg2rad(
                        2 * tmp[sort_ind, 0] - 1, 2 * tmp[sort_ind, 1] - 1
                    )
                )
            ),
            360,
        )
        # For each object vector, compute the Azimuth (in Atmos square)
        at_az = np.rad2deg(np.atan2(x, y))
        # now figure out the relative azimuth of the reference speakers
        # in this ring:
        ref_relative_az = np.mod(ref_at_az - at_az, 360)
        # And find the 2 adjacent speakers in the ring (and their rel_Az)
        ind1 = np.argmin(ref_relative_az)
        ind2 = np.argmin(360 - ref_relative_az)
        az1, az2 = ref_relative_az[ind1], ref_relative_az[ind2]
        # Get the BH_Azimuths for these speakers
        bh_az1, bh_az2 = ref_bh_az[ind1], ref_bh_az[ind2]
        bh_az1 = bh_az2 + np.mod(bh_az1 - bh_az2, 360)  # fix wrap-around case
        bh_az = (bh_az1 * az2 + bh_az2 * az1) / (az1 + az2)
        return bh_az

    def bh_ring_encoding(self, nc, r, az, pwr=None, scale=None, ctr_gain=1.0):
        # Let's make tmpDec, the decode for all objects, assuming they are
        # all located on this elevation level, and they are projected to
        # the walls
        if scale is None:
            scale = 1.15
        if pwr is None:
            pwr = 1.4
        tmp_pwr = np.sqrt(pwr)
        nv = len(az)
        if nc == 1:
            ctr_dec = ((1 - r) * ctr_gain) ** (1 / tmp_pwr)
            tmp_dec = np.ones((nc, nv)) - ctr_dec
            return tmp_dec, ctr_dec

        bh_vec = np.exp(1j * az)
        if nc == 2:
            # special case - make LtRt pair
            tmp = np.sqrt(bh_vec)
            tmp_dec = np.array(
                [
                    [tmp * np.real(tmp * (1 - 1j)) * (r ** (1 / tmp_pwr))],
                    [tmp * np.real(tmp * (1 + 1j)) * (r ** (1 / tmp_pwr))],
                ]
            ) / np.sqrt(2)
            ctr_dec = np.array([(1 + 1j), (1 - 1j)] / 2) * (
                (1 - r) * ctr_gain
            ) ** (1 / tmp_pwr)
            return tmp_dec, ctr_dec
        order = (nc - 1) / 2
        bfh2vs = self.bfh2vs(nc, order)
        # now we can make the ring-encoding functions:
        ring_dec = np.real(
            bfh2vs @ np.stack([bh_vec**i for i in range(int(order) + 1)])
        )
        tmp_dec = ring_dec * np.tile(r ** (1 / tmp_pwr), (nc, 1))
        ctr_dec = (
            np.real(
                bfh2vs @ np.real(np.tile(1j ** np.arange(order + 1), (nv, 1))).T
            )
            * scale
        )
        ctr_dec *= np.tile(((1 - r) * ctr_gain) ** (1 / tmp_pwr), (nc, 1))
        return tmp_dec, ctr_dec

    def azel_to_pan(self, az, el, **kwargs):
        x, y, z = sph2cart(r=1.0, az=az, el=el)
        return self.xyz_to_pan(x, y, z, **kwargs)

    def xyz_to_pan(
        self, x, y, z, is_atmos=False, pwr=None, correction_gain=1.0
    ):
        # Special case : BH3.*.0.0 is made from a re-mix of Ambisonics
        nv = len(x)
        if (
            self.n_mid == 3
            and self.n_upper <= 1
            and self.n_lower == 0
            and self.n_zenith == 0
        ):
            wxyz = np.zeros((3 + self.n_upper, nv))
            wxyz[0, :] = np.ones(nv) * np.sqrt(0.5)
            wxyz[1, :] = x
            wxyz[2, :] = y
            if self.n_upper == 1:
                wxyz[3, :] = z
            A = WXYZ_to_BH3100 if self.n_upper == 1 else WXY_to_BH3000
            return A @ wxyz

        if pwr is None:
            pwr = 1.4

        # Project the unit sphere onto the unit cylinder
        # Start by stretching the sphere, so that the Upper ring lies at
        # Z=1,R=1
        ring_el = np.deg2rad(self.ring_elevation_deg)
        x /= np.cos(ring_el)
        y /= np.cos(ring_el)
        z /= np.sin(ring_el)
        r, az, el = cart2sph(x, y, z)
        stretch = 1 / np.maximum(1, np.maximum(np.abs(z), r))
        x *= stretch
        y *= stretch
        z *= stretch
        r = np.minimum(1, np.sqrt(x**2 + y**2))
        if self.n_lower == 0:
            z = np.maximum(0, z)
        if self.n_upper == 0:
            z = np.minimum(0, z)

        # define the panning between layers (Lower,Middle,Upper)
        gain_u = np.maximum(0, z)
        gain_l = np.maximum(0, -z)
        gain_m = 1 - (gain_u + gain_l)

        # find how close to the Left,Right,Front,Back walls we are (0..1)
        # 0 -> we are at the center of the room [X,Y] = [0.5,0.5], Z=any
        G = np.zeros((self.nchannel, nv))

        # Figure out encode for objects panned into the M Ring:
        i = 0
        if self.n_mid > 0:
            nc = self.n_mid
            bh_az = az if not is_atmos else self.atmos_ref_bh_az(x, y, ring_z=0)
            tmp_dec, ctr_dec = self.bh_ring_encoding(
                self.n_mid, r, bh_az, pwr=pwr
            )
            G[i : (i + nc), :] = tmp_dec + ctr_dec
            G[i : (i + nc), :] *= np.atleast_2d(gain_m ** (1 / pwr))
            i += nc

        # Figure out encode for objects panned into the L Ring:
        if self.n_lower > 0:
            bh_az = (
                az if not is_atmos else self.atmos_ref_bh_az(x, y, ring_z=-1)
            )
            nc = self.n_lower
            tmp_dec, ctr_dec = self.bh_ring_encoding(
                self.n_lower, r, bh_az, pwr=pwr
            )
            G[i : (i + nc), :] = tmp_dec + ctr_dec
            G[i : (i + nc), :] *= np.atleast_2d(gain_l ** (1 / pwr))
            i += nc

        # Figure out encode for objects panned into the U/Z Ring:
        if self.n_upper > 0:
            bh_az = (
                az if not is_atmos else self.atmos_ref_bh_az(x, y, ring_z=-1)
            )
            nc = self.n_upper
            ctr_gain = gain_u if self.n_zenith else 1.0
            tmp_dec, ctr_dec = self.bh_ring_encoding(
                self.n_upper, r, bh_az, pwr=pwr, ctr_gain=ctr_gain
            )
            if self.n_zenith:
                G[i : (i + nc), :] = tmp_dec
                G[(i + nc) : (i + nc + 1), :] = ((1 - r) * gain_u) ** (
                    1 / np.sqrt(pwr)
                )
            else:
                G[i : (i + nc), :] = tmp_dec + ctr_dec
            G[i : (i + nc), :] *= np.atleast_2d(gain_u ** (1 / pwr))
        G *= correction_gain
        nsource = G.shape[1]
        coords = self.coords
        coords.update(source_coords(az, el))
        G_xr = xr.DataArray(
            G,
            dims=[self.dim_name, "source"],
            coords=coords,
            attrs={
                "long_name": f"beehive {self.canonical_name}",
                "format": self.short_name,
            },
        )
        return G_xr

    def speaker_mix_matrix(
        self,
        fmt: SpeakerChannelFormat,
        lfe: bool = True,
    ):
        # BeeHive_Reference_Decode - get matrix to decode any BeeHive fmt to any spkrs

        # This is the uber-method that can decode ANY Beehive format to a
        # speaker array that is defined in terms of NF,NS,NB,NT,NL,NZ. This
        # is just like the NF,NS,NB,NT set defined for Harmony decode of BH,
        # with the addition of NL and NZ speakers.

        # If you restrict the params to { NL=0 & NZ=0 & NT>0 }, then
        # you will have a valid Harmony configuration.

        # The decode matrix produced by this method will not always match
        # perfectly with the matrix built by BeeHive_Release_Decoder_Example,
        # for speaker layouts with a number other than 7 speakers in the
        # horizontal ring, but the difference should be small.

        # The matrices used by BeeHive_Release_Decoder_Example are derived by
        # calling this method, so this method is considered to be the
        # 'reference' method, and BeeHive_Release_Decoder_Example implements
        # an approximation

        NF, NS, NB, NT, NL, NZ = fmt.isf_params
        DoShiftTop = NT < 0
        NT = abs(NT)

        NF_Vals = np.arange(6)
        NS_Vals = np.arange(7)
        NB_Vals = np.arange(6)
        NT_Vals = np.arange(11) / 2
        NL_Vals = np.arange(6)
        NZ_Vals = np.arange(2)

        M, U, L, Z = self.n_mid, self.n_upper, self.n_lower, self.n_zenith

        assert (M <= 15) and (U <= 9) and (L <= 9) and (Z <= 1)
        assert (
            (NF in NF_Vals)
            and (NS in NS_Vals)
            and (NB in NB_Vals)
            and (NT in NT_Vals)
            and (NL in NL_Vals)
            and (NZ in NZ_Vals)
        ), "Speaker configuration is illegal"

        NoInChans = M + U + L + Z
        No_M_Spkrs = NF + 2 + 2 * NS + NB
        No_U_Spkrs = int(2 * NT)  # NT should be a multiple of 0.5
        No_L_Spkrs = NL
        No_Z_Spkrs = NZ  # NZ should be 0 or 1
        NoOutChans = No_M_Spkrs + No_U_Spkrs + No_L_Spkrs + No_Z_Spkrs
        Dec = np.zeros((NoOutChans, NoInChans))
        for inputRing in "mulz":
            encodeRules = ATMOS_REF_CONFIG
            # which ring are we decoding from?
            if inputRing == "m":
                # input is encoded in middle ring
                ringOrder = (M - 1) / 2
                encodeRules = encodeRules[encodeRules[:, 2] == 0, :]
                decodeRing = "m"
                DecFirstCol = 0
            elif inputRing == "u":
                # input is encoded in upper ring
                ringOrder = (U - 1) / 2
                encodeRules = encodeRules[encodeRules[:, 2] == 1, :]
                decodeRing = "u" if No_U_Spkrs > 0 else "m"
                DecFirstCol = M
            elif inputRing == "l":
                # input is encoded in lower ring
                ringOrder = (L - 1) / 2
                encodeRules = encodeRules[encodeRules[:, 2] == -1, :]
                decodeRing = "l" if No_L_Spkrs > 0 else "m"
                DecFirstCol = M + U
            elif inputRing == "z":
                # input is encoded in upper ring
                ringOrder = (Z - 1) / 2
                encodeRules = np.zeros((0, 4))
                if No_Z_Spkrs > 0:
                    decodeRing = "z"
                elif No_U_Spkrs > 0:
                    decodeRing = "u"
                else:
                    decodeRing = "m"
                DecFirstCol = M + U + L
            # if number of input channels in this ring is zero, then Order will
            # be -0.5, and we skip this ring:
            if ringOrder < 0:
                continue

            SpkrXYZ = np.zeros((0, 3))
            SpkrWeight = np.array([])
            SpkrDecZero = np.array([])
            # which ring of speakers are we decoding to?
            if decodeRing == "m":
                DecFirstRow = 0
                y, w = makeIntermediateWeights(
                    [0, 1], NF + 2, 360 * 2 / 7, ringOrder, 1
                )
                z = np.zeros_like(w)
                z[-2:] = 1
                for s in range(len(y)):
                    SpkrXYZ = np.vstack([SpkrXYZ, [y[s], 0, 0]])
                    SpkrWeight = np.hstack([SpkrWeight, w[s]])
                    SpkrDecZero = np.hstack([SpkrDecZero, z[s]])
                zSet = [
                    [1],
                    [100, 0],
                    [0, 100, 0],
                    [0, 100, 0, 0],
                    [0, 0, 100, 0, 0],
                    [0, 0, 0, 100, 0, 0],
                ]
                [y, w] = makeIntermediateWeights(
                    [0, 1], NS + 1, 360 * 2 / 7, ringOrder, 0
                )
                y = y[1:]
                w = w[1:]
                z = zSet[NS]
                for s in range(len(y)):
                    SpkrXYZ = np.vstack([SpkrXYZ, [0, y[s], 0]])
                    SpkrXYZ = np.vstack([SpkrXYZ, [1, y[s], 1]])
                    SpkrWeight = np.hstack([SpkrWeight, w[s], w[s]])
                    SpkrDecZero = np.hstack([SpkrDecZero, z[s], z[s]])
                y, w = makeIntermediateWeights(
                    [0, 1], NB + 2, 360 / 7, ringOrder, -1
                )
                y = y[2:]
                w = w[2:]
                z = np.zeros((1, NB))
                # disp(num2str([y;y1;w;w1],' #.2f'));
                for s in range(len(y)):
                    SpkrXYZ = np.vstack([SpkrXYZ, [y[s], 1, 0]])
                    SpkrWeight = np.hstack([SpkrWeight, w[s]])
                    SpkrDecZero = np.hstack([SpkrDecZero, z[s]])
            elif decodeRing == "u":
                DecFirstRow = No_M_Spkrs
                if NT == round(NT) and not DoShiftTop:
                    zSet = [[1, 1], [0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0, 0]]
                    if NT % 2:
                        y, w = makeIntermediateWeights(
                            np.array([0.5, -0.5]),
                            int(NT + 1),
                            360 / 2,
                            ringOrder,
                            0,
                        )
                        w = w[:-1] * 0.99
                        w[0] = 1
                        y = y[:-1]
                    else:
                        y, w = makeIntermediateWeights(
                            np.array([0.5, -0.5]) * (NT - 1) / max(1, NT),
                            int(NT),
                            360 / 2 * (NT - 1) / max(1, NT),
                            ringOrder,
                            0,
                        )
                        w *= 0.99
                        w[0] = 1
                        w[-1] = 1
                    inds = np.argsort(np.mod(y, 1))
                    y = y[inds]
                    w = w[inds]
                    z = zSet[int(NT)]
                    for s in range(len(y)):
                        SpkrXYZ = np.vstack(
                            [SpkrXYZ, [0.2, y[s], 1], [0.8, y[s], 1]]
                        )
                        SpkrWeight = np.hstack([SpkrWeight, w[s], w[s]])
                        SpkrDecZero = np.hstack([SpkrDecZero, z[s], z[s]])
                else:
                    # added code to handle the non-standard Harmony modes
                    NumTop = 2 * NT
                    if DoShiftTop:
                        # top speakers rotated so there is a rear-ctr
                        Azs, w = makeIntermediateWeights(
                            [np.pi, -np.pi], NumTop + 1, 360, ringOrder, 1
                        )
                        Azs = Azs[:-1]
                        w = w[:-1]
                    else:
                        if NumTop % 2:
                            # Odd number of top speakers rotated so there is no rear-ctr
                            # speaker (but there is a front center)
                            Azs, w = makeIntermediateWeights(
                                [0, 2 * np.pi], NumTop + 1, 360, ringOrder, -1
                            )
                            Azs = Azs[1:]
                            w = w[1:]
                        else:
                            # Even number of top speakers rotated so there is no rear-ctr
                            # speaker
                            assert False, (
                                "This case should be handled by legacy Harmony"
                                " code"
                            )
                    SpkrX = np.cos(Azs)
                    SpkrY = np.sin(Azs)
                    z = np.abs(SpkrY) > 0.95 * np.max(np.abs(SpkrY))
                    SpkrXYZ = np.vstack(
                        [
                            SpkrXYZ,
                            [(1 - SpkrY) / 2, (1 - SpkrX) / 2, SpkrX * 0 + 1],
                        ]
                    )
                    SpkrWeight = np.hstack([SpkrWeight, w])
                    SpkrDecZero = np.hstack([SpkrDecZero, z])
            elif decodeRing == "l":
                DecFirstRow = No_M_Spkrs + No_U_Spkrs
                azSet = [
                    [45, -45],
                    [0, 45, -45],
                    [45, -45, 135, -135],
                    [0, 45, -45, 135, -135],
                ]
                wSet = [1, [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]]
                zSet = [1, [1, 1], [0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0, 0]]
                az = azSet[NL]
                w = wSet[NL]
                z = zSet[NL]
                for s in range(len(az)):
                    SpkrXYZ = np.vstack(
                        [
                            SpkrXYZ,
                            [
                                0.5 - 0.5 * np.sin(np.deg2rad(az[s])),
                                0.5 - 0.5 * np.cos(np.deg2rad(az[s])),
                                -1,
                            ],
                        ]
                    )
                    SpkrWeight = np.vstack([SpkrWeight, w[s]])
                    SpkrDecZero = np.vstack([SpkrDecZero, z[s]])
            elif decodeRing == "z":
                DecFirstRow = No_M_Spkrs + No_U_Spkrs + No_L_Spkrs
                # Decoding the Z input ring to the Z speaker is easy:
                assert ringOrder == 0
                SpkrXYZ = np.vstack([SpkrXYZ, [0.5, 0.5, 1]])
                SpkrWeight = np.hstack([SpkrWeight, [1]])
                SpkrDecZero = np.hstack([SpkrDecZero, [1]])

            # Now, we know the mapping that was used by an encoder to encode
            # signals into this ring (in the encodeRules array), and we know
            # the Atmos locations of the speakers in this ring, so we can now
            # map the speakers in this ring to azimuth angles in this BeeHive
            # ring. Note that the mapping function (from Atmos XY to Azimuth)
            # may be (slightly) different for each ring.
            SpkrAz = 0 * SpkrWeight
            refXY = encodeRules[:, 0:2] - np.tile(
                [0.5, 0.5], (encodeRules.shape[0], 1)
            )
            refXY /= np.sqrt(np.sum(refXY**2, 1, keepdims=True))
            if len(SpkrAz) == 1:
                SpkrAz = 0
            elif len(refXY) > 0:
                for s in range(len(SpkrAz)):
                    XY = SpkrXYZ[s, :2] - np.array([0.5, 0.5])
                    # Pretend we are decoding
                    tmpW, _ = nnls(refXY.T, XY)
                    SpkrAz[s] = np.sum(
                        180
                        / np.pi
                        * np.angle(
                            np.exp(2j * np.pi * encodeRules[:, 3]) * tmpW
                        )
                    )
                    # fprintf('Encoded spkr in ring <#s>, decoded to ring <#s> at az=#f\n', ...
                    #  inputRing, decodeRing, SpkrAz(s));

            # Now, we have gathered all the info about the ring, and it's
            # associated playback speakers, so we can make the ring-decoder and
            # insert it into the Dec matrix:
            NumRingSpeakers = SpkrXYZ.shape[0]
            params = BeehiveRingFilterParams(int(ringOrder))
            if ringOrder == 0:
                # Special case for 0th order ring (composed of one signal)
                # temporary code:
                tmpDec = np.expand_dims(
                    SpkrDecZero
                    / (np.sum(SpkrDecZero**params.pwr) ** (1 / params.pwr)),
                    1,
                )
            else:
                design = BeehiveRingFilterDesign.from_params(params)
                tmpDec = design.get_ring_decode(SpkrAz, SpkrWeight)
            Dec[
                DecFirstRow : (DecFirstRow + NumRingSpeakers),
                DecFirstCol : (DecFirstCol + 2 * int(ringOrder) + 1),
            ] = tmpDec
        Dec = np.vstack((Dec, np.zeros((fmt.n_lfe, Dec.shape[1]))))
        Dec = Dec[fmt.from_cfb_order, :]
        return Dec

    @staticmethod
    def bfh2vs(nchannel, max_order):
        if nchannel > 0:
            order = (nchannel - 1) // 2
            win = np.maximum(
                0, (order + 0.5 - (np.arange(max_order + 1))) ** 0.5
            )
            win[0] *= 0.5
            A = np.exp(
                (
                    (
                        -2j
                        * np.pi
                        * (np.atleast_2d(np.arange(nchannel) / nchannel)).T
                    )
                    @ np.atleast_2d(np.arange(max_order + 1))
                )
            ) @ np.diag(win)
            tmp = np.real(
                np.atleast_2d(A[0, :])
                @ np.exp(
                    np.atleast_2d(np.arange(max_order + 1)).T
                    @ np.atleast_2d(np.arange(360) * np.pi / 180 * 1j)
                )
            )
            A /= nchannel * np.mean(tmp)
        else:
            A = np.zeros((0, int(max_order) + 1))
        return A

    def bfh2ring(self):
        nch = self.nchannel
        assert nch >= 1 and nch % 2 == 1
        order = (nch - 1) // 2
        wxy2vs = AmbisonicChannelFormat.wxy_to_vs(nch, order)
        A = np.zeros((nch, nch))
        A[:, 0] = np.real(wxy2vs[:, 0])
        A[:, 1::2] = np.real(wxy2vs[:, 1:])
        A[:, 2::2] = -np.imag(
            wxy2vs[
                :,
                1:,
            ]
        )
        return A


if __name__ == "__main__":
    A = BeehiveChannelFormat("BH9.5.0.0.1").speaker_mix_matrix(
        SpeakerChannelFormat("7.1.4")
    )
    print(A)
