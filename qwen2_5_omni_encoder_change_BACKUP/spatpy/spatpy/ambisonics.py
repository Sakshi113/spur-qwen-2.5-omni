from dataclasses import dataclass, field, InitVar
import numpy as np
from enum import IntEnum, Enum, auto
from typing import Optional, List, Tuple, Any
import xarray as xr
from spatpy.geometry import Point, PointCloud, cart2sph, source_coords, sph2cart
from spatpy.placement import SphereMesh
from spatpy.speakers import (
    ChannelFormat,
    NamedChannel,
    ChannelOrdering,
    SpeakerChannelFormat,
)
from spatpy.wxy import WXYTransform


@dataclass
class BasisChannelFormat(ChannelFormat):
    pass


FUMA_NAMES_BY_ACN = "WYZX" + "VTRSUQ" + "OMKLNP"


def calculate_wxyz_to_bh3100():
    d = np.sqrt(3) / 6
    a = np.sqrt(2) * np.sqrt(1 / 3 - d**2)
    b = np.sqrt(0.5) * np.sqrt(3) * d
    c = np.sqrt(6) / 3
    e = 2 * np.sqrt(3) * np.sqrt(1 / 3 - d**2)
    return (
        np.array(
            [
                [a, c, 0, -d],
                [a, -0.5 * c, np.sqrt(3) / 2 * c, -d],
                [a, -0.5 * c, -np.sqrt(3) / 2 * c, -d],
                [b, 0, 0, e],
            ]
        )
        / 2
    )


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


class AmbisonicChannelOrdering(ChannelOrdering):
    FUMA = "fuma"  #: Furse-Malham number
    ACN = "acn"  #: Ambisonic channel number
    SID = "sid"  #: Single index designation


@dataclass(frozen=True, eq=True)
class AmbisonicComponent(NamedChannel):
    """Ambisonic signal component"""

    l: int
    """Ambisonic order (harmonic polynomial degree) L"""

    m: int
    """Ambisonic index (harmonic polynomial order) M"""

    xyz_perm: Optional[str] = field(default_factory=lambda: "XYZ")
    """XYZ permutation, for naming"""

    @property
    def is_x(self) -> bool:
        """Whether this component contains an X term"""
        return self.m > 0

    @property
    def z_deg(self) -> int:
        """Degree of Z term"""
        return self.l - abs(self.m)

    @property
    def xy_deg(self) -> int:
        """Maximum degree of X and Y"""
        return abs(self.m)

    @property
    def x_deg(self) -> int:
        """Degree of X term"""
        return self.xy_deg * self.is_x

    @property
    def y_deg(self) -> int:
        """Degree of Y term"""
        return self.xy_deg * np.logical_not(self.is_x)

    @property
    def sid(self) -> int:
        """Single Index Designation number"""
        return (
            self.l * (self.l + 1)
            + (self.z_deg - self.xy_deg)
            + 1 * (self.y_deg > 0)
        )

    @property
    def acn(self) -> int:
        """Ambisonic Channel Number"""
        return self.l * (self.l + 1) + self.m

    @property
    def mpegh_name(self) -> str:
        """Name according to MPEG-H convention, e.g. '21-'"""
        return f"{self.l}{abs(self.m)}" + ("+" if self.m >= 0 else "-")

    @property
    def dlb_name(self) -> str:
        """Name according to DLB naming convention (human readable, represents polynomial terms)"""
        if self.l == 0:
            return "W"
        name = ""
        xyz_perm = self.xyz_perm
        if self.xy_deg > 0:
            name += xyz_perm[0] if self.is_x else xyz_perm[1]
            name += str(self.xy_deg) if self.xy_deg > 1 else ""
        if self.z_deg > 0:
            name += xyz_perm[2]
            name += str(self.z_deg) if self.z_deg > 1 else ""
        return name

    @property
    def fuma(self) -> int:
        """FuMa channel number"""
        if self.l == 0:
            return 0

        # special case
        if self.l == 1:
            return [2, 3, 1][self.m + 1]

        # general case, ordered by size of m
        am = abs(self.m)
        return self.l * self.l + (am > 0) * (2 * am - (0 if self.m < 0 else 1))

    @property
    def fuma_name(self):
        """Single letter FuMa name (only defined if order < 4)"""
        n = self.acn
        if n >= len(FUMA_NAMES_BY_ACN):
            return "-"
        return FUMA_NAMES_BY_ACN[n]

    @classmethod
    def all(cls, order):
        """Yields all components of the given order"""
        for l in range(order + 1):
            for m in range(-l, l + 1):
                yield cls(l, m)


class AmbisonicScaling(IntEnum):
    """Ambisonic scaling conventions"""

    UNSCALED = 0
    N3D = 1
    SN3D = 2
    FUMA = 3
    DXM = 4


class AmbisonicBackend:
    """Backend for ambisonic harmonic basis manipulations"""

    @classmethod
    def scale(cls, fn: AmbisonicScaling, ch: AmbisonicComponent):
        """Returns the scale factor to be applied to the given component according to the scaling function"""
        return {
            AmbisonicScaling.UNSCALED: cls.unscaled_scale,
            AmbisonicScaling.SN3D: cls.sn3d_scale,
            AmbisonicScaling.N3D: cls.n3d_scale,
            AmbisonicScaling.DXM: cls.dxm_scale,
            AmbisonicScaling.FUMA: cls.fuma_scale,
        }[fn](ch)

    @classmethod
    def unscaled_scale(cls, ch: AmbisonicComponent):
        return 1.0

    @classmethod
    def sn3d_scale(cls, ch: AmbisonicComponent):
        """SN3D scale factor"""
        return cls.sqrt(
            (1 if ch.m == 0 else 2)
            * cls.factorial(ch.l - abs(ch.m))
            / cls.factorial(ch.l + abs(ch.m))
        )

    @classmethod
    def n3d_scale(cls, ch: AmbisonicComponent):
        """N3D scale factor"""
        return cls.sn3d_scale(ch) * cls.sqrt(2 * ch.l + 1)

    @classmethod
    def dxm_scale(cls, ch: AmbisonicComponent):
        """DXM scale factor"""
        return (
            cls.sn3d_scale(ch)
            * max(cls.sqrt(2), 2**ch.l)
            * cls.factorial(ch.l)
            * cls.sqrt(
                (2 * ch.l + 1) * (ch.l + 1) / cls.factorial(2 * ch.l + 2)
            )
        )

    @classmethod
    def fuma_scale(cls, ch: AmbisonicComponent):
        """FuMa scale factor"""
        return (
            (1 / cls.sqrt(2))
            if (ch.l == 0 and ch.m == 0)
            else cls.dxm_scale(ch)
        )

    @classmethod
    def sph_y(cls, az, el, ch: AmbisonicComponent):
        """Return the (unscaled) spherical harmonic Y for the given component"""
        scale = cls.sin(-ch.m * az) if ch.m < 0 else cls.cos(ch.m * az)
        p = cls.assoc_legendre(ch.l, abs(ch.m), cls.sin(el))
        scale *= 1 if ch.m % 2 == 0 else -1
        return scale * p

    @classmethod
    def scaled_harmonics(
        cls,
        scale_fn: AmbisonicScaling,
        components: List[AmbisonicComponent],
        az,
        el,
    ) -> List[Tuple[Any, AmbisonicComponent]]:
        """Returns a list of ``(c, scaled sph_y(az, el, c))`` for each component ``c``."""
        result = []
        for ch in components:
            result.append((ch, cls.scale(scale_fn, ch) * cls.sph_y(az, el, ch)))
        return result


class SymbolicAmbisonics(AmbisonicBackend):
    """Sympy ambisonic backend. Useful for printing tables of basis functions."""

    from sympy import sin, cos, factorial, sqrt, assoc_legendre

    @classmethod
    def all_harmonics(cls, order):
        from sympy import (
            sin,
            cos,
            sqrt,
            symbols,
            simplify,
            Q,
            Mul,
            Pow,
            sympify,
        )

        az, el = symbols("az el", real=True)
        scale = dict()
        basis = dict()
        n3d_harmonics = cls.scaled_harmonics(
            AmbisonicScaling.N3D, AmbisonicComponent.all(order), az, el
        )
        for (ch, ex) in n3d_harmonics:
            ex = ex.rewrite(sin)
            ex = simplify(ex)
            ex = ex.refine(Q.positive(cos(el)))
            ex = ex.subs(sin(el) * cos(el), sin(2 * el) / 2)
            factors = ex.as_ordered_factors()
            s = sympify("1")
            basis[ch] = sympify("1")
            for f in factors:
                if not f.free_symbols:
                    s *= f
                else:
                    basis[ch] *= f
            scale[(ch, "N3D")] = simplify(s)
            for fn in ["SN3D", "DXM", "FUMA"]:
                ex = cls.scale(AmbisonicScaling[fn], ch) * cls.sph_y(az, el, ch)
                s = simplify(ex / basis[ch])
                s = s.refine(Q.positive(cos(el)))
                s = s.refine(Q.positive(sin(el)))
                s = simplify(s)
                s = s.subs(sqrt(2) / 2, Pow(sqrt(2), -1, evaluate=False))
                s = s.subs(sqrt(3) / 3, Pow(sqrt(3), -1, evaluate=False))
                s = s.subs(
                    2 * (sqrt(7) / 7),
                    Mul(2, Pow(sqrt(7), -1, evaluate=False), evaluate=False),
                )
                scale[(ch, fn)] = s

        return az, el, scale, basis

    @classmethod
    def basis_table(cls, order: int) -> str:
        """Return a latex string representing the table of basis functions of the given order"""
        from sympy import latex

        az, el, scale, basis = cls.all_harmonics(order)
        az.name = "\\varphi"
        el.name = "\\theta"
        headings = [textrm("ACN"), textrm("DLB"), textrm("Basis")]
        headings += [
            textrm(fn.name) for fn in AmbisonicScaling if fn.name != "UNSCALED"
        ]
        rows = []
        for ch in AmbisonicComponent.all(order):
            b = latex(basis[ch])
            row = [
                str(ch.acn),
                textrm(ch.dlb_name),
                str(b),
            ]
            for fn in AmbisonicScaling:
                if fn.name != "UNSCALED":
                    row.append(latex(scale[(ch, fn.name)]))
            rows.append(row)
        return format_table(headings, rows)


def textrm(s):
    return f"\\textrm{{{s}}}"


def format_table(headings, rows):
    max_col_width = [0 for _ in rows]
    for row in rows:
        for (i, c) in enumerate(row):
            max_col_width[i] = max(max_col_width[i], len(c))

    headings = [h.ljust(max_col_width[i]) for (i, h) in enumerate(headings)]
    s = "\\begin{array}{" + ":".join("c" * len(headings)) + "}\n"
    s += " & ".join(headings) + "\\\\\n"
    s += "\\hline\n"
    for row in rows:
        s += " & ".join(
            [c.ljust(max_col_width[i]) for (i, c) in enumerate(row)]
        )
        s += "\\\\\n"
    s += "\\end{array}"
    return s


class NumpyAmbisonics(AmbisonicBackend):
    """Numpy ambisonics backend. Useful for numerical computations."""

    from numpy import sin, cos, sqrt
    from scipy.special import lpmv, factorial

    @classmethod
    def assoc_legendre(cls, l, m, x, normalize=False):
        x = np.atleast_1d(x)
        p = np.zeros(len(x))
        for i in range(len(x)):
            p[i] = cls.lpmv(m, l, x[i])
            if normalize:
                p[i] *= (-1) ** m * np.sqrt(
                    (l + 1 / 2) * cls.factorial(l - m) / cls.factorial(l + m)
                )
        p = p.squeeze()
        if p.size == 1:
            return p.item()
        return p


@dataclass
class AmbisonicChannelFormat(BasisChannelFormat):
    """
    Set of ambisonic components with independent horizontal and vertical order, and optional ground plane.
    See MATLAB class BFormatVariableResolution.m in MatlabCommon.

    Has static member instances for common named Ambisonics formats.

    ================================ ======== ======= ============================
    Format                           Ordering Scaling Comments
    ================================ ======== ======= ============================
    WXY, WXYZ                        FuMa     FuMa
    HOA1F, HOA2F, HOA3F              FuMa     FuMa    Suited to analog recordings
    BF1, BF2, BF3, ...               SID      DXM     Internal to Dolby
    BF1H, BF2H, BF3H, ...            SID      DXM     No vertical component
    HOA1, HOA2, HOA3, ...            ACN      N3D     Used by MPEG-H
    HOA1S (AmbiX), HOA2S, HOA3S, ... ACN      SN3D    Used by Facebook and Youtube
    ================================ ======== ======= ============================

    See `the Wikipedia page on Ambisonic formats <https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats>`_ for more info.
    """

    name: InitVar[str] = None
    horizontal_degree: Optional[int] = None
    vertical_degree: Optional[int] = None
    scaling: AmbisonicScaling = AmbisonicScaling.DXM
    ordering: AmbisonicChannelOrdering = AmbisonicChannelOrdering.SID

    @classmethod
    def drop_z(self):
        """Return an equivalent format without vertical components"""
        return AmbisonicChannelFormat(
            horizontal_degree=self.horizontal_degree,
            vertical_degree=0,
            scaling=self.scaling,
            ordering=self.ordering,
        )

    @classmethod
    def all_named_formats(cls, max_order=10):
        names = ["WXY", "WXYZ", "AMBIX"]
        for order in range(max_order):
            for stem, flavours in [("HOA", ("F", "S", "")), ("BF", ("H", ""))]:
                for flavour in flavours:
                    names.append(f"{stem}{order}{flavour}")
        return {name: cls(name) for name in names}

    @staticmethod
    def parse(name: str):
        name = name.upper()
        irregular_names = dict(
            AMBIX=(1, 1, "acn", "SN3D"),
            WXY=(1, 0, "sid", "DXM"),
            WXYZ=(1, 1, "sid", "DXM"),
        )
        args = irregular_names.get(name, None)
        if args is not None:
            hdeg, vdeg, ordering, scaling = args
            return dict(
                horizontal_degree=hdeg,
                vertical_degree=vdeg,
                ordering=AmbisonicChannelOrdering(ordering),
                scaling=AmbisonicScaling[scaling],
            )

        if not (name.startswith("BF") or name.startswith("HOA")):
            toks = name.split("|")
            if len(toks) >= 3:
                scaling = toks[-2]
                ordering = toks[-1]
                if toks[0].endswith("H"):
                    hdeg = int(toks[0][:-1])
                    vdeg = 0
                elif len(toks) == 4:
                    hdeg = int(toks[0])
                    vdeg = int(toks[1])
                else:
                    return None
                return dict(
                    horizontal_degree=hdeg,
                    vertical_degree=vdeg,
                    scaling=scaling,
                    ordering=ordering,
                )
            return None

        if name.startswith("HOA"):
            if name[-1] == "S":
                hdeg = int(name[3:-1])
                scaling = AmbisonicScaling.SN3D
                ordering = AmbisonicChannelOrdering.ACN
            elif name[-1] == "F":
                hdeg = int(name[3:-1])
                scaling = AmbisonicScaling.FUMA
                ordering = AmbisonicChannelOrdering.FUMA
            else:
                hdeg = int(name[3:])
                scaling = AmbisonicScaling.N3D
                ordering = AmbisonicChannelOrdering.ACN
            vdeg = hdeg
        else:
            assert name.startswith("BF")
            scaling = AmbisonicScaling.DXM
            ordering = AmbisonicChannelOrdering.SID
            if name[-1] == "H":
                hdeg = int(name[2:-1])
                vdeg = 0
            else:
                hdeg = int(name[2:])
                vdeg = hdeg

        return dict(
            horizontal_degree=hdeg,
            vertical_degree=vdeg,
            scaling=scaling,
            ordering=ordering,
        )

    @property
    def canonical_name(self) -> str:
        if (
            self.scaling == AmbisonicScaling.DXM
            and self.ordering == AmbisonicChannelOrdering.SID
        ):
            if self.horizontal_degree == self.vertical_degree:
                return f"BF{self.order}"
            elif self.vertical_degree == 0:
                return f"BF{self.horizontal_degree}H"

        if self.horizontal_degree == self.vertical_degree:
            suffix = {
                ("SN3D", "ACN"): "S",
                ("FUMA", "FUMA"): "F",
                ("N3D", "ACN"): "",
            }.get((self.scaling.name, self.ordering.name), None)
            if suffix is not None:
                return f"HOA{self.order}{suffix}"

        h = str(self.horizontal_degree)
        v = (
            ("|" + str(self.vertical_degree))
            if self.vertical_degree > 0
            else "H"
        )
        return f"{h}{v}|{self.scaling.name}|{self.ordering.name}"

    @property
    def short_name(self):
        return self.canonical_name

    def __eq__(self, other):
        if self.scaling != other.scaling:
            return False
        if self.ordering != other.ordering:
            return False
        if self.horizontal_degree != other.horizontal_degree:
            return False
        if self.vertical_degree != other.vertical_degree:
            return False
        return True

    @property
    def order(self) -> int:
        """Ambisonic 'order'"""
        return max(self.horizontal_degree, self.vertical_degree)

    def __post_init__(self, name):
        if name is not None:
            kwargs = self.parse(name)
            assert kwargs is not None, f"Unknown format: {name}"
            self.horizontal_degree = kwargs["horizontal_degree"]
            self.vertical_degree = kwargs["vertical_degree"]
            self.scaling = kwargs["scaling"]
            self.ordering = kwargs["ordering"]

        assert (
            self.horizontal_degree is not None
        ), "Either name or horizontal degree must be specified"

        if self.ordering is None:
            self.ordering = AmbisonicChannelOrdering.ACN

        if isinstance(self.ordering, str):
            self.ordering = AmbisonicChannelOrdering[self.ordering]

        if isinstance(self.scaling, str):
            self.scaling = AmbisonicScaling[self.scaling]

        self.backend = NumpyAmbisonics()
        if self.vertical_degree is None:
            self.vertical_degree = self.horizontal_degree

        self.ground_plane = False
        if self.vertical_degree < 0:
            self.vertical_degree = abs(self.vertical_degree)
            self.ground_plane = True

        self._mix_matrices = dict()

    @property
    def components(self) -> List[AmbisonicComponent]:
        """A list of ambisonic components corresponding to this format"""
        hdeg, vdeg = self.horizontal_degree, self.vertical_degree
        result = []
        for l in range(max(hdeg, vdeg) + 1):
            m = min(l, hdeg)
            while m >= 0:
                if self.ground_plane:
                    if (l - m) % 2 != 0 or (l > vdeg and l != m):
                        m -= 1
                        continue
                elif (l - m) > vdeg:
                    m -= 1
                    continue
                if m == 0:
                    result.append(AmbisonicComponent(l, m))
                else:
                    result.append(AmbisonicComponent(l, m))
                    result.append(AmbisonicComponent(l, -m))
                m -= 1
        self.ordering.reorder(result)
        return result

    @property
    def nchannel(self) -> int:
        return len(self.components)

    def component_scale(self, component):
        return self.backend.scale(self.scaling, component)

    def rescale_from(self, other) -> np.ndarray:
        """
        Return the scaling required to convert the components from another format to this format,
        assuming the components are ordered according to this format.
        """
        # assert self.horizontal_degree == other.horizontal_degree
        # assert (self.vertical_degree == other.vertical_degree) or (
        #     self.vertical_degree == 0 or other.vertical_degree == 0
        # )
        return np.array(
            [
                self.component_scale(c) / other.component_scale(c)
                if c in self.components
                else 0
                for c in other.components
            ]
        )

    def rescale_to(self, other):
        """
        Return the scaling required to convert the components from this format to another format,
        assuming the components are ordered according to that format.
        """
        return other.rescale_from(self)

    def reorder_from(self, other):
        """
        Return the permutation required to order the components from another format according to this format.
        """
        perm = self.ordering.argsort(other.components)
        present = [other.components[i] in self.components for i in perm]
        return perm[present]

    def reorder_to(self, other):
        """
        Return the permutation required to order the components from this format according to another format.
        """
        return other.reorder_from(self)

    def reformat_from(self, x, x_format, channel_axis=-1):
        """
        Rescale and reorder a numpy array to this format from another format.
        """
        indices = self.reorder_from(x_format)
        x = x.take(indices=indices, axis=channel_axis)
        x *= self.rescale_from(x_format)
        # this only happens if we are reformating a WXY to WXYZ
        if x_format.nchannel < self.nchannel:
            x = np.hstack(
                (x, np.zeros((x.shape[0], self.nchannel - x_format.nchannel)))
            )
        return x

    def reformat_to(self, x, new_format, channel_axis=-1):
        """
        Rescale and reorder a numpy array from this format to another format.
        """
        return new_format.reformat_from(x, self, channel_axis=channel_axis)

    @property
    def dim_name(self):
        return self.canonical_name

    @property
    def coords(self):
        """Coordinates of this format suitable for using with xarrays"""
        components = self.components
        coords = {self.dim_name: range(len(components))}
        cols = [
            "l",
            "m",
            "acn",
            "sid",
            "x_deg",
            "y_deg",
            "z_deg",
            "mpegh_name",
            "dlb_name",
        ]
        if self.order < 4:
            cols.extend(["fuma_name", "fuma"])

        for col in cols:
            coords[col] = (self.dim_name, [getattr(c, col) for c in components])

        coords["scale"] = (
            self.dim_name,
            [self.component_scale(c) for c in components],
        )

        return coords

    @property
    def components_xr(self):
        return xr.DataArray(
            np.arange(self.nchannel), dims=[self.dim_name], coords=self.coords
        )

    def xyz_to_pan(self, x, y, z) -> xr.DataArray:
        r, az, el = cart2sph(x, y, z)
        return self.azel_to_pan(az, el)

    def azel_to_pan(self, az, el) -> xr.DataArray:
        """Sample the scaled ambisonic basis functions corresponding to this format
        at the given azimuths and elevations."""
        result = self.backend.scaled_harmonics(
            self.scaling, self.components, az, el
        )
        y = np.vstack([x for (_, x) in result])
        coords = self.coords
        coords.update(source_coords(az, el))
        z = xr.DataArray(
            y,
            dims=[self.dim_name, "source"],
            coords=coords,
            attrs={
                "long_name": "higher order B format",
                "format": self.short_name,
                "ambisonic_order": self.order,
                "horizontal_degree": self.horizontal_degree,
                "vertical_degree": self.vertical_degree,
                "ground_plane": self.ground_plane,
                "scaling": self.scaling.name,
                "ordering": self.ordering.name,
            },
        )
        return z

    def planar_mix_matrix(self, planar):
        A = self.ambisonic_mix_matrix(AmbisonicChannelFormat("BF1H"))
        A = A @ planar.xfm.A.T
        return A.T

    def ambisonic_mix_matrix(self, other):
        # handle the case where "self" or "other" has more components
        s1 = [1/c if c != 0 else 0 for c in other.rescale_to(self)]
        s2 = self.rescale_to(other)
        s = s1 if len(s1) > len(s2) else s2
        perm = self.reorder_to(other)
        A = np.diag(s)
        if len(s1) > len(s2):
            A = A[:, self.reorder_to(other)]
        else:
            A = A[other.reorder_from(self), :]
        return A

    def beehive_mix_matrix(self, beehive, is_atmos=False):
        # special case
        if self.order == 1:
            A = None
            if beehive.canonical_name == "BH3.1.0.0":
                fmt = AmbisonicChannelFormat.BF1
                A = WXYZ_to_BH3100
            elif beehive.canonical_name == "BH3.0.0.0":
                fmt = AmbisonicChannelFormat.BF1H
                A = WXY_to_BH3000
            if A is not None:
                scale = self.rescale_to(fmt)
                perm = self.reorder_to(fmt)
                return np.diag(scale[perm]) @ (A[perm, :])

        """See HOA_to_BeeHive.m"""
        S = SphereMesh(4)
        VS = (S.vertices[0, :], S.vertices[1, :], S.vertices[2, :])
        az = np.arange(360)
        VM = sph2cart(r=1.0, az=az, el=0)
        VU = sph2cart(r=1.0, az=az, el=beehive.ring_elevation_deg, deg=True)
        VL = sph2cart(r=1.0, az=az, el=-beehive.ring_elevation_deg, deg=True)
        VZ = (0, 0, 1)
        WS = np.ones(len(VS[0]))
        WM = np.ones(len(az))
        WU = np.ones(len(az))
        WL = np.ones(len(az))
        WZ = np.ones(1)
        W = (WS / np.sum(WS)) * 0.1
        W = np.hstack([W] + [wi / np.sum(wi) for wi in (WM, WU, WL, WZ)])
        x, y, z = np.array([]), np.array([]), np.array([])
        for v in [VS, VM, VU, VL, VZ]:
            vx, vy, vz = v
            x = np.hstack((x, vx))
            y = np.hstack((y, vy))
            z = np.hstack((z, vz))
        A = beehive.xyz_to_pan(x, y, z, is_atmos=is_atmos) * np.sqrt(np.abs(W))
        B = self.xyz_to_pan(x, y, z) * np.sqrt(np.abs(W))
        M = np.linalg.lstsq(A.T, B.T, rcond=None)[0]
        M[np.abs(M - 0) < 0.0002] = 0
        M[np.abs(M - 1) < 0.0001] = 1
        M[np.abs(M + 1) < 0.0001] = -1
        return M

    def speaker_mix_matrix(
        self, fmt: SpeakerChannelFormat, lfe: bool = True, **kwargs
    ):
        Sp = SphereMesh(3)
        SpreadPoints = Sp.vertices

        points = PointCloud.from_cartesian(
            x=SpreadPoints[0, :], y=SpreadPoints[1, :], z=SpreadPoints[2, :]
        )
        PanSphere = self.azel_to_pan(
            points.points_xr.az.values, points.points_xr.el.values
        )

        TargetSmoothPan = fmt.azel_to_hoa_decode(
            self.order, points.az, points.el, **kwargs
        )

        DecodeMat = (
            (TargetSmoothPan @ PanSphere.T) * 1 / np.sum(PanSphere**2, 1)
        )
        self._mix_matrices[fmt.canonical_name] = DecodeMat
        if not lfe:
            return DecodeMat[np.logical_not(fmt.is_lfe), :]
        return DecodeMat


_all_named_formats = AmbisonicChannelFormat.all_named_formats()
for (name, fmt) in _all_named_formats.items():
    setattr(AmbisonicChannelFormat, name, fmt)


def names_table(order: int) -> str:
    """Return a latex string representing the names of all ambisonic components of the given order."""
    headings = [
        textrm("ACN"),
        "l",
        "m",
        textrm("MPEG-H"),
        textrm("SID"),
        textrm("\\#FuMa"),
        textrm("FuMa"),
        textrm("DLB"),
    ]
    rows = []
    for ch in AmbisonicComponent.all(order):
        rows.append(
            [
                str(ch.acn),
                str(ch.l),
                str(ch.m),
                textrm(ch.mpegh_name),
                str(ch.sid),
                str(ch.fuma),
                textrm(ch.fuma_name),
                textrm(ch.dlb_name),
            ]
        )
    return format_table(headings, rows)


@dataclass
class PlanarChannelFormat(BasisChannelFormat):
    name: InitVar[str] = None

    def __post_init__(self, name):
        name = name.upper()
        self.short_name = name
        self.canonical_name = name
        self.xfm = WXYTransform[name]

    def __repr__(self):
        return f"PlanarChannelFormat('{self.canonical_name}')"

    @classmethod
    def parse(cls, s):
        if s.upper() in WXYTransform.__members__:
            return dict(name=s)
        return None

    @property
    def nchannel(self):
        return len(self.xfm.channel_names)

    @property
    def dim_name(self):
        return self.canonical_name

    @property
    def coords(self):
        return {
            self.dim_name: np.arange(self.nchannel),
            "dlb_name": (self.dim_name, self.xfm.channel_names),
        }


if __name__ == "__main__":
    from spatpy.beehive import BeehiveChannelFormat
    from spatpy.formats import string_to_channel_format

    mix_matrix = AmbisonicChannelFormat.HOA2.speaker_mix_matrix(
        SpeakerChannelFormat.SURROUND_7_1_4
    )
    bh_matrix = AmbisonicChannelFormat.HOA2.beehive_mix_matrix(
        BeehiveChannelFormat("BH9.5.0.1")
    )
    lrs = string_to_channel_format("LRS")
    planar_matrix = AmbisonicChannelFormat("HOA1").planar_mix_matrix(lrs)
    x = planar_matrix @ np.random.random((4, 100))
    # print(names_table(4))
    # print(SymbolicAmbisonics.basis_table(4))
    print()
