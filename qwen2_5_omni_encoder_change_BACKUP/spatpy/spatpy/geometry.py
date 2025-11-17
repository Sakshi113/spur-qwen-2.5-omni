import plotly.express as px
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Union
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from fractions import Fraction

def source_coords(az, el):
    return {
        "source": range(len(az)),
        "az": (
            "source",
            az,
            {"long_name": "source azimuth", "units": "radians"},
        ),
        "el": (
            "source",
            el,
            {"long_name": "source elevation", "units": "radians"},
        ),
    }

def triplot(points, simplices, polar=False, **kwargs):
    ntri = len(simplices)
    mesh_x = [
        p
        for i in range(2 * ntri)
        for p in (points[simplices[i // 2, :], 0] if i % 2 == 0 else [None])
    ]
    mesh_y = [
        p
        for i in range(2 * ntri)
        for p in (points[simplices[i // 2, :], 1] if i % 2 == 0 else [None])
    ]

    if polar:
        trace = go.Scatterpolar(
            theta=mesh_x, thetaunit="radians", r=mesh_y, **kwargs
        )
    else:
        trace = go.Scatter(x=mesh_x, y=mesh_y, **kwargs)
    return trace


def polarticks(nmin=-1, nmax=1, denominator=2, offset=0, **kwargs):
    vals = []
    text = []
    for i in range(nmin, nmax + 1):
        n = 1 if i == nmax else denominator
        for j in range(n):
            numerator = i * denominator + j
            vals.append(numerator * np.pi / denominator + offset)
            f = Fraction(numerator, denominator)
            if f.numerator == -1:
                s = "-"
            elif f.numerator == 1:
                s = ""
            else:
                s = str(f.numerator)
            if f.numerator != 0:
                s += "π"
            if f.denominator != 1:
                s += "/" + str(f.denominator)
            text.append(s)
    return dict(tickvals=vals, ticktext=text, tickmode="array", **kwargs)


def _isscalar(x):
    if isinstance(x, (np.ndarray, xr.DataArray)) and x.ndim == 0:
        return True
    return not isinstance(x, (list, tuple, np.ndarray, xr.DataArray))


def _veclen(*args):
    n = 0
    for arg in args:
        if not _isscalar(arg):
            n = len(arg)
    return n


def sphgrid(n_az=100, n_el=100):
    """
    Return a tuple of ``(az, el)`` formed by the cartesian product of
    linearly spaced azimuths and elevations of length ``n_az`` and ``n_el`` respectively.
    """
    az = np.linspace(-np.pi, np.pi + (2 * np.pi / n_az), n_az, endpoint=False)
    el = np.linspace(-np.pi / 2, np.pi / 2, n_el, endpoint=True)
    grid = np.dstack(np.meshgrid(az, el)).reshape(-1, 2)
    az = grid[:, 0]
    el = grid[:, 1]
    return (az, el)


def _xy_to_az(x, y):
    return np.arctan2(y, x)


def _zr_to_el(z, r):
    if _isscalar(r):
        if np.isclose(r, 0.0):
            return 0.0
        return np.arcsin(z / r)
    else:
        safe_r = np.logical_not(np.isclose(r, 0.0))
        tmp_el = np.zeros_like(z)
        tmp_el[safe_r] = np.arcsin(z[safe_r] / r[safe_r])
        return tmp_el


def _razel_to_xyz(r, az, el):
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return (x, y, z)


def sph2cart(
    r: Union[list, tuple, np.ndarray, xr.DataArray, float] = 0.0,
    az: Union[list, tuple, np.ndarray, xr.DataArray, float] = 0.0,
    el: Optional[Union[list, tuple, np.ndarray, xr.DataArray, float]] = None,
    deg: bool = False,
) -> Union[
    Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Convert spherical to cartesian coordinates. Arguments may be any combination of vectors or floating point numbers.

    Args:
        r: radius
        az: azimuth
        el: elevation, if ``None`` will default to :math:`\\frac{\\pi}{2}`
        deg: provided azimuth and elevation are in degrees (default is radians)

    Returns:
        A tuple of ``(x, y, z)``. If all arguments are scalar then all elements are scalar,
        otherwise they are all numpy arrays.
    """

    if el is None:
        el = 0.0
    n = _veclen(r, az, el)
    if n > 0:
        if _isscalar(r):
            r = np.zeros(n) + r
        if _isscalar(az):
            az = np.zeros(n) + az
        if _isscalar(el):
            el = np.zeros(n) + el

    if deg:
        az = np.deg2rad(az)
        el = np.deg2rad(el)

    x, y, z = _razel_to_xyz(r, az, el)

    return x, y, z


def cart2sph(
    x: Union[list, tuple, np.ndarray, xr.DataArray, float] = 0.0,
    y: Union[list, tuple, np.ndarray, xr.DataArray, float] = 0.0,
    z: Union[list, tuple, np.ndarray, xr.DataArray, float] = 0.0,
    deg: bool = False,
) -> Union[
    Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Convert cartesian to spherical coordinates. Arguments may be any combination of vectors or floating point numbers.

    Args:
        deg: Convert returned azimuth and elevation to degrees (default is radians).

    Returns:
        A tuple of ``(r, az, el)``. If all arguments are scalar then all elements are scalar,
        otherwise they are all numpy arrays.
    """
    n = _veclen(x, y, z)

    if n > 0:
        x = np.zeros(n) + x
        y = np.zeros(n) + y
        z = np.zeros(n) + z

    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-20)

    # azimuth is measured anti-clockwise from positive x direction
    az = _xy_to_az(x, y)
    # el = 0 corresponds to positive z direction, el = pi corresponds to negative z direction
    el = _zr_to_el(z, r)

    return r, np.rad2deg(az) if deg else az, np.rad2deg(el) if deg else el


def modulate_radius(p: xr.DataArray, r=None) -> xr.DataArray:
    """
    Keeping azimuth and elevation the same, change the radius of all points to ``r``.

    Args:
        p (DataArray): set of points to be modulated
        r (DataArray, optional): set of points to be modulated. Defaults to ``abs(p)``.
    """
    if r is None:
        r = np.abs(p).values
    x, y, z = sph2cart(r, p.az.values, p.el.values)
    dim = p.az.dims[0]
    p = p.assign_coords(x=(dim, x))
    p = p.assign_coords(y=(dim, y))
    p = p.assign_coords(z=(dim, z))
    p = p.assign_coords(r=(dim, r))
    return p


def scale_maxdim(p: xr.DataArray, limit) -> Tuple[xr.DataArray, float]:
    """
    Scale the set of points such that the max - min in the largest (x, y, z) dimension
    is ``limit``.

    Returns:
        A tuple of ``(scaled_points: xr.DataArray, scale_factor: float)``.
    """
    maxv = max(
        np.max(p.x) - np.min(p.x),
        np.max(p.y) - np.min(p.y),
        np.max(p.z) - np.min(p.z),
    ).item()
    scale = limit / maxv
    x = scale * p.x.values[:]
    y = scale * p.y.values[:]
    z = scale * p.z.values[:]

    kwargs = dict()
    if "tag" in p.coords:
        kwargs["tags"] = p.tag.values[:]
    return (
        PointCloud.from_cartesian(x=x, y=y, z=z, **kwargs).points_xr,
        scale,
    )


@dataclass(order=True, frozen=True)
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tag: str = field(default_factory=str, compare=False)

    @property
    def az(self):
        return _xy_to_az(self.x, self.y)

    @property
    def az_deg(self):
        return np.rad2deg(self.az)

    @property
    def el(self):
        return _zr_to_el(self.z, self.r)

    @property
    def el_deg(self):
        return np.rad2deg(self.el)

    @property
    def r(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @classmethod
    def from_spherical(cls, r=1.0, az=0.0, el=None, deg=False, tag=None):
        x, y, z = sph2cart(r, az, el, deg=deg)
        return cls(x, y, z, tag=tag)

    def to_spherical(self, deg=False):
        return (
            self.r,
            self.az_deg if deg else self.az,
            self.el_deg if deg else self.el,
        )

    def __add__(self, other):
        if other == 0:
            return self
        if isinstance(other, float):
            return Point(
                self.x + other, self.y + other, self.z + other, tag=self.tag
            )
        if isinstance(other, PointCloud):
            return other.__add__(self)
        return Point(
            self.x + other.x, self.y + other.y, self.z + other.z, tag=self.tag
        )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, PointCloud):
            return other.__rsub__(self)
        if isinstance(other, float):
            return self + (-1.0 * other)
        return Point(
            self.x - other.x, self.y - other.y, self.z - other.z, tag=self.tag
        )

    def __mul__(self, a):
        return Point(self.x * a, self.y * a, self.z * a, tag=self.tag)

    def __truediv__(self, a):
        return self * (1 / a)

    def r2_distance(self, other):
        return (self - other).r

    def rotate(self, az=0.0, el=0.0, deg=False):
        """
        Rotate by a set azimuth/elevation. See documentation for :obj:`PointCloud.rotate`.
        """
        az = self.az + (np.deg2rad(az) if deg else az)
        el = self.el + (np.deg2rad(el) if deg else el)
        return Point.from_spherical(self.r, az, el, tag=self.tag)

    def rotate_axis(self, axis, theta, deg=False):
        """
        Rotate around an axis. See documentation for :obj:`PointCloud.rotate_axis`.
        """

        if deg:
            theta = np.deg2rad(theta)
        (x, y, z) = (self.x, self.y, self.z)
        if axis == "x":
            y, z = y * np.cos(theta) - z * np.sin(theta), y * np.sin(
                theta
            ) + z * np.cos(theta)
        elif axis == "y":
            x, z = x * np.cos(theta) + z * np.sin(theta), -x * np.sin(
                theta
            ) + z * np.cos(theta)
        elif axis == "z":
            x, y = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(
                theta
            ) + y * np.cos(theta)
        return Point(x=x, y=y, z=z, tag=self.tag)


@dataclass
class PointCloud:
    points: Optional[List[Point]] = None
    points_xr: Optional[xr.DataArray] = None

    def __len__(self):
        return len(self.points_xr)

    def __getitem__(self, idx):
        return self.points_xr[idx]

    @property
    def x(self):
        return self.points_xr.x.values

    @property
    def y(self):
        return self.points_xr.y.values

    @property
    def z(self):
        return self.points_xr.z.values

    @property
    def r(self):
        return self.points_xr.r.values

    @property
    def intensity(self):
        return self.points_xr.r.values

    @property
    def az(self):
        return self.points_xr.az.values

    @property
    def az_deg(self):
        return np.rad2deg(self.az)

    @property
    def el(self):
        return self.points_xr.el.values

    @property
    def el_deg(self):
        return np.rad2deg(self.el)

    @property
    def tags(self):
        if "tag" not in self.points_xr.coords:
            return None
        return [t if t is not None else "" for t in self.points_xr.tag.values]

    def __post_init__(self):
        points = self.points
        if points is None:
            points = []
        if self.points_xr is not None:
            return
        fields = dict(
            tag=[], x=[], y=[], z=[], az=[], az_deg=[], el=[], el_deg=[], r=[]
        )
        for p in points:
            for k in fields.keys():
                fields[k].append(getattr(p, k))
        self.points_xr = xr.DataArray(
            np.arange(len(points)),
            dims=["point"],
            coords={
                "point": range(len(points)),
                "tag": ("point", fields["tag"]),
                "x": ("point", fields["x"], dict(units="metre")),
                "y": ("point", fields["y"], dict(units="metre")),
                "z": ("point", fields["z"], dict(units="metre")),
                "az": (
                    "point",
                    fields["az"],
                    dict(long_name="azimuth", symbol="φ", units="radian"),
                ),
                "az_deg": (
                    "point",
                    fields["az_deg"],
                    dict(long_name="azimuth", symbol="φ", units="degree"),
                ),
                "el": (
                    "point",
                    fields["el"],
                    dict(long_name="elevation", symbol="θ", units="radian"),
                ),
                "el_deg": (
                    "point",
                    fields["el_deg"],
                    dict(long_name="elevation", symbol="θ", units="degree"),
                ),
                "r": (
                    "point",
                    fields["r"],
                    dict(long_name="radius", symbol="ρ", units="metre"),
                ),
            },
        )

    @classmethod
    def from_response(cls, resp: xr.DataArray, mic_offset=None):
        p = modulate_radius(resp.copy(deep=True))

        if mic_offset is not None and "micx" in p.coords:
            p.x.values[:] = p.x.values[:] + (p.micx.item() * mic_offset)
            p.y.values[:] = p.y.values[:] + (p.micy.item() * mic_offset)
            p.z.values[:] = p.z.values[:] + (p.micz.item() * mic_offset)
        return cls(points_xr=p)

    @classmethod
    def from_spherical(cls, r=1.0, az=0.0, el=0.0, deg=False, tags=None):
        n = _veclen(r, az, el)

        if n == 0:
            return cls(
                points=[Point.from_spherical(r, az, el, deg=deg, tag=tags)]
            )

        if tags is None:
            tags = [None for _ in range(n)]

        if _isscalar(r):
            r = np.zeros(n) + r

        if _isscalar(az):
            az = np.zeros(n) + az

        if _isscalar(el):
            el = np.zeros(n) + el

        return cls(
            points=[
                Point.from_spherical(ri, ai, ei, deg=deg, tag=t)
                for (ri, ai, ei, t) in zip(r, az, el, tags)
            ]
        )

    @classmethod
    def randn(cls, npoints, tag_fmt=None):
        """Make ``npoints`` randomly distributed points."""
        if tag_fmt is None:
            tag_fmt = "p[{i}]"
        return cls.from_cartesian(
            x=np.random.randn(npoints),
            y=np.random.randn(npoints),
            z=np.random.randn(npoints),
            tags=[tag_fmt.format(i=i) for i in range(npoints)],
        )

    @classmethod
    def from_cartesian(cls, x=0.0, y=0.0, z=0.0, tags=None):
        n = 0
        if isinstance(x, (list, tuple, np.ndarray)):
            n = len(x)
        elif isinstance(y, (list, tuple, np.ndarray)):
            n = len(y)
        elif isinstance(y, (list, tuple, np.ndarray)):
            n = len(y)

        if n == 0:
            return cls(points=[Point(x, y, z, tag=tags)])

        if tags is None:
            tags = [None for _ in range(n)]

        if not isinstance(x, (list, tuple, np.ndarray)):
            x = [x for _ in range(n)]

        if not isinstance(y, (list, tuple, np.ndarray)):
            y = [y for _ in range(n)]

        if not isinstance(z, (list, tuple, np.ndarray)):
            z = [z for _ in range(n)]

        return cls(
            points=[
                Point(xi, yi, zi, tag=t)
                for (xi, yi, zi, t) in zip(x, y, z, tags)
            ]
        )

    def __add__(self, other):
        if isinstance(other, Point):
            return PointCloud([(p + other) for p in self.points])

        return PointCloud(
            [(p + q) for (p, q) in zip(self.points, other.points)]
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Point):
            return PointCloud([(p - other) for p in self.points])
        return PointCloud(
            [(p - q) for (p, q) in zip(self.points, other.points)]
        )

    def __mul__(self, a):
        return PointCloud([p * a for p in self.points])

    def __rmul__(self, a):
        return self.__mul__(a)

    def expand_by(self, a):
        return self.__mul__(a)

    def r2_distance(self, other):
        if isinstance(other, Point):
            return np.array([p.r2_distance(other) for p in self.points])
        return np.array(
            [p.r2_distance(q) for (p, q) in zip(self.points, other.points)]
        )

    def rotate(self, az=0.0, el=0.0, deg=False):
        """
        Rotate by a set azimuth/elevation

        .. plotly::
            :include-source:
            :context:

            from spatpy.geometry import PointCloud
            import plotly.graph_objects as go

            points = PointCloud.randn(5)
            go.Figure(points.scatterpolar(name="original"))

        .. plotly::
            :include-source:
            :context:

            rotated = points.rotate(az=90, el=45, deg=True)
            go.Figure(rotated.scatterpolar(name="rotated"))
        """
        return PointCloud(
            [p.rotate(az=az, el=el, deg=deg) for p in self.points]
        )

    def rotate_axis(self, axis, theta, deg=False):
        """
        Rotate around an axis.

        .. plotly::
            :include-source:
            :context: reset

            from spatpy.geometry import PointCloud
            from plotly import graph_objects as go

            points = PointCloud.randn(5)
            go.Figure(points.scatterpolar(name="original"))

        .. plotly::
            :include-source:
            :context:

            rotated = points.rotate_axis("z", theta=45.0, deg=True)
            go.Figure(rotated.scatterpolar(name="rotated around z axis"))
        """
        return PointCloud(
            [p.rotate_axis(axis, theta, deg=deg) for p in self.points]
        )

    def scale_maxdim(self, limit):
        """
        Scale the set of points such that the max - min in the largest (x, y, z) dimension
        is ``limit``.

        Returns:
            A tuple of ``(scaled_points: PointCloud, scale_factor: float)``.

        .. plotly::
            :include-source:
            :context: reset

            from spatpy.geometry import PointCloud
            import plotly.graph_objects as go

            points = PointCloud.randn(10)
            go.Figure(points.scatter(name="original"))

        .. plotly::
            :include-source:
            :context:

            scaled, scale_factor = points.scale_maxdim(limit=42)
            go.Figure(scaled.scatter(name="scaled"))
        """
        p, s = scale_maxdim(self.points_xr, limit)
        return PointCloud(points_xr=p), s

    def scale_radius(self, scale):
        """
        Return a new :obj:`PointCloud` with the radius of all points scaled.
        Pass a vector to scale each point individually, or a scalar to scale all points by the same amount.
        """
        p = self.points_xr
        return PointCloud.from_spherical(
            r=p.r * scale, az=p.az, el=p.el, tags=self.tags
        )

    def interpolate_grid(
        self, values, n=20, method=None, fill=True, fill_value=np.nan
    ):
        if method is None:
            method = "linear"
        x, y, z = np.mgrid[
            min(self.x) : max(self.x) : (n * 1j),
            min(self.y) : max(self.y) : (n * 1j),
            min(self.z) : max(self.z) : (n * 1j),
        ]
        interp = griddata(
            (self.x, self.y, self.z),
            values,
            (x, y, z),
            method=method,
            fill_value=fill_value,
        )
        interp, x, y, z = (
            interp.flatten(),
            x.flatten(),
            y.flatten(),
            z.flatten(),
        )
        if not fill:
            inds = np.logical_not(np.isnan(interp))
            interp = interp[inds]
            x = x[inds]
            y = y[inds]
            z = z[inds]
        return interp, PointCloud(x, y, z)

    def annotations(self, **kwargs):
        if self.tags is None:
            return []
        ann = []
        for (i, tag) in enumerate(self.tags):
            p = self.points[i]
            ann.append(
                dict(
                    showarrow=kwargs.pop("showarrow", False),
                    text=tag.item(),
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    **kwargs,
                )
            )
        return ann

    def scatter(self, **kwargs):
        """
        Produce a :obj:`go.Scatter3d` trace of the points with some sensible defaults.
        Any keyword arguments will be passed through to plotly.

        .. plotly::
            :include-source:

            from spatpy.geometry import PointCloud
            import plotly.graph_objects as go
            points = PointCloud.randn(10)
            go.Figure(points.scatter(name="scatter example"))
        """
        mode = kwargs.pop("mode", "markers")
        if self.tags is not None and any(self.tags):
            if "text" not in mode:
                mode += "+text"
            kwargs["text"] = kwargs.get("text", self.tags)
            kwargs["textposition"] = kwargs.get("textposition", "top center")
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode=mode,
            **kwargs,
        )

    def scatterpolar(self, **kwargs):
        """
        Produce a :obj:`go.Scatter3d` trace of the points with some sensible defaults for polar coordinates.
        Any keyword arguments will be passed through to plotly.

        .. plotly::
            :include-source:

            from spatpy.geometry import PointCloud
            import plotly.graph_objects as go
            points = PointCloud.randn(15)
            fig = go.Figure(points.scatterpolar(name="scatterpolar example"))
        """

        return self.scatter(
            mode=kwargs.get("mode", "markers"),
            marker=kwargs.get(
                "marker",
                dict(
                    color=self.az,
                    colorscale="phase",
                    cmin=-np.pi,
                    cmax=np.pi,
                ),
            ),
            customdata=kwargs.get(
                "customdata", self.mesh3d_customdata(self.intensity)
            ),
            hovertemplate=kwargs.get(
                "hovertemplate",
                "φ: %{customdata[0]:.1f}°<br>θ: %{customdata[1]:.1f}°<br>ρ:"
                " %{customdata[2]:.1f}",
            ),
            **kwargs,
        )

    def plot_radial_delaunay(
        self,
        polar=False,
        transpose=False,
        wrap=True,
        show_bounds=True,
        **kwargs,
    ):
        """
        Plot the mesh formed by performing delaunay triangulation on the set of points.
        This is the mesh used for :obj:`PointCloud.plot_mesh3d` and :obj:`PointCloud.plot_outline`.

        .. plotly::
            :include-source:
            :context:

            from spatpy.geometry import PointCloud
            from spatpy import placement
            from plotly import graph_objects as go

            dist = placement.uniform_spherical_distribution(25)
            el = dist.el
            az = dist.az
            r = np.abs(np.cos(el) * np.sin(az))
            points = PointCloud.from_spherical(az=az, el=el, r=r)
            go.Figure(points.scatterpolar())

        .. plotly::
            :include-source:
            :context:

            points.plot_mesh3d(flatshading=True)

        .. plotly::
            :include-source:
            :context:

            points.plot_radial_delaunay(wrap=True, name="wrapped mesh")

        .. plotly::
            :include-source:
            :context:

            points.plot_radial_delaunay(wrap=False, name="unwrapped mesh")

        .. plotly::
            :include-source:
            :context:

            points.plot_radial_delaunay(polar=True, name="polar mesh")
        """
        points, simplices = self.delaunay if wrap else self.delaunay_unwrapped
        az = points[:, 0]
        el = points[:, 1]
        if polar:
            transpose = not transpose

        if polar and transpose:
            el = (el + 2 * np.pi) % (2 * np.pi)
            az = (az + 2 * np.pi) % (2 * np.pi)

        if transpose:
            points = np.vstack((el, az)).T
        traces = [triplot(points, simplices, polar=polar, **kwargs)]
        if not polar and (not wrap and show_bounds):
            x = [-np.pi, -np.pi, np.pi, np.pi, -np.pi]
            y = [-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2]
            if transpose:
                (y, x) = (x, y)
            traces.append(go.Scatter(x=x, y=y, name="unit sphere"))
        fig = go.Figure(traces)

        xaxis = dict(title="φ", **polarticks(-3, 3, 2))
        yaxis = dict(title="θ", **polarticks(-1, 1, 4, offset=-np.pi / 2))
        if transpose:
            (yaxis, xaxis) = (xaxis, yaxis)
        if polar:
            fig.update_layout(
                polar=dict(
                    angularaxis=dict(thetaunit="radians"),
                    radialaxis=yaxis,
                )
            )
        else:
            fig.update_xaxes(**xaxis)
            fig.update_yaxes(**yaxis)
        return fig

    def compute_radial_delaunay(self, wrap=True):
        az = self.az
        el = self.el

        # stack up 3 versions of the points
        el_block = np.concatenate((el, el, el))
        az_block = np.concatenate((az - 2 * np.pi, az, az + 2 * np.pi))

        # do delaunay triangulation
        tri_block = Delaunay(np.vstack((az_block, el_block)).T)

        # only keep simplices inside the [-pi, pi] range, see plot_radial_delaunay
        if wrap:
            tri_indices = np.any(
                (np.abs(az_block[tri_block.simplices]) <= np.pi),
                axis=1,
            )
            filtered = (tri_block.simplices[tri_indices]) % len(az)
            unique = np.logical_not(
                np.any(np.diff(np.sort(filtered, axis=1), axis=1) == 0, axis=1)
            )
            simplices = filtered[unique, :]
            points = np.vstack((az, el)).T
        else:
            simplices = tri_block.simplices
            points = tri_block.points

        # wrap simplices back around into normal indexing
        return (points, simplices)

    @property
    def delaunay(self):
        return self.compute_radial_delaunay(wrap=True)

    @property
    def delaunay_unwrapped(self):
        return self.compute_radial_delaunay(wrap=False)

    def projection_outline(self, angularaxis=None, min_points=12, eps=1e-3):
        """ "
        Perform delaunay triangulation and trace the outline of the figure in the given axis.
        See the documentation for :obj:`PointCloud.plot_outline` for a visualisation.

        Returns:
            An ordered list of indices of the points comprising the outline.
        """
        points = self
        if angularaxis is None:
            angularaxis = "az"

        in_plane = np.isclose(points.z if angularaxis == "az" else points.y, 0)

        # if there are enough points in the plane (e.g. uniform sampling), just use those
        if np.count_nonzero(in_plane) >= min_points:
            # sort by angle along axis we are interested in
            perm = np.argsort(points[in_plane][angularaxis].values)
            return (np.arange(len(points))[in_plane])[perm]

        try:
            _, simplices = self.delaunay
        except:
            return np.array([], dtype=np.int64)

        start = np.argmax(points.x)
        paths = []
        r, az, el = cart2sph(
            x=points.x, y=points.y if angularaxis == "az" else points.z
        )

        for direction in [1, -1]:
            current = start
            path = []
            done = False
            while not done:
                path.append(current)
                idx = (
                    (az[simplices] * direction) < (az[current] * direction)
                ) & np.tile(
                    np.expand_dims(np.any(simplices == current, axis=1), 1),
                    (1, 3),
                )
                if not np.any(idx):
                    break
                best_r = np.argmax(r[simplices[idx]])
                current = simplices[idx][best_r]
                if current != start and current in path:
                    done = True
            paths.append(path)

        combined_path = paths[0][1:] + paths[1][::-1]
        # combine points up until a certain precision
        angles = self.az if angularaxis == "az" else self.el
        i = 0
        final_path = [combined_path[i]]
        i = 1
        while i < len(combined_path):
            prev = final_path[-1]
            cur = combined_path[i]
            if (
                np.abs(
                    (angles[cur] - angles[prev])
                    % (2 * np.pi if angularaxis == "az" else np.pi)
                )
                > eps
            ):
                final_path.append(cur)
            else:
                final_path[-1] = cur if self.r[cur] > self.r[prev] else prev
            i += 1
        return np.array(final_path, dtype=np.int64)

    def plot_outline(self, outline=None, angularaxis=None, **kwargs):
        """
        Return a :obj:`go.Scatterpolar` trace of the projection outline after performing delaunay triangulation
        as in :obj:`PointCloud.projection_outline`. Keyword arguments passed through to plotly.

        .. plotly::
            :include-source:
            :context: reset

            from spatpy.geometry import PointCloud
            from spatpy import placement
            import plotly.graph_objects as go

            dist = placement.uniform_spherical_distribution(25)
            el = dist.el
            az = dist.az
            r = np.abs(np.cos(el) * np.sin(az))
            points = PointCloud.from_spherical(az=az, el=el, r=r)
            go.Figure(points.plot_outline(angularaxis="az"))
        """

        if outline is None:
            outline = self.projection_outline(angularaxis=angularaxis)
        if angularaxis is None or angularaxis == "az":
            theta = self.az[outline]
        else:
            theta = self.el[outline]
        r = self.points_xr.r[outline].values
        return go.Scatterpolar(
            thetaunit="radians",
            theta=theta.tolist() + [theta[0]],
            r=r.tolist() + [r[0]],
            **kwargs,
        )

    def plot_mesh3d(self, **kwargs):
        """Return a :obj:`go.Figure` containing the trace from :obj:`PointCloud.mesh3d`."""
        return go.Figure(self.mesh3d(**kwargs))

    def mesh3d_customdata(self, intensity):
        return np.vstack((self.az_deg, self.el_deg, self.r, intensity)).T

    @staticmethod
    def interpolate_phase(intensity, simplices):
        ncell = simplices.shape[0]
        interpolated = np.zeros(ncell)
        for i in range(ncell):
            p = intensity[simplices[i, :]]
            mp = p[0] + np.pi
            mp = np.mod(
                mp + (np.mod(p[1] - mp + 2 * np.pi, 2 * np.pi) - np.pi) / 2,
                2 * np.pi,
            )
            mp = np.mod(
                mp + (np.mod(p[2] - mp + 2 * np.pi, 2 * np.pi) - np.pi) / 3,
                2 * np.pi,
            )
            interpolated[i] = mp - np.pi
        return interpolated

    def get_mesh3d_data(
        self, intensity=None, colorscale=None, cmin=-np.pi, cmax=np.pi
    ):
        points, simplices = self.delaunay

        vertexcolor = None
        if intensity is None:
            intensity = self.intensity
        if colorscale is not None:
            vertexcolor = px.colors.sample_colorscale(
                colorscale,
                np.clip((intensity - cmin) / (cmax - cmin), 0.0, 1.0),
            )

        return dict(
            x=self.x,
            y=self.y,
            z=self.z,
            i=simplices[:, 0],
            j=simplices[:, 1],
            k=simplices[:, 2],
            customdata=self.mesh3d_customdata(intensity),
            vertexcolor=vertexcolor,
        )

    def mesh3d(self, polar=True, intensity_fmt=None, **kwargs):
        """
        Produce a :obj:`go.Mesh3d` trace of the points with some sensible defaults.
        Any keyword arguments will be passed through to plotly.
        See also the :obj:`PointCloud.plot_mesh3d` convenience method.

        .. plotly::
            :include-source:
            :context: reset

            from spatpy.geometry import PointCloud
            from spatpy import placement
            import plotly.graph_objects as go
            points = placement.uniform_spherical_distribution(20)
            go.Figure(points.mesh3d(polar=True, name="polar mesh example"))

        .. plotly::
            :include-source:
            :context:

            points.plot_mesh3d(polar=False, name="cartesian mesh example")
        """
        if polar:
            if intensity_fmt is None:
                intensity_fmt = "v: %{customdata[3]:.2f}"
            else:
                intensity_fmt = intensity_fmt.replace(
                    "intensity", "customdata[3]"
                )
            if kwargs.get("hovertemplate") is None:
                kwargs["hovertemplate"] = (
                    "φ: %{customdata[0]:.1f}°<br>θ: %{customdata[1]:.1f}°<br>ρ:"
                    " %{customdata[2]:.1f}<br>"
                    + intensity_fmt
                    + "<extra></extra>"
                )

        if kwargs.get("lighting") is None:
            kwargs["lighting"] = dict(roughness=0.4, specular=0.1, ambient=0.95)
        # if kwargs.get("flatshading") is None:
        #     kwargs["flatshading"] = True

        triangles = go.Mesh3d(
            **self.get_mesh3d_data(
                intensity=kwargs.pop("intensity", self.intensity),
                colorscale=kwargs.pop("colorscale", None),
                cmin=kwargs.pop("cmin", -np.pi),
                cmax=kwargs.pop("cmax", np.pi),
            ),
            **kwargs,
        )
        return triangles

    def sample_approx(self, locs):
        """
        Return a boolean mask of the closest points in this array to the
        given points.
        """
        subset = np.zeros(len(self), dtype=np.bool)
        for l in locs.points:
            closest = np.argmin([l.r2_distance(s) for s in self.points])
            subset[closest] = True
        return subset
