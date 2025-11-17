import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple, Any, List, Dict
from dataclasses import dataclass, field
from functools import lru_cache

from spatpy.geometry import Point, PointCloud
from spatpy import uniform_sampling
import xarray as xr
import math
from pathlib import Path


def _sph_distribution(
    n_points: Optional[int] = None,
    radius: float = 1.0,
    p0: Optional[PointCloud] = None,
    max_iters: Optional[int] = None,
    max_az_err_deg: float = 0.1,
    max_el_err_deg: float = 0.1,
) -> Tuple[PointCloud, Optional[go.Figure]]:
    if n_points is None:
        assert p0 is not None
        n_points = len(p0)

    if max_iters is None:
        max_iters = 2 * n_points

    if p0 is None:
        # approximate raised cosine distribution
        elevations = np.random.normal(
            np.pi / 2, np.pi / 2 * np.sqrt(1 / 3 - 2 / (np.pi**2)), n_points
        )
        # uniform distribution
        azimuths = np.random.rand(n_points) * 2 * np.pi
        p0 = PointCloud.from_spherical(az=azimuths, el=elevations - np.pi / 2)

    az = p0.az
    el = p0.el

    # az_opt = AdamOptimiser(alpha=0.1, beta1=0.9, beta2=0.99)
    # el_opt = AdamOptimiser(alpha=0.1, beta1=0.9, beta2=0.99)

    i = 0
    done = False
    while i < max_iters and not done:
        d_az, d_el = uniform_sampling.compute_gradient(az, el)
        az -= d_az
        el -= d_el
        if (
            np.rad2deg(np.max(np.abs(d_az))) < max_az_err_deg
            and np.rad2deg(np.max(np.abs(d_el))) < max_el_err_deg
        ):
            done = True
        i += 1

    p = PointCloud.from_spherical(r=radius, az=az, el=el - np.pi / 2)

    # Rotate to place point 1 on the north pole
    p = p.rotate_axis("z", -az[0])
    p = p.rotate_axis("y", -el[0])

    # Rotate to place point closest to equator at 0 azimuth
    closest_to_equator = np.argmax(np.abs(p.el - np.pi / 2))
    p = p.rotate_axis("z", -az[closest_to_equator])

    # Reorder in a descending spiral
    idx = np.argsort(p.az, kind="stable")
    p = PointCloud([p.points[i] for i in idx])
    idx = np.argsort(p.el, kind="stable")
    p = PointCloud([p.points[i] for i in idx])

    return p


def circular_array(
    n,
    radius_m=1,
    elevation=0.0,
    offset=0,
    tag_prefix=None,
    elevation_jitter_deg=0.0,
    endpoint=False,
):
    """
    Evenly spaced circular microphone array.

    Args:
        n (int): number of azimuth angles
        radius_m (float, optional): radius of array in metres
        elevation (float, optional): elevation of array (radians)
        offset (float, optional): starting azimuth offset (radians)
        tag_prefix (str, optional): prefix applied to each tag
        elevation_jitter_deg (float, optional): jitter elevations randomly by up to this amount


    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.circular_array(5, tag_prefix="mic")
        go.Figure(points.scatter(name="circular mic array"))
    """
    az = np.linspace(-np.pi + offset, np.pi + offset, n, endpoint=endpoint)
    el = elevation + np.random.randn(n) * np.deg2rad(elevation_jitter_deg)
    tags = None
    if tag_prefix is not None:
        tags = [f"{tag_prefix}{i}" for i in range(n)]
    return PointCloud.from_spherical(radius_m, az, el, tags=tags)


def equiangular_half_spherical_array(
    n, m, radius_m=1, tag_prefix=None, az_offset=0
):
    """
    Half-spherical array with equiangular sampling.

    Args:
        n (int): number of azimuth angles
        m (int): number of elevation angles
        radius_m (float, optional): radius of array in metres
        tag_prefix (str, optional): prefix applied to each tag
        az_offset (float, optional): starting azimuth offset (radians)

    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.equiangular_half_spherical_array(9, 5)
        go.Figure(points.scatterpolar())
    """
    az = np.linspace(-np.pi + az_offset, np.pi + az_offset, n, endpoint=False)
    els = np.linspace(0, np.pi / 2, m, endpoint=True)

    tags = None
    points = []
    for el in els:
        n = len(points)
        if el == -np.pi / 2 or el == np.pi / 2:
            x = -np.pi if el == -np.pi / 2 else np.pi
            if tag_prefix is not None:
                tags = f"{tag_prefix}{n}"
            points.extend(
                PointCloud.from_spherical(radius_m, x, el, tags=tags).points
            )
        else:
            if tag_prefix is not None:
                tags = [f"{tag_prefix}{n + i}" for i in range(len(az))]
            points.extend(
                PointCloud.from_spherical(radius_m, az, el, tags=tags).points
            )
    return PointCloud(points)


def beehive_array(n_middle, n_upper=0, n_lower=0, n_zenith=1, radius=1.0):
    """
    Spherical array with beehive sampling.

    Args:
        n_middle (int): middle ring channel count
        n_upper (int, optional): upper ring channel count
        n_lower (int, optional): lower ring channel count
        n_zenith (int, optional): zenith channel count (0 or 1)
        radius (float, optional): radius of array in units

    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.beehive_array(15, 9, 5, 1)
        go.Figure(points.scatterpolar())
    """

    points = []
    for (ring_name, n_ring, ring_el) in [
        ("L", n_lower, -45),
        ("M", n_middle, 0),
        ("U", n_upper, 45),
    ]:
        az = np.linspace(-np.pi, np.pi, n_ring, endpoint=False)
        el = np.zeros_like(az) + np.deg2rad(ring_el)
        tags = [f"{ring_name}{i + 1}" for i in range(n_ring)]
        points.extend(
            PointCloud.from_spherical(radius, az, el, tags=tags).points
        )
    if n_zenith == 1:
        points.append(Point(x=0, y=0, z=1.0, tag="Z1"))
    return PointCloud(points)


def equiangular_spherical_array(n, m, radius_m=1.0, az_offset=0):
    """
    Spherical array with equiangular sampling.

    Args:
        n (int): number of azimuth angles
        m (int): number of elevation angles
        radius_m (float, optional): radius of array in metres
        az_offset (float, optional): starting azimuth offset (radians)

    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.equiangular_spherical_array(9, 7)
        go.Figure(points.scatterpolar())
    """
    az = np.linspace(-np.pi + az_offset, np.pi + az_offset, n, endpoint=False)
    els = np.linspace(-np.pi / 2, np.pi / 2, m, endpoint=True)
    points = []
    for el in els:
        if el == -np.pi / 2 or el == np.pi / 2:
            x = -np.pi if el == 0 else np.pi
            points.extend(PointCloud.from_spherical(radius_m, x, el).points)
        else:
            points.extend(PointCloud.from_spherical(radius_m, az, el).points)
    return PointCloud(points)


def uniform_spherical_distribution(
    n_points: int, radius=1.0, max_az_err_deg=0.1, max_el_err_deg=0.1
) -> PointCloud:
    """
    Spherical array with uniform sampling.

    Args:
        n_points (int): number of points
        radius (float, optional): radius of array

    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.uniform_spherical_distribution(30)
        go.Figure(points.scatterpolar())
    """
    return _sph_distribution(
        n_points=n_points,
        radius=radius,
        max_az_err_deg=max_az_err_deg,
        max_el_err_deg=max_el_err_deg,
    )


def goldberg_distribution(n: int, radius=1.0) -> PointCloud:
    """
    Spherical array with Icosahedral Goldberg sampling. See :obj:`SphereMesh` class
    for more details.

    ===  =====================================================
    N    No of Vertices
    ===  =====================================================
    0      12 (the 12 vertices of an icosohedron)
    1      42
    2     162
    3     642
    4    2562
    5    10242 (very big - not recommended for general use)
    ===  =====================================================

    Args:
        n (int): order of sampling ``N``
        radius (float, optional): radius of array

    Returns:
        A :obj:`PointCloud` representing the array

    .. plotly::
        :include-source:

        from spatpy import placement
        import plotly.graph_objects as go

        points = placement.goldberg_distribution(2)
        go.Figure(points.scatterpolar())
    """
    sm = SphereMesh(n)
    points = PointCloud.from_cartesian(
        x=sm.vertices[0, :], y=sm.vertices[1, :], z=sm.vertices[2, :]
    )
    if radius != 1.0:
        points = points.scale_radius(radius)
    return points


@lru_cache
@dataclass
class SphereMesh:
    """
    The vertices of ``SphereMesh(N)`` correpond to the centres of the
    faces of a the Icosahedral Goldberg ``G(0,2^N)`` polyhedron, which
    has 12 pentagons and ``10*(4^N)-10`` hexagons. The number of vertices
    is equal to ``10*(4^N)+2``.

    ===  =====================================================
    N    No of Vertices
    ===  =====================================================
    0      12 (the 12 vertices of an icosohedron)
    1      42
    2     162
    3     642
    4    2562
    5    10242 (very big - not recommended for general use)
    ===  ====================================================="""

    order: int = 1
    basepoly: int = 3  # Use triangles by default
    fudge: Optional[float] = None

    def __post_init__(self):
        assert self.basepoly in [3, 4]
        if self.basepoly == 3:
            # x = SphereMesh(3); csvwrite('spheremesh_matlab.txt', x.Vertices);
            # self.matlab_vertices = np.loadtxt(
            #     "spheremesh_matlab.txt", delimiter=","
            # )
            self.fudges = np.hstack(
                [
                    [
                        0,
                        0,
                        0.05235825199633838800,
                        0.00387034958228468780,
                        0.00077944508381187934,
                        0.00018317218311130997,
                        0.00022519531250000003,
                    ],
                    np.zeros(10),
                ]
            )
            if self.fudge is not None:
                self.fudges[self.order] = self.fudge
            self.nface = 0
            self.faces = np.zeros((3, 20 * (4**self.order)))
            self.nvertex = 0
            self.vertices = np.zeros((3, 10 * (4**self.order) + 2))
            T = (1 + np.sqrt(5)) / 2
            for f1 in [-1, 1]:
                for f2 in [-1, 1]:
                    for f3 in [-1, 1]:
                        v1 = np.array([0, T * f2, f1]).T
                        v2 = np.array([T * f3, f2, 0])
                        v3 = np.array([f3, 0, T * f1])
                        if (
                            np.dot(
                                (v1 + v2 + v3), np.cross(v1, np.cross(v2, v3))
                            )
                            > 0
                        ):
                            self.add_faces(v1, v2, v3, 0)
                        else:
                            self.add_faces(v1, v3, v2, 0)
            for flip1 in [-1, 1]:
                for flip2 in [-1, 1]:
                    for orient in range(3):
                        tmp = np.array([[0, T, 1], [0, T, -1], [T, 1, 0]]).T
                        tmp = np.diag([flip2, flip1, 1]) @ tmp
                        tmp = np.roll(tmp, orient, 0)
                        v1 = tmp[:, 0]
                        v2 = tmp[:, 1]
                        v3 = tmp[:, 2]
                        if (
                            np.dot(
                                (v1 + v2 + v3), np.cross(v1, np.cross(v2, v3))
                            )
                            > 0
                        ):
                            self.add_faces(v1, v2, v3, 0)
                        else:
                            self.add_faces(v1, v3, v2, 0)
        else:
            self.faces = np.zeros((4, 6 * (4**self.order)))
            self.vertices = np.zeros((3, 6 * (4**self.order) + 2))

        # self.make_edges()

    def get_equator_weights(self):
        """get a vector of weights for processing equator"""
        OffEquatorW = 0.5
        OffEquatorHeight = 0.8 * 2 ^ (-self.order)
        W = (np.abs(self.vertices[1, :]) < 1e-5) * (1 - OffEquatorW) + (
            np.abs(self.vertices[1, :]) < OffEquatorHeight
        ) * OffEquatorW
        W_upper = 1 - W
        W_upper[self.vertices[1, :] < 0] = 0
        W_lower = 1 - W
        W_lower[self.vertices[1, :] > 0] = 0

    def __str__(self):
        return (
            f"SphereMesh({self.nvertex}-vertices) - mean/max spacing"
            f" {np.sqrt(4*np.pi/self.nvertex)*180/np.pi:.1f}/{75*2^(-self.order):.1f} degrees"
        )

    def get_vertex_index(self, xyz):
        if np.isscalar(xyz):
            v = xyz
        else:
            # assert not np.isclose(np.sum(xyz**2), 0)
            xyz = xyz / np.sqrt(np.sum(xyz**2))
            v = None
            if self.nvertex:
                v = np.argwhere(
                    np.sum(
                        np.abs(
                            self.vertices[:, : self.nvertex]
                            - np.tile(np.expand_dims(xyz, 1), (1, self.nvertex))
                        ),
                        0,
                    )
                    < 1e-8
                )
                if len(v) == 0:
                    v = None
                else:
                    v = v[0]

            if v is None:
                v = self.nvertex
                self.vertices[:, self.nvertex] = xyz
                # if (
                #     np.max(np.abs(self.matlab_vertices[:, self.nvertex] - xyz))
                #     > 1e-4
                # ):
                #     print(f"difference at vertex {self.nvertex}")
                self.nvertex += 1

            return v

    def add_faces(self, vo1, vo2, vo3, order, v1=None, v2=None, v3=None):
        v1 = np.array([1, 0, 0]) if v1 is None else v1
        v2 = np.array([0, 1, 0]) if v2 is None else v2
        v3 = np.array([0, 0, 1]) if v3 is None else v3
        vo1 = vo1 / np.sqrt(np.dot(vo1, vo1))
        vo2 = vo2 / np.sqrt(np.dot(vo2, vo2))
        vo3 = vo3 / np.sqrt(np.dot(vo3, vo3))
        v1 = v1 + np.sign(v1) * self.fudges[order]
        v1 /= np.sum(v1)
        v2 = v2 + np.sign(v2) * self.fudges[order]
        v2 /= np.sum(v2)
        v3 = v3 + np.sign(v3) * self.fudges[order]
        v3 /= np.sum(v3)
        if order == self.order:
            v1 = self.get_vertex_index(np.array([vo1, vo2, vo3]).T @ v1)
            v2 = self.get_vertex_index(np.array([vo1, vo2, vo3]).T @ v2)
            v3 = self.get_vertex_index(np.array([vo1, vo2, vo3]).T @ v3)
            self.faces[0, self.nface] = v1
            self.faces[1, self.nface] = v2
            self.faces[2, self.nface] = v3
            self.nface += 1
        elif order < self.order:
            v12 = (v1 + v2) / 2
            v13 = (v1 + v3) / 2
            v23 = (v2 + v3) / 2
            self.add_faces(vo1, vo2, vo3, order + 1, v1, v12, v13)
            self.add_faces(vo1, vo2, vo3, order + 1, v2, v23, v12)
            self.add_faces(vo1, vo2, vo3, order + 1, v3, v13, v23)
            self.add_faces(vo1, vo2, vo3, order + 1, v23, v13, v12)

    def make_edges(self):
        raise NotImplementedError()


def spherical_distribution_accuracy(
    distributions: List[PointCloud], max_poly_order=12, verbose=True
):
    worst_snr_db = np.zeros((len(distributions), max_poly_order))
    if verbose:
        print(
            """\
SphereMesh is intended to generate points on the surface of the unit
sphere so that polynomial functions (in x,y,z) can be integrated
numerically, by simply evaluating the polynomial function at the
sample-points and taking the mean value of the results. The following
table shows the error, relative to the RMS value of the true function, in dB
"""
        )
        print(
            "NumPts "
            + " ".join(
                [f"Ord {str(i)}".rjust(6) for i in range(1, max_poly_order + 1)]
            )
        )
        print("------ " + " ".join(["-" * 6 for _ in range(max_poly_order)]))
    # Lets iterate over different size SphereMesh sets:
    for (i, distribution) in enumerate(distributions):
        if verbose:
            print(str(len(distribution)).rjust(6), end="")

        points = distribution.points_xr
        # let's build every possible polynomial term for 1st-order to max-order
        for poly_order in range(1, max_poly_order + 1):
            worstSNR = 0
            # build every type of polynomial we can make, of this order = <ord>
            for xo in range(poly_order + 1):
                for yo in range(poly_order - xo + 1):
                    zo = poly_order - xo - yo

                    # OK, we are trying to integate (over the surface of the
                    # unit-sphere), the function:  x^xo * y^yo * z^zo
                    # (where xo+yo+zo=ord)

                    # Figure out the expected result of integration:
                    if all(np.mod([xo, yo, zo], 2) == 0):
                        # Here's a closed-form solution to the integral of a polynomial
                        # term in x,y,z for the special case where all of the powers are
                        # even
                        expectedMean = (
                            math.gamma((xo + 1) / 2)
                            * math.gamma((yo + 1) / 2)
                            * math.gamma((zo + 1) / 2)
                            / math.gamma((xo + yo + zo + 3) / 2)
                            / 2
                            / np.pi
                        )
                    else:
                        # If any of the variables (x, y or z) have an odd power, then
                        # the integral must be zero:
                        expectedMean = 0

                    # Let's evaluate the polynomial at all the points, and subtract
                    # off the expected mean value (so <polyvals> should be all zeros
                    # (or close to zeros))
                    polyvalues = (
                        -expectedMean
                        + points.x.values**xo
                        * points.y.values**yo
                        * points.z.values**zo
                    )
                    # compute the SNR, by comparing the integral (the mean value) to
                    # the RMS value of the polynomial
                    polyRMS = np.sqrt(np.mean(polyvalues**2))
                    polyIntegration = np.mean(polyvalues)
                    if polyRMS > 0:
                        SNR = abs(polyIntegration) / polyRMS
                        # Let's track the largest error
                        worstSNR = max(worstSNR, SNR)
            worst_snr_db[i, poly_order - 1] = (
                0
                if worstSNR == 0
                else max(-999, round(20 * np.log10(worstSNR)))
            )
            # Print one entry in the table, reporting the SNR in dB
            if verbose:
                print(
                    str(worst_snr_db[i, poly_order - 1]).rjust(7),
                    end="",
                )
        if verbose:
            print()

    if verbose:
        print()
    return worst_snr_db


@dataclass
class DeviceGeometry:
    locs: List[Point] = field(default_factory=list)
    centroid: Any = None
    c_m_per_s: float = 343.3

    def __post_init__(self):
        if self.centroid is None:
            self.centroid = sum(self.locs) / len(self.locs)

    def sample_approx(self, dist):
        subset = self.locs.sample_approx(dist)
        new_geom = DeviceGeometry(
            locs=PointCloud(points_xr=self.locs[subset]),
            c_m_per_s=self.c_m_per_s,
        )
        return new_geom, subset

    @property
    def count(self):
        return len(self.locs)

    @property
    def largest_separation(self):
        max_sep = 0
        for i in range(self.count):
            for j in range(i):
                sep_ij = self.locs[i].r2_distance(self.locs[j])
                if sep_ij > max_sep:
                    max_sep = sep_ij
        return max_sep

    @classmethod
    def circular_mics(
        cls,
        nmic: int = 4,
        size_mm: float = 30,
        centre_offset_mm: float = 0,
        centre_offset_deg: float = 0,
        rotation_deg: float = 0,
        separation_scale: List[Tuple[int, float]] = None,
    ):
        size_m = size_mm * 1e-3
        locs = circular_array(nmic, size_m).rotate(az=rotation_deg, deg=True)
        if separation_scale is not None:
            for (mic, scale) in separation_scale:
                locs[mic] *= scale
        centre_loc = Point.from_spherical(
            r=centre_offset_mm / 1e3,
            az=centre_offset_deg,
            deg=True,
            tag="centre",
        )
        locs += centre_loc
        return cls(locs=locs)

    @classmethod
    def centred_linear_mics_mm(cls, spacing_mm):
        centre = sum(spacing_mm) / 2
        locs = (
            np.concatenate(([0.0], np.cumsum(np.array(spacing_mm)))) - centre
        ) * 1e-3
        return cls(locs=PointCloud.from_cartesian(y=[loc for loc in locs]))

    @classmethod
    def turntable_sources(
        cls,
        az_step_deg: float = 10.0,
        elevation_jitter_deg: float = 0.0,
        source_dist_m: float = 1,
    ):
        count = int(360 / az_step_deg)
        locs = circular_array(
            count, source_dist_m, elevation_jitter_deg=elevation_jitter_deg
        )
        return cls(locs)

    @classmethod
    def equiangular_sphere_sources(
        cls,
        az_step_deg: float = 30.0,
        el_step_deg: float = 30.0,
        source_dist_m: float = 1,
    ):
        az_count = int(360 / az_step_deg)
        el_count = int(180 / el_step_deg) + 1
        locs = equiangular_spherical_array(az_count, el_count, source_dist_m)
        return cls(locs)

    @classmethod
    def uniform_sphere_sources(cls, n=84, source_dist_m: float = 1):
        locs = uniform_spherical_distribution(n, source_dist_m)
        return cls(locs)

    @classmethod
    def goldberg_sources(cls, n=3, source_dist_m: float = 1):
        locs = goldberg_distribution(n, source_dist_m)
        return cls(locs)

    @property
    def el(self):
        return self.locs.el

    @property
    def el_deg(self):
        return self.locs.el_deg

    @property
    def az(self):
        return self.locs.az

    @property
    def az_deg(self):
        return self.locs.az_deg

    @property
    def f_nyq_hz(self):
        return (self.c_m_per_s / self.largest_separation) / 2

    def relative_delays(self, other):
        delays = np.zeros((len(other.locs), len(self.locs)))
        for i in range(len(self.locs)):
            for j in range(len(other.locs)):
                delays[j, i] = (
                    self.locs.points[i].r2_distance(other.locs.points[j])
                    / self.c_m_per_s
                )

        return delays

    def plot_locations(self, body_trace=None) -> go.Figure:
        """
        Produce a figure showing the microphone locations, optionally with a body mesh
        """
        fig = go.Figure()
        fig.add_trace(
            self.locs.scatter(name="mics", marker=dict(color="black")),
        )

        if body_trace is not None:

            fig.add_trace(body_trace)

        fig.update_scenes(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"
        )
        if np.max(self.locs.z) > np.min(self.locs.z):
            fig.update_scenes(aspectmode="data")
        fig.update_scenes(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-3, y=0.5, z=0.5),
            ),
        )
        return fig


if __name__ == "__main__":
    sphere_meshes = [goldberg_distribution(i) for i in range(5)]
    spherical_distribution_accuracy(sphere_meshes)
    uniform_dists = [
        uniform_spherical_distribution(len(mesh)) for mesh in sphere_meshes
    ]
    spherical_distribution_accuracy(uniform_dists)
