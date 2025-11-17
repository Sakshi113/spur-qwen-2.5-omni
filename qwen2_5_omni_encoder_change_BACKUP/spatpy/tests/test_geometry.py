import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from spatpy.geometry import Point
from spatpy.placement import (
    DeviceGeometry,
    goldberg_distribution,
    spherical_distribution_accuracy,
)
import numpy as np
import math


def test_sph2cart():
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                p = Point(x=x, y=y, z=z)
                q = Point.from_spherical(r=p.r, az=p.az, el=p.el)
                assert (
                    np.isclose(p.x, q.x)
                    and np.isclose(p.y, q.y)
                    and np.isclose(p.z, q.z)
                )


def test_conventions():
    p = Point(x=1, y=0, z=0)
    assert np.isclose(p.az, 0.0)
    assert np.isclose(p.el, 0.0)

    p = Point(x=0, y=1, z=0)
    assert np.isclose(p.az, np.pi / 2)
    assert np.isclose(p.el, 0.0)

    p = Point(x=0, y=-1, z=0)
    assert np.isclose(p.az, -np.pi / 2)
    assert np.isclose(p.el, 0.0)

    p = Point(x=-1, y=0, z=0)
    assert np.isclose((np.pi + p.az) % (2 * np.pi), 0.0)
    assert np.isclose(p.el, 0.0)

    p = Point(x=0, y=0, z=1)
    assert np.isclose(p.el, np.pi / 2)

    p = Point(x=0, y=0, z=-1)
    assert np.isclose(p.el, -np.pi / 2)


def test_placement():
    sources = DeviceGeometry.turntable_sources(az_step_deg=5)
    assert np.isclose(
        np.abs(sources.locs[0].az_deg - sources.locs[1].az_deg), 5
    )


def test_sphere_mesh():
    # better than -300dB up to fifth order
    sphere_meshes = [goldberg_distribution(i) for i in range(5)]
    worst_snr_db = spherical_distribution_accuracy(
        sphere_meshes, max_poly_order=5, verbose=False
    )
    assert np.all(worst_snr_db < -300)


if __name__ == "__main__":
    test_sphere_mesh()
    test_conventions()
    test_sph2cart()
    test_placement()
