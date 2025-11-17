from stl import mesh
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional
from plotly import graph_objects as go
from spatpy.geometry import Point, PointCloud
from spatpy.placement import DeviceGeometry
from pathlib import Path
from stltovoxel import convert_meshes
import json

# https://chart-studio.plotly.com/~empet/15276/converting-a-stl-mesh-to-plotly-gomes/#/
def stl2mesh3d(stl_file):
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(
        stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
    )
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    x, y, z = vertices.T
    return x, y, z, i, j, k


def plot_stl_file(stl_file, **kwargs):
    x, y, z, i, j, k = stl2mesh3d(stl_file)
    z -= np.min(z)
    x /= np.max(x)
    y /= np.max(y)
    z /= np.max(z)
    z -= 0.5
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


@dataclass
class DeviceModel:
    """
    A 3D solid with some microphones attached. Example shown below.

    .. plotly::
        :include-source:

        from spatpy.device_model import DeviceModel

        phone = DeviceModel(
            name="find_x3_pro",
            height=163.6,
            width=74.0,
            depth=8.26,
            mic_order=["back", "top", "bottom"],
            mic_position=dict(
                back=(47.5, 23.7, 9.7),
                top=(62.2, 0.0, 3.6),
                bottom=(25.0, 163.6, 4.1),
            ),
        )
        phone.plot_locations()
    """

    name: str
    mic_order: List[str]
    mic_position: Dict
    mic_position_anchor: Optional[str] = None

    height: Optional[float] = None
    width: Optional[float] = None
    depth: Optional[float] = None
    scale: float = 1e-3
    stl_file: Optional[str] = None

    def __post_init__(self):
        if self.stl_file is not None:
            x, y, z, i, j, k = stl2mesh3d(self.stl_file)
            self.depth = np.max(x) - np.min(x)
            self.height = np.max(y) - np.min(y)
            self.width = np.max(z) - np.min(z)
            x *= self.scale
            y *= self.scale
            z *= self.scale
            self.body = x, y, z, i, j, k
            points = []
            if self.mic_position_anchor in [None, 'top_left']:
                # convert from phone stl file which is oriented
                # with top back right corner on origin
                y += self.scale * self.height / 2
                z -= self.scale * self.width / 2
                device_top_left_corner = Point(
                    x=self.depth * self.scale / 2,
                    y=self.height * self.scale / 2,
                    z=self.width * self.scale / 2,
                )
                for (tag, (w, h, d)) in self.mic_position.items():
                    # this puts the origin in the middle of the body
                    # with the y axis along the longest dimension of the phone
                    # and the x axis in the look direction (as per our convention)
                    points.append(
                        (
                            Point(x=d - self.depth, y=-h, z=w - self.width, tag=tag)
                            * self.scale
                            + device_top_left_corner
                        )
                    )
            else:
                assert self.mic_position_anchor == 'absolute'
                for (tag, (x, y, z)) in self.mic_position.items():
                    points.append(
                        (
                            Point(x=x, y=y, z=z, tag=tag) * self.scale
                        )
                    )
            self.mics = DeviceGeometry(PointCloud(points))
        else:
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        p = Point(
                            x=dx * self.scale * self.depth / 2,
                            y=dy * self.scale * self.height / 2,
                            z=dz * self.scale * self.width / 2,
                        )
                        corners.append(p)

            device_top_left_corner = corners[-1]
            self.body = PointCloud(corners)
            points = []
            assert self.mic_position_anchor in [None, 'top_left']
            for (tag, (w, h, d)) in self.mic_position.items():
                # this puts the origin in the middle of the body
                # with the y axis along the longest dimension of the phone
                # and the x axis in the look direction (as per our convention)
                points.append(
                    (
                        Point(x=d - self.depth, y=-h, z=w - self.width, tag=tag)
                        * self.scale
                        + device_top_left_corner
                    )
                )
            self.mics = DeviceGeometry(PointCloud(points))

    @property
    def body_trace(self):
        trace = None
        if isinstance(self.body, PointCloud):
            trace = self.body.mesh3d(
                polar=False,
                color="gray",
                opacity=0.4,
                name=self.name,
                hoverinfo="skip",
                showlegend=True,
                showscale=False,
            )
        elif self.body is not None:
            x, y, z, i, j, k = self.body
            trace = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color="gray",
                opacity=0.4,
                name=self.name,
                hoverinfo="skip",
                showlegend=True,
                showscale=False,
            )
        return trace

    def plot_locations(self) -> go.Figure:
        return self.mics.plot_locations(body_trace=self.body_trace)

    @classmethod
    def from_spec(cls, spec, spec_dir=None):
        if isinstance(spec, str) or isinstance(spec, Path):
            with open(spec, "rt") as fobj:
                spec = json.loads(fobj.read())
        assert spec["kind"] in ["brick", "stl"]
        # assert spec["mics"]["anchor"] == "top_left"
        if spec_dir is None:
            spec_dir = Path(__file__).parent.parent / 'data' / 'devices'
        stl_file = None
        if spec["kind"] == "stl":
            stl_file = spec_dir / spec["stl_file"]
        return cls(
            name=spec["name"],
            height=spec.get("height", None),
            width=spec.get("width", None),
            depth=spec.get("depth", None),
            mic_order=spec["mics"]["order"],
            mic_position=spec["mics"]["position"],
            mic_position_anchor=spec["mics"]["anchor"],
            stl_file=stl_file,
        )

    def populate_cells(self, cells):
        # it's a brick
        if isinstance(self.body, PointCloud):
            cells[:, :, :] = True
        else:
            mesh_obj = mesh.Mesh.from_file(self.stl_file)
            org_mesh = np.hstack(
                (
                    mesh_obj.v0[:, np.newaxis],
                    mesh_obj.v1[:, np.newaxis],
                    mesh_obj.v2[:, np.newaxis],
                )
            )
            vol, scale, shift = convert_meshes(
                [org_mesh], resolution=cells.shape[2], parallel=True
            )
            vol = vol.transpose(2, 1, 0)
            cells[np.nonzero(vol)] = True
