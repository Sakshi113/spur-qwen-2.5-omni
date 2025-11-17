from dataclasses import dataclass
from typing import Optional
import xarray as xr
import numpy as np
from spatpy.geometry import PointCloud, Point
from spatpy.device_model import DeviceModel


@dataclass
class VoxelGrid:
    nx: int
    ny: int
    nz: int
    scale: float = 1e-3
    name: Optional[str] = None

    def __post_init__(self):
        self.cells = xr.DataArray(
            np.zeros(
                (2 * self.nx + 1, 2 * self.ny + 1, 2 * self.nz + 1),
                dtype=np.bool8,
            ),
            name=self.name,
            dims=["x", "y", "z"],
            coords=dict(
                x=self.scale * (np.arange(2 * self.nx + 1) - self.nx),
                y=self.scale * (np.arange(2 * self.ny + 1) - self.ny),
                z=self.scale * (np.arange(2 * self.nz + 1) - self.nz),
            ),
            attrs=dict(scale=self.scale),
        )

    def xyz_indices(self, x, y, z):
        return (
            np.array(
                [
                    round(x / self.scale),
                    round(y / self.scale),
                    round(z / self.scale),
                ]
            )
            + self.origin
        )

    @property
    def origin(self):
        return np.array([self.nx, self.ny, self.nz])

    def point_indices(
        self,
        points: PointCloud,
        centre: Optional[Point] = None,
        radius: float = 1.0,
    ):
        indices = np.vstack(
            [
                np.floor(points.x * radius / self.scale).astype(np.int32),
                np.floor(points.y * radius / self.scale).astype(np.int32),
                np.floor(points.z * radius / self.scale).astype(np.int32),
            ]
        ) + np.expand_dims(self.origin, 1)
        if centre is not None:
            indices += np.expand_dims(
                self.xyz_indices(
                    centre.x.item(), centre.y.item(), centre.z.item()
                ),
                1,
            )
        return indices


    @classmethod
    def from_device_model(cls, model: DeviceModel, resolution=1e-3):
        scale = model.scale / resolution
        grid = cls(
            nx=int(np.floor(model.depth * scale / 2)),
            ny=int(np.floor(model.height * scale / 2)),
            nz=int(np.floor(model.width * scale / 2)),
            scale=resolution,
            name=model.name,
        )
        model.populate_cells(grid.cells.values)
        return grid
