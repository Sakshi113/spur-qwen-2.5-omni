if __name__ == "__main__":
    import numpy as np
    from mayavi import mlab
    from pathlib import Path
    from spatpy.device_model import DeviceModel
    from spatpy.device_model.pml import VoxelGrid
    import json

    device_dir = Path(__file__).parent.parent / "data" / "devices"
    phone = DeviceModel.from_spec(
        device_dir / "find_x5_pro.json", spec_dir=device_dir
    )
    grid_resolution_m = 1e-3
    body_voxels = VoxelGrid.from_device_model(
        phone, resolution=grid_resolution_m
    )
    fig = phone.plot_locations()
    fig.show()
    voxel_resolution = 1e-3

    scale = phone.scale / voxel_resolution
    cells = np.zeros(
        (
            int(phone.depth * scale),
            int(phone.height * scale),
            int(phone.width * scale),
        )
    )
    phone.populate_cells(cells)
    x, y, z = np.nonzero(cells)

    # https://stackoverflow.com/a/37863912
    mlab.points3d(x, y, z, mode="cube", scale_mode="vector")

    mlab.show()

    print()
