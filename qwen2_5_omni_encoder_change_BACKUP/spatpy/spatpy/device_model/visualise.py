def main():
    import numpy as np
    from mayavi import mlab
    from mayavi.mlab import contour3d

    trace = np.load("find_x3_pro_back_mic_trace.npy")
    decimate_by = 3
    x, y, z = np.mgrid[
        0 : trace.shape[1] : decimate_by,
        0 : trace.shape[2] : decimate_by,
        0 : trace.shape[3] : decimate_by,
    ]

    view_distance = 2 * trace.shape[1] / 3
    trace = trace[:, x, y, z]
    c = contour3d(
        x,
        y,
        z,
        trace[0, :, :, :],
        contours=20,
        transparent=True,
    )

    @mlab.animate(delay=100)
    def anim():
        mlab.view(distance=view_distance)
        for i in range(1, trace.shape[0]):
            c.mlab_source.set(scalars=trace[i, :, :, :])
            yield

    anim()
    mlab.show()


if __name__ == "__main__":
    main()
