import numpy as np
import argparse
import sys


def compute_gradient(az, el):
    P = len(az)

    cosColat = np.cos(el)
    sinColat = np.sin(el)
    cosAz = np.cos(az)
    sinAz = np.sin(az)

    cosAzCosColat = cosAz * cosColat
    cosAzSinColat = cosAz * sinColat
    sinAzSinColat = sinAz * sinColat
    sinAzCosColat = cosColat * sinAz

    d_az = np.zeros(P)
    d_el = np.zeros(P)
    n = 0
    for i in range(P):
        j = np.arange(i + 1, P)
        n += len(j)
        temp1 = sinAzSinColat[i] - sinAzSinColat[j]
        temp2 = cosColat[i] - cosColat[j]
        temp3 = cosAzSinColat[i] - cosAzSinColat[j]
        temp4 = 1 / (temp1 * temp1 + temp2 * temp2 + temp3 * temp3)
        temp5 = temp4 * temp4
        # Differentiate wrt theta1
        d_el[i] -= np.sum(
            (
                2 * cosAzCosColat[i] * temp3
                - 2 * sinColat[i] * temp2
                + 2 * sinAzCosColat[i] * temp1
            )
            * temp5
        )
        # Differentiate wrt phi1
        d_az[i] += np.sum(
            (2 * sinAzSinColat[i] * temp3 - 2 * cosAzSinColat[i] * temp1)
            * temp5
        )
        # Differentiate wrt theta2
        d_el[j] += (
            2 * cosAzCosColat[j] * temp3
            - 2 * sinColat[j] * temp2
            + 2 * sinAzCosColat[j] * temp1
        ) * temp5
        # Differentiate wrt phi2
        d_az[j] -= (
            2 * sinAzSinColat[j] * temp3 - 2 * cosAzSinColat[j] * temp1
        ) * temp5

    d_az /= n
    d_el /= n
    return d_az, d_el


def uniform_sample_sphere(r=1, n_points=25, n_iters=50):
    # approximate raised cosine distribution
    el = np.random.normal(
        np.pi / 2, np.pi / 2 * np.sqrt(1 / 3 - 2 / (np.pi**2)), n_points
    )
    # uniform distribution
    az = np.random.rand(n_points) * 2 * np.pi

    for i in range(n_iters):
        d_az, d_el = compute_gradient(az, el)
        az -= d_az
        el -= d_el

    x = r * np.cos(az) * np.sin(el)
    y = r * np.sin(az) * np.sin(el)
    z = r * np.cos(el)
    return (x, y, z)


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly sample points on the sphere. Prints x,y,z to stdout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    samp = parser.add_argument_group("Sampling options")
    samp.add_argument(
        "--radius",
        type=float,
        help="radius of the sphere (units)",
        default=1.0,
    )
    samp.add_argument(
        "--points", type=int, help="desired number of points", default=80
    )
    samp.add_argument(
        "--iterations", help="number of iterations", type=int, default=50
    )
    output = parser.add_argument_group("Output options")
    output.add_argument("--format", default="%.18e", help="format specifier")
    output.add_argument("--delimiter", default=",", help="delimiter")
    output.add_argument("--header", default="", help="header row")
    output.add_argument("--footer", default="", help="footer row")
    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = make_parser()
    options = parser.parse_args(args)
    x, y, z = uniform_sample_sphere(
        r=options.radius, n_points=options.points, n_iters=options.iterations
    )
    np.savetxt(
        sys.stdout,
        np.vstack((x, y, z)).T,
        delimiter=options.delimiter,
        fmt=options.format,
        header=options.header,
        footer=options.footer,
    )


if __name__ == "__main__":
    main()
