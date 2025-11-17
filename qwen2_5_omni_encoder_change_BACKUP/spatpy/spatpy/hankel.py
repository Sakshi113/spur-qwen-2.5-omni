import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def HankelFunction(x, n=None, apply_correct_phase=False):
    """
    Spherical Hankel function (second kind) of order n.

    The default behaviour of this function deviates from the 'standard'
    :math:`h ^{(2)} _n(x)` function by a phase shift that is a multiple of :math:`\\frac{\\pi}{2}`, but the
    resulting Hankel function will have the property that the far-field
    response is the same for all orders (in the limit as :math:`x \\rightarrow \\infty`).

    However, if ``apply_correct_phase`` is set, the result matches the
    `definition of the Spherical Hankel function <http://wikipedia.org/wiki/Bessel_function#Spherical_Hankel_functions>`_ of the second kind.

    Args:
        x: may be an array, of any size/shape, of non-negative real values
        n: must be a scalar or array of integers between 0 and 10000 (n < 0 is permitted if `apply_correct_phase` is set).

    If ``x`` and ``n`` are both arrays, they are combined together by binary-singleton
    expansion.

    When (real) ``x`` > 0, this implementation is equivalent to the MATLAB:

    .. code-block::

        HankelFunction(x, n ,true) = conj((-1i)^(1+n)*besselh(n+0.5,2,x,0)*sqrt(pi/2/x))


    but it is implemented with straightforward maths formulae.

    When an omni-directional pressure cosine-wave is emitted from a spherical
    sound source, if we assume the peak amplitude at 1m is unity, the we specicfy
    the far-field response at radius r as a complex exponential combined with a
    :math:`\\frac{1}{r}` roll-off over distance:

    .. math::

        \\textrm{response}(r) = \\frac{1}{r}e^{-ikr}

    This is the same as the 0th-order Spherical Hankel Function:

    .. code-block:: python

        HankelFunction(k*r, 0) * k

    The higher-order Spherical Hankel Functions describe the response for an
    n-th order radiation pattern. In the far field, the higher order patterns
    converge to the same values as the 0th order:

    .. math::

        \\lim_{kr \\to \\infty} \\textrm{HankelFunction}(kr, n) = \\textrm{HankelFunction}(kr, 0)

    We can think of the params as:

    * :math:`kr` = the distance from the origin, measured in radians
    * :math:`k` = the wavenumber (radians/m)
    * :math:`n`  = the order of the radiating pattern

    Example:

    .. code-block:: python

        freq = 200 # Hz
        k = 2*pi*freq/343.3 # 343.3 = speed of sound (m/s)
        r = 5 # m
        h0 = HankelFunction( k*r, 0 ) * k
        h0 = 0.1708 + 0.1041j # 1st-order response at 5m
        print(abs(h0)) # 0.2000,  Gain at 5m = 1/5
        h3 = HankelFunction( k*r, 3 ) * k
        h3 = 0.1970 + 0.0438j # 3rd-order response at 5m
        print(abs(h3)) # 0.2018, Gain at 5m > 1/5

    To give you an idea of how 'strong' these Hankel functions are, when a sound
    source at 1m distance is received at a Higher-Order-Ambisonics microphone, the
    3rd-order HOA components will be boosted (relative to the far-field) by:

    - 1dB at 300Hz  <-- so, above 300Hz, we can consider 1m to be 'far field'
    - 10dB at 106Hz
    - 20dB at 66Hz
    - 50dB at 20Hz

    The responses of 0th, 1st, 2nd and 3rd order patterns at 1m and 5m may be plotted by
    calling :obj:`plot_demo`.

    The Spherical Bessel functions can be computed as:

    .. code-block:: python

        j_n = lambda x: np.real(HankelFunction(x, n, True))  # first kind
        y_n = lambda x: -np.imag(HankelFunction(x, n, True))  # second kind

    The Spherical Hankel function of the first kind is:

    .. code-block:: python

        h1_n = lambda x: np.conj(HankelFunction(x, n, True))

    - Authored by D.McGrath, 2013
    - Modified in Jan 2015 to correct a phase-error
    - Modified in May 2019 to provide helpful plot when function called with no args
    - Modified in Jul 2020 to allow for opt_apply_correct_phase and n<0
    """
    x = np.atleast_2d(x)
    n = np.atleast_2d(n).T
    assert not (
        np.any(np.real(x) < 0) or np.any(np.imag(x) != 0)
    ), "Expecting x to contain only non-negative real values"
    if n is None:
        n = np.zeros(1, dtype=np.int32)
    assert np.all(
        n == n.astype(np.int32)
    ), "Expecting n to contain only real integers"
    if apply_correct_phase:
        PhaseN = (1j) ** np.mod(n + 1, 4)
    else:
        PhaseN = 0 * n + 1
    n = np.abs(n + 0.5) - 0.5  # Trick for handling negative values of n
    n = n.astype(np.int32)
    assert np.all(
        n <= 10000
    ), "Expecting magnitude of n to be no greater than 10000"
    assert np.all(n >= 0) or apply_correct_phase, (
        "Don't provide negative values of n without also setting"
        " apply_correct_phase"
    )
    y = np.ones((len(n), x.shape[1]), dtype=np.complex128)
    f = np.ones_like(y)
    fi = -1j / (2 * x)
    for m in range(1, np.max(n) + 1):
        f *= fi * (n + m) * (n - m + 1) / m
        y += f

    # optionally, apply a phase shift (but it's simpler if we don't). By
    # leaving the following line commented out, the behaviour of this function
    # will now deviate from the 'standard' by a phase shift that is a multiple
    # of pi/2, but the resulting Hankel function will now have the property
    # that the far-field response is the same for all orders (in the limit as
    # x->inf).

    # y = ( (1i^(n+1)) *y)

    # Now for the phase shift and attunation over distance:
    y *= np.exp(-1j * x) / x * PhaseN

    return y.T


def hankel_db(r, fs, c=343.3, order=3):
    k = 2 * np.pi * fs / c  # Wave number (rad/s)
    h = HankelFunction(k * r, np.arange(order + 1))
    return 20 * np.log10(np.abs(h / h[:, 0:1]))


def plot_demo():
    """Make a plot to illustrate the principle of Hankel functions.
    
    .. plotly::
        :include-source:

        from spatpy import hankel
        hankel.plot_demo()
    
    """
    fs = 10 ** np.linspace(1, 3, 40)  # 10Hz...1kHz
    radii = [1, 5]
    fig = make_subplots(
        rows=len(radii),
        shared_xaxes=True,
        subplot_titles=[
            f"Magnitude relative to zeroth order at {r}m"
            for r in radii
        ],
    )
    for (i, r) in enumerate(radii):
        y = hankel_db(r=r, fs=fs)
        for j in range(y.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=fs,
                    y=y[:, j],
                    name=f"order {j}",
                    legendgrouptitle=dict(text=f"r = {r}m"),
                    legendgroup=str(i),
                ),
                row=i + 1,
                col=1,
            )
    fig.update_yaxes(title="dB")
    fig.update_xaxes(title="Frequency (Hz)", type="log")
    return fig


if __name__ == "__main__":
    fig = plot_demo()
    fig.show()
    print()
