import numpy as np
from enum import Enum


class WXYTransform(Enum):
    """
    A set of transforms for first-order planar ambisonics, derived from
    David McGrath's spatial capture cookbook.

    Attributes:
        A numpy.ndarray: The matrix which transforms :math:`WXY` to this format
        channel_names List[str]: List of channel names of this format

    """

    WXY = (
        1,
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        ["W", "X", "Y"],
    )
    """Omni-directional microphone signal :math:`W` along with two dipole (figure eight) microphone signals :math:`X,Y`
    that have their maximum positive gains aligned with the :math:`X` and :math:`Y` axes respectively."""

    LRS = (
        1 / 2,
        [
            [1, 1 / 2, (3**0.5) / 2],
            [1, 1 / 2, -(3**0.5) / 2],
            [1, -1, 0],
        ],
        ["L", "R", "S"],
    )
    """Three cardioid microphone signals at azimuth angles of 60, −60 and 180 respectively."""

    WT = (
        1.0,
        [
            [1, 0, 0],
            [0, 1, 1j],
        ],
        ["W*", "T"],
    )
    """
    A :math:`WT` stream is composed of 2 audio signals :math:`(W,T)` such that, for plane waves incident
    at the microphone from azimuth :math:`\\varphi` in the horizontal plane,
    the amplitudes of the :math:`W` and :math:`T` signals are equal, but the :math:`T` signal is advanced in phase by θ,
    relative to :math:`W`. (Note: the letter :math:`T` is used because we think of the :math:`T` signal a being
    a Twisted version of the :math:`W` signal, in the sense that :math:`T` incorporates a phase shift that
    twists around from 0 to 360 as the azimuth angle varies from 0 to 360)."""

    ABCD = (
        1 / 2,
        [
            [1, 2**0.5, 2**0.5],
            [1, -(2**0.5), -(2**0.5)],
            [1, 2**0.5, -(2**0.5)],
            [1, -(2**0.5), 2**0.5],
        ],
        ["A", "B", "C", "D"],
    )

    FGH = (
        1 / 2,
        [
            [1, (-np.sqrt(6) + np.sqrt(2)) / 4, (np.sqrt(6) + np.sqrt(2)) / 4],
            [1, (np.sqrt(6) - np.sqrt(2)) / 4, (-np.sqrt(6) - np.sqrt(2)) / 4],
            [1, -1 / np.sqrt(2), -1 / np.sqrt(2)],
        ],
        ["F", "G", "H"],
    )

    LtRt = (
        1 / (2 * np.sqrt(2)),
        [
            [1 + 1j, 1 - 1j, 1 + 1j],
            [1 - 1j, 1 + 1j, -1 + 1j],
        ],
        ["Lt", "Rt"],
    )
    """
    An :math:`LtRt` stream is composed of 2 audio signals :math:`(Lt,Rt)` that, for plane waves
    incident at the microphone from azimuth :math:`\\varphi` in the horizontal plane, are equivalent to:
    
    .. math::

        \\begin{aligned} 
            Lt &= W \\times e ^ {j\\Phi(\\varphi)} \\times \\cos \\left( 2\\varphi − \\frac{\\pi}{4} \\right) \\\\ 
            Rt &= W \\times e ^ {j\\Phi(\\varphi)} \\times \\cos \\left( 2\\varphi + \\frac{\\pi}{4} \\right)
        \\end{aligned}
    
    where :math:`\\Phi` is any unity-gain phase shift that is a function of azimuth, :math:`\\varphi`.
    """

    BH3000 = (
        1.0,
        [
            [0.35355, 0.35355, 0.35355],
            [0.40825, -0.20412, -0.20412],
            [0.00000, 0.35355, -0.35355],
        ],
        ["M1", "M2", "M3"],
    )
    """
    Beehive format - see ``https://confluence.dolby.net/kb/x/qwY6DQ``
    """

    def __init__(self, scale, A, channel_names):

        self.A = scale * np.array(A)
        self.channel_names = channel_names

    @classmethod
    def from_channel_name(cls, name):
        """Return the format corresponding to the given channel name"""
        for xfm in cls:
            if name in xfm.channel_names:
                return xfm
        return None

    @classmethod
    def all_channel_names(cls):
        names = []
        for xfm in cls:
            names.extend(xfm.channel_names)
        return names
