import numpy

from . import _compatibility


class Box:
    """Triclinic simulation box.

    The convention for defining the bounds of the box is based on
    `LAMMPS <https://docs.lammps.org/Howto_triclinic.html>`_. This
    means that the lower corner of the box is placed at ``low``, and
    the size and shape of the box is determined by ``high`` and ``tilt``.

    Parameters
    ----------
    low : list
        Origin of the box
    high : list
        "High" of the box, used to compute edge lengths.
    tilt : list
        Tilt factors ``xy``, ``xz``, and ``yz`` for a triclinic box.
        Default of ``None`` is a strictly orthorhombic box.

    """

    def __init__(self, low, high, tilt=None):
        self.low = low
        self.high = high
        self.tilt = tilt

    @classmethod
    def cast(cls, value):
        """Cast an array to a :class:`Box`.

        If ``value`` has 6 elements, it is unpacked as an orthorhombic box::

            x_lo, y_lo, z_lo, x_hi, y_hi, z_hi = value

        If ``value`` has 9 elements, it is unpacked as a triclinic box::

            x_lo, y_lo, z_lo, x_hi, y_hi, z_hi, xy, xz, yz = value

        Parameters
        ----------
        value : list
            6-element or 9-element array representing the box.

        Returns
        -------
        :class:`Box`
            A simulation box matching the array.

        """
        if isinstance(value, Box):
            return value
        v = numpy.array(
            value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=float
        )
        if v.shape == (9,):
            return Box(v[:3], v[3:6], v[6:])
        elif v.shape == (6,):
            return Box(v[:3], v[3:])
        else:
            raise TypeError(f"Unable to cast boxlike object with shape {v.shape}")

    @classmethod
    def from_matrix(cls, low, matrix):
        """Create a Box from low and matrix.

        Parameters
        ----------
        low : list
            Origin of the box.
        matrix : :class:`numpy.ndarray`
            Upper triangular matrix in LAMMPS style:
            [[lx, xy, xz],
             [0, ly, yz],
             [0, 0, lz]]

        Returns
        -------
        :class:`Box`
            A simulation box.

        Raises
        ------
        TypeError
            If `low` is not length 3.
        TypeError
            If `matrix` is not a 3x3 array.
        ValueError
            If `matrix` is not upper triangular.

        """
        low = numpy.array(low, dtype=float)
        arr = numpy.array(matrix, dtype=float)

        if low.shape != (3,):
            raise TypeError("Low must be a 3-tuple")
        if arr.shape != (3, 3):
            raise TypeError("Box matrix must be a 3x3 array")
        if arr[1, 0] != 0 or arr[2, 0] != 0 or arr[2, 1] != 0:
            raise ValueError("Box matrix must be upper triangular")

        # Extract diagonal elements for box lengths
        lx, ly, lz = arr[0, 0], arr[1, 1], arr[2, 2]
        high = low + [lx, ly, lz]

        # Extract tilt factors
        xy, xz, yz = arr[0, 1], arr[0, 2], arr[1, 2]
        tilt = [xy, xz, yz] if numpy.any([xy, xz, yz]) else None

        return cls(low, high, tilt)

    @property
    def low(self):
        """:class:`numpy.ndarray`: Box low."""
        return self._low

    @low.setter
    def low(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=float)
        if v.shape != (3,):
            raise TypeError("Low must be a 3-tuple")
        self._low = v

    @property
    def high(self):
        """:class:`numpy.ndarray`: Box high."""
        return self._high

    @high.setter
    def high(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=float)
        if v.shape != (3,):
            raise TypeError("High must be a 3-tuple")
        self._high = v

    @property
    def tilt(self):
        """:class:`numpy.ndarray`: Box tilt factors."""
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        v = value
        if v is not None:
            v = numpy.array(v, ndmin=1, copy=True, dtype=float)
            if v.shape != (3,):
                raise TypeError("Tilt must be a 3-tuple")
        self._tilt = v
