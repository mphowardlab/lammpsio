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

    def to_matrix(self):
        """Convert a :class:`Box` to an upper triangular matrix.

        Parameters
        ----------
        box : :class:`Box`
            The box to convert.

        Returns
        -------
        :class:`numpy.ndarray`
            Upper triangular matrix in LAMMPS style::

                [[lx, xy, xz],
                 [0, ly, yz],
                 [0, 0, lz]]

        """
        low = self.low
        high = self.high
        tilt = self.tilt if self.tilt is not None else [0, 0, 0]

        return (
            numpy.array(
                [
                    [high[0] - low[0], tilt[0], tilt[1]],
                    [0, high[1] - low[1], tilt[2]],
                    [0, 0, high[2] - low[2]],
                ]
            ),
            low,
        )

    @classmethod
    def from_matrix(cls, low, matrix, force_triclinic=False):
        """Create a Box from low and matrix.

        Parameters
        ----------
        low : list
            Origin of the box.
        matrix : :class:`numpy.ndarray`
            Upper triangular matrix in LAMMPS style::

                [[lx, xy, xz],
                 [0, ly, yz],
                 [0, 0, lz]]
        force_triclinic : bool
            If ``True``, forces the box to be triclinic even if the tilt
            factors are zero. Default is ``False``.

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

        # Calculate high from the matrix
        high = low + numpy.diag(arr)

        # Extract tilt factors
        tilt = [arr[0, 1], arr[0, 2], arr[1, 2]]
        if not force_triclinic and not numpy.any(tilt):
            tilt = None

        return cls(low, high, tilt)

    def to_hoomd_convention(self):
        """Convert a :class:`Box` to HOOMD-blue convention.

        Parameters
        ----------
        box : :class:`Box`
            The box to convert.

        Returns
        -------
        :class:`numpy.ndarray`
            A matrix of box dimensions in HOOMD-blue convention.

        """
        L = self.high - self.low
        if self.tilt is not None:
            tilt = self.tilt.copy()
            tilt[0] /= L[1]
            tilt[1:] /= L[2]
        else:
            tilt = [0, 0, 0]

        return numpy.concatenate((L, tilt))

    @classmethod
    def from_hoomd_convention(
        cls, box_data, low=None, force_triclinic=False, dimensions=None
    ):
        """Convert box data in HOOMD-blue convention to LAMMPS-convention

        Parameters
        ----------
        box_data : list
            An array of box dimensions in HOOMD-blue convention.
        low : list
            Origin of the box. If ``None``, the box is centered at the origin.
            Default is ``None``.
        force_triclinic : bool
            If ``True``, forces the box to be triclinic even if the tilt
            factors are zero. Default is ``False``.
        dimensions : int
            The number of dimensions of the box. If ``None``, it is inferred
            from the box data. Default is ``None``.

        Returns
        -------
        box : :class:`Box`
            A simulation box in LAMMPS convention.
        """
        if box_data.shape != (6,):
            raise TypeError("Box data must be a 6-tuple")
        if dimensions is None:
            dimensions = 3 if box_data[2] != 0 else 2
        if dimensions not in (2, 3):
            raise ValueError("Dimensions must be 2 or 3")

        L = box_data[:3]
        tilt = box_data[3:]
        if dimensions == 3:
            tilt[0] *= L[1]
            tilt[1:] *= L[2]
        elif dimensions == 2:
            tilt[0] *= L[1]
            tilt[1] = 0
            tilt[2] = 0
            # HOOMD boxes can have Lz = 0, but LAMMPS does not allow this.
            if L[2] == 0:
                L[2] = 1.0

        matrix = numpy.array(
            [[L[0], tilt[0], tilt[1]], [0, L[1], tilt[2]], [0, 0, L[2]]]
        )

        # center the box if low is not provided
        if low is None:
            low = -0.5 * numpy.sum(matrix, axis=1)

        return Box.from_matrix(low, matrix, force_triclinic)

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
