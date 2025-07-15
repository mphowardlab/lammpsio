import numpy

from . import _compatibility


class Box:
    r"""Simulation box.

    In LAMMPS, the simulation box is specified by three parameters: `low`,
    `high`, and `tilt`. `low` defines the origin (lower corner) of the box,
    while `high` specifies how far the box extends along each axis. The
    difference between `high` and `low` gives three lengths $L_x$, $L_y$, and
    $L_z$. `tilt` has three factors ($L_{xy}$, $L_{xz}$, $L_{yz}$) that skew the
    edges to create non-orthorhombic simulation boxes. These parameters define
    a box matrix consisting of three lattice vectors **a**, **b**, and **c**:

    .. math::

        [\mathbf{a} \quad \mathbf{b} \quad \mathbf{c} ] =
        \begin{bmatrix}
            L_x & L_{xy} & L_{xz} \\
            0 & L_y & L_{yz} \\
            0 & 0 & L_z
        \end{bmatrix}

    For more details on how to convert between the LAMMPS parameters and box
    matrix see the `LAMMPS documentation
    <https://docs.lammps.org/Howto_triclinic.html#transformation-from-general-to-restricted-triclinic-boxes>`_.

    .. warning::

        `high` is the upper bound of the simulation box **only** when it is
        orthorhombic.

    Parameters
    ----------
    low : list
        Origin of the box
    high : list
        High parameter used to compute edge lengths.
    tilt : list
        Tilt factors ``xy``, ``xz``, and ``yz`` for a triclinic box. Default of
        ``None`` is a strictly orthorhombic box, implying all are zero.

    Examples
    --------
    Construct a triclinic simulation box:

    .. code-block:: python

        box = lammpsio.Box([-5.0, -10.0, 0.0], [1.0, 10.0, 8.0], [1.0, -2.0, 0.5])

    The coordinates of the box from the range of [`low`, `high`] are:
    - x: [-5.0, 1.0]
    - y: [-10.0, 10.0]
    - z: [0.0, 8.0]

    Construct a orthorhombic simulation box:

    .. code-block:: python

        box = lammpsio.Box([-5.0, -10.0, 0.0], [1.0, 10.0, 8.0])

    """

    def __init__(self, low, high, tilt=None):
        self.low = low
        self.high = high
        self.tilt = tilt

    @classmethod
    def cast(cls, value):
        """Cast from an array.

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
        `Box`
            A simulation box.

        Examples
        --------
        Construct an orthorhombic simulation box by casting an array:

        .. code-block:: python

            box = lammpsio.Box.cast([-5.0, -10.0, 0.0, 1.0, 10.0, 8.0])

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
    def from_matrix(cls, low, matrix, force_triclinic=False):
        """Cast from an origin and matrix.

        Parameters
        ----------
        low : list
            Origin of the box.
        matrix : `numpy.ndarray`
            Box matrix.
        force_triclinic : bool
            If ``True``, forces the box to be triclinic even if the tilt
            factors are zero. Default is ``False``.

        Returns
        -------
        `Box`
            A simulation box.

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

    @property
    def low(self):
        """(3,) `numpy.ndarray` of `float`: Low parameter.

        The low of the box is the origin.
        """
        return self._low

    @low.setter
    def low(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=float)
        if v.shape != (3,):
            raise TypeError("Low must be a 3-tuple")
        self._low = v

    @property
    def high(self):
        """(3,) `numpy.ndarray` of `float`: High parameter.

        The high of the box is used to compute the lengths $L_x$, $L_y$, and
        $L_z$.
        """
        return self._high

    @high.setter
    def high(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=float)
        if v.shape != (3,):
            raise TypeError("High must be a 3-tuple")
        self._high = v

    @property
    def tilt(self):
        """(3,) `numpy.ndarray` of `float`: Tilt parameters.

        The 3 tilt factors, $L_{xy}$, $L_{xz}$, and $L_{yz}$ define the
        shape of the box. The default of ``None`` is strictly orthorhombic,
        meaning all are zero.
        """
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        v = value
        if v is not None:
            v = numpy.array(v, ndmin=1, copy=True, dtype=float)
            if v.shape != (3,):
                raise TypeError("Tilt must be a 3-tuple")
        self._tilt = v
