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

    The coordinates of the box having the following `low` and `high` lists::

        [x_lo, y_lo, z_lo] = [-5.0, -10.0, 0.0]

        [x_hi, y_hi, z_hi] = [1.0, 10.0, 8.0]

    The tilt factors ``[1.0, -2.0, 0.5]`` corresponds to
    ``xy``, ``xz``, and ``yz`` respectively.

    Construct a orthorhombic simulation box:

    .. code-block:: python

        box = lammpsio.Box([-5.0, -10.0, 0.0], [1.0, 10.0, 8.0])

    The box construction is same as before except the tilt
    factors are all zero or ``None``, meaning the box is orthorhombic.

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

        The array defines an orthorhombic box, with each element being cast
        to the `low` and `high` lists in the `Box` format.

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
        """Convert to an origin and matrix.

        Parameters
        ----------
        box : `Box`
            The box to convert.

        Returns
        -------
        list
            Origin of the box.
        `numpy.ndarray`
            Upper triangular matrix in LAMMPS style::

                [[lx, xy, xz],
                 [0, ly, yz],
                 [0, 0, lz]]

        Examples
        --------
        Convert a box to an origin and matrix:

        .. code-block:: python

            low, matrix = box.to_matrix()

        `low` is the origin of the box, and `matrix` is the upper triangular
        matrix that defines the box dimensions and tilt factors.

        """

        low = self.low
        high = self.high
        tilt = self.tilt if self.tilt is not None else [0, 0, 0]

        return (
            low,
            numpy.array(
                [
                    [high[0] - low[0], tilt[0], tilt[1]],
                    [0, high[1] - low[1], tilt[2]],
                    [0, 0, high[2] - low[2]],
                ]
            ),
        )

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

        Examples
        --------
        Construct a simulation box from a low and matrix:

        .. code-block:: python

            box_matrix = numpy.array([
                [1, 1.0, -2.0],
                [0, 1, 0.5],
                [0, 0, 1]
            ])

            box = lammpsio.Box.from_matrix(low=[0, 0, 0], matrix=box_matrix)

        This creates a triclinic box of unit length in each direction with origin
        at [0, 0, 0] and the tilt factors [1.0, -2.0, 0.5] corresponding to
        $L_{xy}$, $L_{xz}$, and $L_{yz}$ respectively.

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
        """Convert to HOOMD-blue convention.

        This convention for defining the box is based on
        `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/v5.0.1/hoomd/box.html>`_.

        Parameters
        ----------
        box : `Box`
            The box to convert.

        Returns
        -------
        :class:`numpy.ndarray`
            A matrix of box dimensions in HOOMD-blue convention.

        Examples
        --------
        Convert a box to HOOMD-blue convention:

        .. skip: next if(lammps == None, reason="lammps not installed")

        .. invisible-code-block: python

            filename = tmp_path / "atoms.lammpstrj"
            lmps = lammps.lammps(cmdargs=["-log", "none", "-nocite"])

            cmds = []
            cmds += ["units lj"]
            cmds += ["atom_style atomic"]
            cmds += ["boundary p p p"]

            box_length = 3

            cmds += [f"region box block 0 {box_length} 0 {box_length} 0 {box_length}"]
            cmds += ["create_box 1 box"]

            cmds += ["create_atoms 1 random 10 12345 box"]
            cmds += ["mass 1 1.0"]

            cmds += [f"dump 1 all custom 100 {filename} id x y z ix iy iz"]
            cmds += ["run 1"]

            lmps.commands_list(cmds)
            lmps.close()

        .. code-block:: python

            dump_file = lammpsio.DumpFile(filename)

            for snapshot in dump_file:
                hoomd_box = snapshot.box.to_hoomd_convention()

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
        """Cast from HOOMD-blue convention.

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
        box : `Box`
            A simulation box in LAMMPS convention.

        Examples
        --------
        Convert a box to the LAMMPS convention:

        .. code-block:: python

            lammps_box = snapshot.box.from_hoomd_convention(hoomd_box)

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

        return cls.from_matrix(low, matrix, force_triclinic)

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
