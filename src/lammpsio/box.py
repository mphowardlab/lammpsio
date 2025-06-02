import numpy

from . import _compatibility


class Box:
    """Triclinic simulation box.
    
    ``lammpsio.Box`` defines the simulation box using LAMMPS's convention. 
    In LAMMPS, the simulation box is specified by three parameters: `low`, `high`, and `tilt`. 
    The `low` parameter defines the origin (lower corner) of the box, while `high` specifies how far the box extends along each axis. 
    The `tilt` parameter contains tilt factors (xy, xz, yz) that skew the edges to create non-orthorhombic simulation boxes.

    These parameters can be transformed into a box matrix with the following form:

    .. math:: 
        \\begin{bmatrix}
            1 & b_x & c_x \\\\
            0 & b_y & c_y \\\\
            0 & 0 & c_z
        \\end{bmatrix},
    
    where each column of this matrix represents one of the three edge vectors (**A**, **B**, & **C**) that define the triclinic simulation box, 
    and each of the matrix elements are derived from the `low`, `high`, and `tilt` values. 
    The diagonal elements ($a_x$, $b_y$, $c_z$) represent the box lengths along each axis, while the off-diagonal elements come from the tilt factors. 
    For more details on how to convert between the LAMMPS parameters and 
    box matrix see the [LAMMPS documentation](https://docs.lammps.org/Howto_triclinic.html#transformation-from-general-to-restricted-triclinic-boxes).

    For our orthorhombic unit cell, all tilt factors are zero, so `tilt` has the default value of `None`, resulting in a simple rectangular box. 
    We choose `low` to be at ``[0, 0, 0]`` and `high` is the diagonal values of the box. 

    Parameters
    ----------
    low : list
        Origin of the box
    high : list
        "High" of the box, used to compute edge lengths.
    tilt : list
        Tilt factors ``xy``, ``xz``, and ``yz`` for a triclinic box.
        Default of ``None`` is a strictly orthorhombic box.

    Examples
    --------

    Construct a triclinic simulation box:

    .. code-block:: python

        box = lammpsio.Box([-5.0, -10.0, 0.0], [1.0, 10.0, 8.0], [1.0, -2.0, 0.5])

    """

    def __init__(self, low, high, tilt=None):
        self.low = low
        self.high = high
        self.tilt = tilt

    @classmethod
    def cast(cls, value):
        """Cast an array to a `Box`.

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
            A simulation box matching the array.

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

    @property
    def low(self):
        """:class:`numpy.ndarray`: Box low.

        The low of the box is used as the origin of the box.
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
        """:class:`numpy.ndarray`: Box high.

        The high of the box is used to compute the edge lengths of the box.
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
        """:class:`numpy.ndarray`: Box tilt factors.

        The tilt factors, ``xy``, ``xz``, and ``yz`` are used to define the
        shape of the box. The default of ``None`` is a strictly orthorhombic.

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
