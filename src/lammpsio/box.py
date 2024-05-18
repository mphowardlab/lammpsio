import numpy


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
        v = numpy.array(value, ndmin=1, copy=False, dtype=float)
        if v.shape == (9,):
            return Box(v[:3], v[3:6], v[6:])
        elif v.shape == (6,):
            return Box(v[:3], v[3:])
        else:
            raise TypeError(f"Unable to cast boxlike object with shape {v.shape}")

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

    @property
    def _vectors(self):
        v = numpy.diag(self.high - self.low)
        if self.tilt is not None:
            v[1, 0] = self.tilt[0]
            v[2, :2] = self.tilt[1:]
        return v

    def unscale(self, scaled_position):
        return scaled_position @ self._vectors + self.low

    def scale(self, position):
        return (position - self.low) @ numpy.linalg.inv(self._vectors)

    def unwrap_scaled(self, scaled_position, image):
        return scaled_position + image

    def wrap_scaled(self, scaled_position):
        image = numpy.floor(scaled_position)
        return scaled_position - image, image

    def unwrap(self, position, image):
        return position + image @ self._vectors

    def wrap(self, position):
        position_wrapped_scaled, image = self.wrap_scaled(self.scale(position))
        return self.unscale(position_wrapped_scaled), image
