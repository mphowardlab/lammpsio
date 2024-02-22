import numpy


class Topology:
    def __init__(self, N, num_members):
        self._N = int(N)
        self._num_members = num_members

        self._id = None
        self._typeid = None
        self._members = None

    @property
    def N(self):
        return self._N

    @property
    def id(self):
        """:class:`numpy.ndarray`: Bond IDs."""
        if not self.has_id():
            self._id = numpy.arange(1, self.N + 1)
        return self._id

    @id.setter
    def id(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=int)
            if v.shape != (self.N,):
                raise TypeError("Ids must be a size N array")
            if not self.has_id():
                self._id = numpy.arange(1, self.N + 1)
            numpy.copyto(self._id, v)
        else:
            self._id = None

    def has_id(self):
        """Check if configuration has connection IDs.

        Returns
        -------
        bool
            True if connection IDs have been initialized.

        """
        return self._id is not None

    @property
    def typeid(self):
        """:class:`numpy.ndarray`: Bond typeids."""
        if not self.has_typeid():
            self._typeid = numpy.zeros(self.N, dtype=int)
        return self._typeid

    @typeid.setter
    def typeid(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=int)
            if v.shape != (self.N,):
                raise TypeError("typeids must be a size N array")
            self._typeid = v
        else:
            self._typeid = None

    def has_typeid(self):
        """Check if configuration has connection typeids.

        Returns
        -------
        bool
            True if connection typeids have been initialized.

        """
        return self._typeid is not None

    @property
    def members(self):
        """:class:`numpy.ndarray`: Bond members."""
        if not self.has_members():
            self._members = numpy.zeros([self.N, self._num_members], dtype=int)
        return self._members

    @members.setter
    def members(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=2, copy=False, dtype=int)
            if v.shape != (self.N, self._num_members):
                raise TypeError("Members must be a size N x number of arrays array")
            self._members = v
        else:
            self._members = None

    def has_members(self):
        """Check if configuration has connection members.

        Returns
        -------
        bool
            True if particle members have been initialized.

        """
        return self._members is not None


class Bonds(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=2)


class Angles(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=3)
