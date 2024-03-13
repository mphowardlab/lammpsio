import numpy


class Topology:
    def __init__(self, N, num_members):
        self._N = N
        self._num_members = num_members

        self._id = None
        self._typeid = None
        self._members = None
        self._num_types = None

    @property
    def N(self):
        """:class:`int`: number of connections."""
        return self._N

    @property
    def id(self):
        """:class:`numpy.ndarray`: IDs."""
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
            self._typeid = numpy.ones(self.N, dtype=int)
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
        """:class:`numpy.ndarray`: Connection members."""
        if not self.has_members():
            self._members = numpy.ones((self.N, self._num_members), dtype=int)
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

    @property
    def num_types(self):
        """:class:`Int`: num_types."""
        if not self.has_num_types():
            self._num_types = numpy.amax(numpy.unique(self.typeid))
        return self._num_types

    @num_types.setter
    def num_types(self, value):
        if value is not None:
            self._num_types = int(value)
        else:
            self._num_types = None

    def has_num_types(self):
        """Check if configuration has num_types.

        Returns
        -------
        bool
            True if connection num_types have been initialized.

        """
        return self._num_types is not None

    def reorder(self, order, check_order=True):
        """Reorder the connections in place.

        Parameters
        ----------
        order : list
            New order of indexes.
        check_order : bool
            If true, validate the new ``order`` before applying it.

        """
        # sanity check the sorting order before applying it
        if check_order and self.N > 1:
            sorted_order = numpy.sort(order)
            if (
                sorted_order[0] != 0
                or sorted_order[-1] != self.N - 1
                or not numpy.all(sorted_order[1:] == sorted_order[:-1] + 1)
            ):
                raise ValueError("New order must be an array from 0 to N-1")

        if self.has_id():
            self._id = self._id[order]
        if self.has_typeid():
            self._typeid = self._typeid[order]
        if self.has_members():
            self._members = self._members[order]


class Bonds(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=2)


class Angles(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=3)


class Dihedrals(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=4)


class Impropers(Topology):
    def __init__(self, N):
        super().__init__(N=N, num_members=4)
