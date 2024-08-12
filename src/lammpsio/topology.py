import numpy

from . import _compatibility


class Topology:
    """Particle topology.

    Parameters
    ----------
    N : int
        Number of connections.
    num_members : int
        Number of members in a connection.
    num_types : int
        Number of connection types.

    Attributes
    ----------
    num_types : int
        Number of connection types.

    """

    def __init__(self, N, num_members, num_types=None):
        self._N = N
        self._num_members = num_members
        self.num_types = num_types

        self._id = None
        self._typeid = None
        self._members = None

    @property
    def N(self):
        """int: Number of connections."""
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
            v = numpy.array(
                value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=int
            )
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
        """:class:`numpy.ndarray`: Connection typeids."""
        if not self.has_typeid():
            self._typeid = numpy.ones(self.N, dtype=int)
        return self._typeid

    @typeid.setter
    def typeid(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=int
            )
            if v.shape != (self.N,):
                raise TypeError("typeids must be a size N array")
            if not self.has_typeid():
                self._typeid = numpy.ones(self.N, dtype=int)
            numpy.copyto(self._typeid, v)
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
            v = numpy.array(
                value, ndmin=2, copy=_compatibility.numpy_copy_if_needed, dtype=int
            )
            if v.shape != (self.N, self._num_members):
                raise TypeError("Members must be a size N x number of members array")
            if not self.has_members():
                self._members = numpy.ones((self.N, self._num_members), dtype=int)
            numpy.copyto(self._members, v)
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
        """int: Number of connection types"""
        if self._num_types is not None:
            return self._num_types
        else:
            if self.has_typeid():
                return numpy.amax(self.typeid)
            else:
                return 1

    @num_types.setter
    def num_types(self, value):
        if value is not None:
            self._num_types = int(value)
        else:
            self._num_types = None

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
    """Particle bonds.

    Parameters
    ----------
    N : int
        Number of bonds.
    num_types : int
        Number of bond types.

    """

    def __init__(self, N, num_types=None):
        super().__init__(N=N, num_members=2, num_types=num_types)


class Angles(Topology):
    """Particle angles.

    Parameters
    ----------
    N : int
        Number of angles.
    num_types : int
        Number of angle types.

    """

    def __init__(self, N, num_types=None):
        super().__init__(N=N, num_members=3, num_types=num_types)


class Dihedrals(Topology):
    """Particle dihedrals.

    Parameters
    ----------
    N : int
        Number of diehdrals.
    num_types : int
        Number of dihedral types.

    """

    def __init__(self, N, num_types=None):
        super().__init__(N=N, num_members=4, num_types=num_types)


class Impropers(Topology):
    """Particle improper dihedrals.

    Parameters
    ----------
    N : int
        Number of improper dihedrals.
    num_types : int
        Number of improper dihedral types.

    """

    def __init__(self, N, num_types=None):
        super().__init__(N=N, num_members=4, num_types=num_types)


class TypeMap:
    """Map between HOOMD types and snapshot typeids.

    Parameters
    ----------
    type_map : dict
        A map from the :attr:`Snapshot.typeid` to a HOOMD type.

    Attributes
    ----------
    type_map : dict
        A map from the :attr:`Snapshot.typeid` to a HOOMD type

    """

    def __init__(
        self,
        particle_type_map,
        bond_type_map=None,
        angle_type_map=None,
        dihedral_type_map=None,
        improper_type_map=None,
    ):
        self._particle_type_map = particle_type_map
        self._bond_type_map = bond_type_map
        self._angle_type_map = angle_type_map
        self._dihedral_type_map = dihedral_type_map
        self._improper_type_map = improper_type_map

    @property
    def particle_type_map(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD type."""
        return self._particle_type_map

    @particle_type_map.setter
    def particle_type_map(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("type_map must be a dictionary")
            self._particle_type_map = value
        else:
            self._particle_type_map = None

    def has_particle_type_map(self):
        """Check if TypeMap has a type map.

        Returns
        -------
        bool
            True if type map has been initialized.

        """
        return self._particle_type_map is not None

    @property
    def bond_type_map(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD bond type."""
        return self._bond_type_map

    @bond_type_map.setter
    def bond_type_map(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("bond_type_map must be a dictionary")
            self._bond_type_map = value
        else:
            self._bond_type_map = None

    def has_bond_type_map(self):
        """Check if TypeMap has a bond type map.

        Returns
        -------
        bool
            True if bond type map has been initialized.

        """
        return self._bond_type_map is not None

    @property
    def angle_type_map(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD angle type."""
        return self._angle_type_map

    @angle_type_map.setter
    def angle_type_map(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("angle_type_map must be a dictionary")
            self._angle_type_map = value
        else:
            self._angle_type_map = None

    def has_angle_type_map(self):
        """Check if TypeMap has an angle type map.

        Returns
        -------
        bool
            True if angle type map has been initialized.

        """
        return self._angle_type_map is not None

    @property
    def dihedral_type_map(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD dihedral type."""
        return self._dihedral_type_map

    @dihedral_type_map.setter
    def dihedral_type_map(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("dihedral_type_map must be a dictionary")
            self._dihedral_type_map = value
        else:
            self._dihedral_type_map = None

    def has_dihedral_type_map(self):
        """Check if TypeMap has a dihedral type map.

        Returns
        -------
        bool
            True if dihedral type map has been initialized.

        """
        return self._dihedral_type_map is not None

    @property
    def improper_type_map(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD improper type."""
        return self._improper_type_map

    @improper_type_map.setter
    def improper_type_map(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("improper_type_map must be a dictionary")
            self._improper_type_map = value
        else:
            self._improper_type_map = None

    def has_improper_type_map(self):
        """Check if TypeMap has an improper type map.

        Returns
        -------
        bool
            True if improper type map has been initialized.

        """
        return self._improper_type_map is not None

    @property
    def particle_types(self):
        """Get the HOOMD types.

        Returns
        -------
        list
            List of HOOMD types.

        """
        return list(self._particle_type_map.values())

    @property
    def bond_types(self):
        """Get the bond types.

        Returns
        -------
        list
            List of bond types.

        """
        if self.has_bond_type_map():
            return list(self._bond_type_map.values())
        else:
            raise ValueError("No bond type map found")

    @property
    def angle_types(self):
        """Get the angle types.

        Returns
        -------
        list
            List of angle types.

        """
        if self.has_type_map() and self.has_angle_type_map():
            return list(self._angle_type_map.values())
        else:
            return []

    @property
    def dihedral_types(self):
        """Get the dihedral types.

        Returns
        -------
        list
            List of dihedral types.

        """
        if self.has_type_map() and self.has_dihedral_type_map():
            return list(self._dihedral_type_map.values())
        else:
            return []

    @property
    def improper_types(self):
        """Get the improper types.

        Returns
        -------
        list
            List of improper types.

        """
        if self.has_type_map() and self.has_improper_type_map():
            return list(self._improper_type_map.values())
        else:
            return []
