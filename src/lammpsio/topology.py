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
        particle=None,
        bond=None,
        angle=None,
        dihedral=None,
        improper=None,
    ):
        self._particle = particle
        self._bond = bond
        self._angle = angle
        self._dihedral = dihedral
        self._improper = improper

    @property
    def particle(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD type."""
        return self._particle

    @particle.setter
    def particle(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("type_map must be a dictionary")
            self._particle = {value}
        else:
            self._particle = None

    @property
    def bond(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD bond type."""
        return self._bond

    @bond.setter
    def bond(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("bond_type_map must be a dictionary")
            self._bond = {value}
        else:
            self._bond = None

    @property
    def angle(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD angle type."""
        return self._angle

    @angle.setter
    def angle(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("angle_type_map must be a dictionary")
            self._angle = {value}
        else:
            self._angle = None

    @property
    def dihedral(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD dihedral type."""
        return self._dihedral

    @dihedral.setter
    def dihedral(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("dihedral_type_map must be a dictionary")
            self._dihedral = {value}
        else:
            self._dihedral = None

    @property
    def improper(self):
        """dict: A map from the :attr:`Snapshot.typeid` to a HOOMD improper type."""
        return self._improper

    @improper.setter
    def improper(self, value):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("improper_type_map must be a dictionary")
            self._improper = {value}
        else:
            self._improper = None

    @property
    def particle_types(self):
        """Get the HOOMD types.

        Returns
        -------
        list
            List of HOOMD types.

        """
        return tuple(self._particle.values())

    @property
    def bond_types(self):
        """Get the bond types.

        Returns
        -------
        list
            List of bond types.

        """
        if self.bond is not None:
            return tuple(self._bond.values())
        else:
            return tuple()

    @property
    def angle_types(self):
        """Get the angle types.

        Returns
        -------
        list
            List of angle types.

        """
        if self.angle is not None:
            return tuple(self._angle.values())
        else:
            return tuple()

    @property
    def dihedral_types(self):
        """Get the dihedral types.

        Returns
        -------
        list
            List of dihedral types.

        """
        if self.dihedral is not None:
            return tuple(self._dihedral.values())
        else:
            return tuple()

    @property
    def improper_types(self):
        """Get the improper types.

        Returns
        -------
        list
            List of improper types.

        """
        if self.improper is not None:
            return tuple(self._improper.values())
        else:
            return tuple()

    @property
    def reverse_particle(self):
        """Reverse the particle type map."""
        if self._particle is not None:
            return {types: typeid for typeid, types in self._particle.items()}
        else:
            self._particle = None

    @property
    def reverse_bond(self):
        """Reverse the bond type map."""
        if self._bond is not None:
            return {types: typeid for typeid, types in self._bond.items()}
        else:
            self._bond = None

    @property
    def reverse_angle(self):
        """Reverse the angle type map."""
        if self._angle is not None:
            return {types: typeid for typeid, types in self._angle.items()}
        else:
            self._angle = None

    @property
    def reverse_dihedral(self):
        """Reverse the dihedral type map."""
        if self._dihedral is not None:
            return {types: typeid for typeid, types in self._dihedral.items()}
        else:
            self._dihedral = None

    @property
    def reverse_improper(self):
        """Reverse the improper type map."""
        if self._improper is not None:
            return {types: typeid for typeid, types in self._improper.items()}
        else:
            self._improper = None
