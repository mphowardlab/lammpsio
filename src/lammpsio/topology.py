"""Topology (connection and type) information."""

import collections.abc
from typing import Any, Optional

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
        Number of connection types. Default of ``None`` means the number of
        types is determined from the unique typeids.

    All values of indexes follow the LAMMPS 1-indexed convention, but the
    arrays themselves are 0-indexed. Lazy array initialization is used as for
    the `Snapshot`.

    """

    def __init__(
        self, N: int, num_members: int, num_types: Optional[int] = None
    ) -> None:
        self._N = N
        self._num_members = num_members
        self.num_types = num_types

        self._type_label = None
        self._id = None
        self._typeid = None
        self._members = None

    @property
    def N(self) -> int:
        """int: Number of connections."""
        return self._N

    @property
    def id(self) -> numpy.ndarray:
        """(*N*,) `numpy.ndarray` of `int`: Unique identifiers (IDs).

        The default value on initialization runs from 1 to `N`.

        """
        if not self.has_id():
            self._id = numpy.arange(1, self.N + 1)
        return self._id

    @id.setter
    def id(self, value: Any) -> None:
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

    def has_id(self) -> bool:
        """Check if configuration has connection IDs.

        Returns
        -------
        bool
            ``True`` if connection IDs have been initialized.

        """
        return self._id is not None

    @property
    def typeid(self) -> numpy.ndarray:
        """(*N*,) `numpy.ndarray` of `int`: Connection type IDs.

        The default value on initialization is 1 for all entries.

        """
        if not self.has_typeid():
            self._typeid = numpy.ones(self.N, dtype=int)
        return self._typeid

    @typeid.setter
    def typeid(self, value: Any) -> None:
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

    def has_typeid(self) -> bool:
        """Check if configuration has connection typeids.

        Returns
        -------
        bool
            ``True`` if connection typeids have been initialized.

        """
        return self._typeid is not None

    @property
    def members(self) -> numpy.ndarray:
        """(*N*, *M*) `numpy.ndarray` of `int`: Connection members.

        The default value on initialization is 1 for all entries. *M* is the
        number of members in the connection.

        """
        if not self.has_members():
            self._members = numpy.ones((self.N, self._num_members), dtype=int)
        return self._members

    @members.setter
    def members(self, value: Any) -> None:
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

    def has_members(self) -> bool:
        """Check if configuration has connection members.

        Returns
        -------
        bool
            ``True`` if particle members have been initialized.

        """
        return self._members is not None

    @property
    def num_types(self) -> int:
        """int: Number of connection types."""
        if self._num_types is not None:
            return self._num_types
        else:
            if self.has_typeid():
                return numpy.amax(self.typeid)
            else:
                return 1

    @num_types.setter
    def num_types(self, value: Any) -> None:
        if value is not None:
            self._num_types = int(value)
        else:
            self._num_types = None

    @property
    def type_label(self) -> Optional["LabelMap"]:
        """LabelMap: Labels of connection type IDs."""

        return self._type_label

    @type_label.setter
    def type_label(self, value: Optional["LabelMap"]) -> None:
        if value is not None:
            if not isinstance(value, LabelMap):
                raise TypeError("type_label must be a LabelMap")
            self._type_label = value
        else:
            self._type_label = None

    def reorder(self, order, check_order=True):
        """Reorder the connections in place.

        Parameters
        ----------
        order : list
            New order of indexes.
        check_order : bool
            If ``True``, validate the new ``order`` before applying it.

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
    """Bond connections between particles.

    All values of indexes follow the LAMMPS 1-indexed convention, but the
    arrays themselves are 0-indexed.

    Parameters
    ----------
    N : int
        Number of bonds.
    num_types : int
        Number of bond types. Default of ``None`` means the number of types is
        determined from the unique typeids.

    Example
    -------
    Create bonds:

    .. code-block:: python

        bonds = lammpsio.topology.Bonds(N=3, num_types=1)
        bonds.id = [1, 2, 3]
        bonds.typeid = [1, 1, 1]
        bonds.members = [[1, 2], [2, 3], [1, 3]]

    This creates a molecule with three bonds of the same type.
    The bonds are defined between particle IDs 1-2, 2-3, and 1-3 to form a triangle
    structure.

    """

    def __init__(self, N: int, num_types: Optional[int] = None) -> None:
        super().__init__(N=N, num_members=2, num_types=num_types)


class Angles(Topology):
    """Angle connections between particles.

    All values of indexes follow the LAMMPS 1-indexed convention, but the
    arrays themselves are 0-indexed.

    Parameters
    ----------
    N : int
        Number of angles.
    num_types : int
        Number of angle types. Default of ``None`` means the number of types is
        determined from the unique typeids.

    Example
    -------
    Create angles:

    .. code-block:: python

        angles = lammpsio.topology.Angles(N=3, num_types=1)
        angles.id = [1, 2, 3]
        angles.typeid = [1, 1, 1]
        angles.members = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]

    This creates a molecule with three angles of the same type.
    The angles are defined between particles IDs 1-2-3, 2-3-1, and 3-1-2
    to form a triangle structure.

    """

    def __init__(self, N: int, num_types: Optional[int] = None) -> None:
        super().__init__(N=N, num_members=3, num_types=num_types)


class Dihedrals(Topology):
    """Dihedral connections between particles.

    All values of indexes follow the LAMMPS 1-indexed convention, but the
    arrays themselves are 0-indexed.

    Parameters
    ----------
    N : int
        Number of dihedrals.
    num_types : int
        Number of dihedral types. Default of ``None`` means the number of types
        is determined from the unique typeids.

    Example
    -------
    Create dihedrals:

    .. code-block:: python

        dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)
        dihedrals.id = [1, 2]
        dihedrals.typeid = [1, 2]
        dihedrals.members = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ]

    This creates two dihedrals of two different types for a molecule
    consisting of eight atoms.

    """

    def __init__(self, N: int, num_types: Optional[int] = None) -> None:
        super().__init__(N=N, num_members=4, num_types=num_types)


class Impropers(Topology):
    """Improper dihedral connections between particles.

    All values of indexes follow the LAMMPS 1-indexed convention, but the
    arrays themselves are 0-indexed.

    Parameters
    ----------
    N : int
        Number of improper dihedrals.
    num_types : int
        Number of improper dihedral types. Default of ``None`` means the number
        of types is determined from the unique typeids.

    Example
    -------
    Create improper dihedrals:

    .. code-block:: python

        impropers = lammpsio.topology.Impropers(N=2, num_types=2)
        impropers.id = [1, 2]
        impropers.typeid = [1, 2]
        impropers.members = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ]

    This creates two improper dihedrals of two different types for a molecule
    consisting of five atoms.

    """

    def __init__(self, N: int, num_types: Optional[int] = None) -> None:
        super().__init__(N=N, num_members=4, num_types=num_types)


class LabelMap(collections.abc.MutableMapping[int, str]):
    """Map between integer type IDs and string type names.

    A `LabelMap` is effectively a dictionary associating a label (type) with a
    particle's or connection's typeid. These labels can be useful for tracking
    the meaning of typeids. They are also automatically used when interconverting
    with HOOMD GSD files that require such labels.

    Parameters
    ----------
    map : dict
        Map of typeids to types.

    Example
    -------
    Create `LabelMap`:

    .. code-block:: python

        type_label = lammpsio.topology.LabelMap({1: "A", 2: "B"})

    This creates a dictionary mapping numeric type ID labels 1 and 2 used by LAMMPS
    to alphanumeric type labels "A" and "B", such as those used by HOOMD-blue.

    to alphanumeric type labels "A" and "B", such as those used by HOOMD-blue. The
    map can be accessed directly:
    
    .. code-block:: python
    
        assert type_label[1] == "A"
    
    The `LabelMap` additionally supports inverse mapping from type label to type ID:
    
    .. code-block:: python
    
        assert type_label.inverse["B"] == 2
    """

    def __init__(self, map=None):
        self._map = {}
        self._inverse_map = {}
        if map is not None:
            self.update(map)

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        self._map[key] = value
        self._inverse_map[value] = key

    def __delitem__(self, key):
        value = self._map.pop(key)
        del self._inverse_map[value]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    @property
    def types(self):
        """tuple of str: Types in map."""
        return tuple(self._map.values())

    @property
    def typeid(self):
        """tuple of int: Type IDs in map."""
        return tuple(self._map.keys())

    @property
    def inverse(self) -> dict[str, int]:
        """dict: Inverse map from type label to type id."""
        return self._inverse_map
