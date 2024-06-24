import warnings

import numpy

from . import _compatibility
from .box import Box
from .topology import Angles, Bonds, Dihedrals, Impropers


class Snapshot:
    """Particle configuration.

    Parameters
    ----------
    N : int
        Number of particles in configuration.
    box : :class:`Box`
        Simulation box.
    step : int
        Simulation time step counter. Default of ``None`` means
        time step is not specified.

    Attributes
    ----------
    step : int
        Simulation time step counter.

    """

    def __init__(self, N, box, step=None, num_types=None):
        self._N = N
        self._box = Box.cast(box)
        self.step = step

        self._id = None
        self._position = None
        self._velocity = None
        self._image = None
        self._molecule = None
        self._num_types = None
        self._typeid = None
        self._charge = None
        self._mass = None
        self._bonds = None
        self._angles = None
        self._dihedrals = None
        self._impropers = None

    @classmethod
    def from_hoomd_gsd(cls, frame):
        """Create from a HOOMD GSD frame.

        Parameters
        ----------
        frame : :class:`gsd.hoomd.Frame` or `gsd.hoomd.Snapshot`
            HOOMD GSD frame to convert.

        Returns
        -------
        :class:`Snapshot`
            Snapshot created from HOOMD GSD frame.
        dict
            A map from the :attr:`Snapshot.typeid` to the HOOMD type.

        """
        # ensures frame is well formed and that we have NumPy arrays
        frame.validate()

        # process HOOMD box to LAMMPS box
        L = frame.configuration.box[:3]
        tilt = frame.configuration.box[3:]
        if frame.configuration.dimensions == 3:
            tilt[0] *= L[1]
            tilt[1:] *= L[2]
        elif frame.configuration.dimensions == 2:
            tilt[0] *= L[1]
            # HOOMD boxes can have Lz = 0, but LAMMPS does not allow this.
            if L[2] == 0:
                L[2] = 1.0
        box = Box(low=-0.5 * L, high=0.5 * L, tilt=tilt)

        snap = Snapshot(
            N=frame.particles.N,
            box=box,
            step=frame.configuration.step,
        )
        snap.position = frame.particles.position
        snap.velocity = frame.particles.velocity
        snap.image = frame.particles.image
        snap.typeid = frame.particles.typeid + 1
        snap.charge = frame.particles.charge
        snap.mass = frame.particles.mass

        snap.molecule = frame.particles.body + 1
        if numpy.any(snap.molecule < 0):
            warnings.warn("Some molecule IDs are negative, remapping needed.")

        if (
            frame.bonds.N > 0
            or frame.angles.N > 0
            or frame.dihedrals.N > 0
            or frame.impropers.N > 0
        ):
            warnings.warn("Conversion of topology from gsd is not supported")
        type_map = {typeid + 1: i for typeid, i in enumerate(frame.particles.types)}

        return snap, type_map

    def to_hoomd_gsd(self, type_map=None):
        """Create a HOOMD GSD frame.

        Parameters
        ----------
        type_map : dict
            A map from the :attr:`Snapshot.typeid` to a HOOMD type.
            If not specified, the typeids are used as the types.

        Returns
        -------
        :class:`gsd.hoomd.Frame` or :class:`gsd.hoomd.Snapshot`
            Converted HOOMD GSD frame.

        """
        if _compatibility.gsd_version is None:
            raise ImportError("GSD package not found")

        frame = _compatibility.gsd_frame_class()

        if self.step is not None:
            frame.configuration.step = int(self.step)

        # we could shift the box later, but for now this is an error
        if not numpy.allclose(-self.box.low, self.box.high):
            raise ValueError("GSD boxes must be centered around 0")
        L = self.box.high - self.box.low
        if self.box.tilt is not None:
            tilt = self.box.tilt
        else:
            tilt = [0, 0, 0]
        frame.configuration.box = numpy.concatenate((L, tilt))

        # sort tags if specified and not increasing because GSD guarantees an order
        reverse_order = None
        if self.has_id() and self.N > 1 and not numpy.all(self.id[1:] > self.id[:-1]):
            order = numpy.argsort(self.id)
            self.reorder(order, check_order=False)
            # build the reverse map to undo the sort later
            reverse_order = numpy.zeros(self.N, dtype=int)
            for i, v in enumerate(order):
                reverse_order[v] = i

        frame.particles.N = self.N
        if self.has_position():
            frame.particles.position = self.position.copy()
        if self.has_velocity():
            frame.particles.velocity = self.velocity.copy()
        if self.has_image():
            frame.particles.image = self.image.copy()
        if self.has_typeid():
            frame.particles.typeid = numpy.zeros(self.N, dtype=int)
            if type_map is None:
                sorted_typeids = numpy.sort(numpy.unique(self.typeid))
                frame.particles.types = [str(typeid) for typeid in sorted_typeids]
                for typeidx, typeid in enumerate(sorted_typeids):
                    frame.particles.typeid[self.typeid == typeid] = typeidx
            else:
                frame.particles.types = list(type_map.values())
                reverse_type_map = {
                    typeid: typeidx for typeidx, typeid in enumerate(type_map.keys())
                }
                for i, typeid in enumerate(self.typeid):
                    frame.particles.typeid[i] = reverse_type_map[typeid]
        if self.has_charge():
            frame.particles.charge = self.charge.copy()
        if self.has_mass():
            frame.particles.mass = self.mass.copy()
        if self.has_molecule():
            frame.particles.body = self.molecule - 1

        # undo the sort so object goes back the way it was
        if reverse_order is not None:
            self.reorder(reverse_order, check_order=False)

        if (
            self.has_bonds()
            or self.has_angles()
            or self.has_dihedrals()
            or self.has_impropers()
        ):
            warnings.warn("Conversion of topology to gsd is not supported")

        return frame

    @property
    def N(self):
        """int: Number of particles."""
        return self._N

    @property
    def box(self):
        """:class:`Box`: Simulation box."""
        return self._box

    @property
    def id(self):
        """:class:`numpy.ndarray`: Particle IDs."""
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
        """Check if configuration has particle IDs.

        Returns
        -------
        bool
            True if particle IDs have been initialized.

        """
        return self._id is not None

    @property
    def position(self):
        """:class:`numpy.ndarray`: Positions."""
        if not self.has_position():
            self._position = numpy.zeros((self.N, 3), dtype=float)
        return self._position

    @position.setter
    def position(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=2, copy=_compatibility.numpy_copy_if_needed, dtype=float
            )
            if v.shape != (self.N, 3):
                raise TypeError("Positions must be an Nx3 array")
            if not self.has_position():
                self._position = numpy.zeros((self.N, 3), dtype=float)
            numpy.copyto(self._position, v)
        else:
            self._position = None

    def has_position(self):
        """Check if configuration has positions.

        Returns
        -------
        bool
            True if positions have been initialized.

        """
        return self._position is not None

    @property
    def image(self):
        """:class:`numpy.ndarray`: Images."""
        if not self.has_image():
            self._image = numpy.zeros((self.N, 3), dtype=int)
        return self._image

    @image.setter
    def image(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=2, copy=_compatibility.numpy_copy_if_needed, dtype=int
            )
            if v.shape != (self.N, 3):
                raise TypeError("Images must be an Nx3 array")
            if not self.has_image():
                self._image = numpy.zeros((self.N, 3), dtype=int)
            numpy.copyto(self._image, v)
        else:
            self._image = None

    def has_image(self):
        """Check if configuration has images.

        Returns
        -------
        bool
            True if images have been initialized.

        """
        return self._image is not None

    @property
    def velocity(self):
        """:class:`numpy.ndarray`: Velocities."""
        if not self.has_velocity():
            self._velocity = numpy.zeros((self.N, 3), dtype=float)
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=2, copy=_compatibility.numpy_copy_if_needed, dtype=float
            )
            if v.shape != (self.N, 3):
                raise TypeError("Velocities must be an Nx3 array")
            if not self.has_velocity():
                self._velocity = numpy.zeros((self.N, 3), dtype=float)
            numpy.copyto(self._velocity, v)
        else:
            self._velocity = None

    def has_velocity(self):
        """Check if configuration has velocities.

        Returns
        -------
        bool
            True if velocities have been initialized.

        """
        return self._velocity is not None

    @property
    def molecule(self):
        """:class:`numpy.ndarray`: Molecule tags."""
        if not self.has_molecule():
            self._molecule = numpy.zeros(self.N, dtype=int)
        return self._molecule

    @molecule.setter
    def molecule(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=int
            )
            if v.shape != (self.N,):
                raise TypeError("Molecules must be a size N array")
            if not self.has_molecule():
                self._molecule = numpy.zeros(self.N, dtype=int)
            numpy.copyto(self._molecule, v)
        else:
            self._molecule = None

    def has_molecule(self):
        """Check if configuration has molecule tags.

        Returns
        -------
        bool
            True if molecule tags have been initialized.

        """
        return self._molecule is not None

    @property
    def num_types(self):
        """int: Number of atom types."""
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

    @property
    def typeid(self):
        """:class:`numpy.ndarray`: Types."""
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
                raise TypeError("Type must be a size N array")
            if not self.has_typeid():
                self._typeid = numpy.ones(self.N, dtype=int)
            numpy.copyto(self._typeid, v)
        else:
            self._typeid = None

    def has_typeid(self):
        """Check if configuration has types.

        Returns
        -------
        bool
            True if types have been initialized.

        """
        return self._typeid is not None

    @property
    def charge(self):
        """:class:`numpy.ndarray`: Charges."""
        if not self.has_charge():
            self._charge = numpy.zeros(self.N, dtype=float)
        return self._charge

    @charge.setter
    def charge(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=float
            )
            if v.shape != (self.N,):
                raise TypeError("Charge must be a size N array")
            if not self.has_charge():
                self._charge = numpy.zeros(self.N, dtype=float)
            numpy.copyto(self._charge, v)
        else:
            self._charge = None

    def has_charge(self):
        """Check if configuration has charges.

        Returns
        -------
        bool
            True if charges have been initialized.

        """
        return self._charge is not None

    @property
    def mass(self):
        """:class:`numpy.ndarray`: Masses."""
        if not self.has_mass():
            self._mass = numpy.ones(self.N, dtype=float)
        return self._mass

    @mass.setter
    def mass(self, value):
        if value is not None:
            v = numpy.array(
                value, ndmin=1, copy=_compatibility.numpy_copy_if_needed, dtype=float
            )
            if v.shape != (self.N,):
                raise TypeError("Mass must be a size N array")
            if not self.has_mass():
                self._mass = numpy.ones(self.N, dtype=float)
            numpy.copyto(self._mass, v)
        else:
            self._mass = None

    def has_mass(self):
        """Check if configuration has masses.

        Returns
        -------
        bool
            True if masses have been initialized.

        """
        return self._mass is not None

    @property
    def bonds(self):
        """Bonds: Bond data."""
        return self._bonds

    @bonds.setter
    def bonds(self, value):
        if value is not None:
            if not isinstance(value, Bonds):
                raise TypeError("bonds must be Bonds")
            self._bonds = value
        else:
            self._bonds = None

    def has_bonds(self):
        """Check if configuration has bonds.

        Returns
        -------
        bool
            True if bonds is initialized and there is at least one bond.

        """
        return self._bonds is not None and self._bonds.N > 0

    @property
    def angles(self):
        """Angles: Angle data."""
        return self._angles

    @angles.setter
    def angles(self, value):
        if value is not None:
            if not isinstance(value, Angles):
                raise TypeError("angles must be Angles")
            self._angles = value
        else:
            self._angles = None

    def has_angles(self):
        """Check if configuration has angles.

        Returns
        -------
        bool
            True if angles is initialized and there is at least one angle.

        """
        return self._angles is not None and self._angles.N > 0

    @property
    def dihedrals(self):
        """Dihedrals: Dihedral data."""
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, value):
        if value is not None:
            if not isinstance(value, Dihedrals):
                raise TypeError("dihedrals must be Dihedrals")
            self._dihedrals = value
        else:
            self._dihedrals = None

    def has_dihedrals(self):
        """Check if configuration has dihedrals.

        Returns
        -------
        bool
            True if dihedrals is initialized and there is at least one dihedral.

        """
        return self._dihedrals is not None and self._dihedrals.N > 0

    @property
    def impropers(self):
        """Impropers: Improper data."""
        return self._impropers

    @impropers.setter
    def impropers(self, value):
        if value is not None:
            if not isinstance(value, Impropers):
                raise TypeError("impropers must be Impropers")
            self._impropers = value
        else:
            self._impropers = None

    def has_impropers(self):
        """Check if configuration has impropers.

        Returns
        -------
        bool
            True if impropers is initialized and there is at least one improper.

        """
        return self._impropers is not None and self._impropers.N > 0

    def reorder(self, order, check_order=True):
        """Reorder the particles in place.

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
        if self.has_position():
            self._position = self._position[order]
        if self.has_velocity():
            self._velocity = self._velocity[order]
        if self.has_image():
            self._image = self._image[order]
        if self.has_molecule():
            self._molecule = self._molecule[order]
        if self.has_typeid():
            self._typeid = self._typeid[order]
        if self.has_charge():
            self._charge = self._charge[order]
        if self.has_mass():
            self._mass = self._mass[order]
