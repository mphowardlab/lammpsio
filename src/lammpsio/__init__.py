import gzip
import os
import pathlib

import numpy

def _readline(file_,require=False):
    """Read and require a line."""
    line = file_.readline()
    if require and len(line) == 0:
        raise OSError('Could not read line from file')
    return line

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
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape == (9,):
            return Box(v[:3], v[3:6], v[6:])
        elif v.shape == (6,):
            return Box(v[:3], v[3:])
        else:
            raise TypeError('Unable to cast boxlike object with shape {}'.format(v.shape))

    @property
    def low(self):
        """:class:`numpy.ndarray`: Box low."""
        return self._low

    @low.setter
    def low(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=numpy.float64)
        if v.shape != (3,):
            raise TypeError('Low must be a 3-tuple')
        self._low = v

    @property
    def high(self):
        """:class:`numpy.ndarray`: Box high."""
        return self._high

    @high.setter
    def high(self, value):
        v = numpy.array(value, ndmin=1, copy=True, dtype=numpy.float64)
        if v.shape != (3,):
            raise TypeError('High must be a 3-tuple')
        self._high = v

    @property
    def tilt(self):
        """:class:`numpy.ndarray`: Box tilt factors."""
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        v = value
        if v is not None:
            v = numpy.array(v, ndmin=1, copy=True, dtype=numpy.float64)
            if v.shape != (3,):
                raise TypeError('Tilt must be a 3-tuple')
        self._tilt = v

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
    def __init__(self, N, box, step=None):
        self._N = N
        self._box = Box.cast(box)
        self.step = step

        self._id = None
        self._position = None
        self._velocity = None
        self._image = None
        self._molecule = None
        self._typeid = None
        self._charge = None
        self._mass = None

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
            self._id = numpy.arange(1, self.N+1)
        return self._id

    @id.setter
    def id(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.int32)
            if v.shape != (self.N,):
                raise TypeError('Ids must be a size N array')
            if not self.has_id():
                self._id = numpy.arange(1, self.N+1)
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
            self._position = numpy.zeros((self.N, 3), dtype=numpy.float64)
        return self._position

    @position.setter
    def position(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.float64)
            if v.shape != (self.N, 3):
                raise TypeError('Positions must be an Nx3 array')
            if not self.has_position():
                self._position = numpy.zeros((self.N, 3), dtype=numpy.float64)
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
            self._image = numpy.zeros((self.N, 3), dtype=numpy.int32)
        return self._image

    @image.setter
    def image(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.int32)
            if v.shape != (self.N, 3):
                raise TypeError('Images must be an Nx3 array')
            if not self.has_image():
                self._image = numpy.zeros((self.N, 3), dtype=numpy.int32)
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
            self._velocity = numpy.zeros((self.N, 3), dtype=numpy.float64)
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.float64)
            if v.shape != (self.N, 3):
                raise TypeError('Velocities must be an Nx3 array')
            if not self.has_velocity():
                self._velocity = numpy.zeros((self.N, 3), dtype=numpy.float64)
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
            self._molecule = numpy.zeros(self.N, dtype=numpy.int32)
        return self._molecule

    @molecule.setter
    def molecule(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.int32)
            if v.shape != (self.N,):
                raise TypeError('Molecules must be a size N array')
            if not self.has_molecule():
                self._molecule = numpy.zeros(self.N, dtype=numpy.int32)
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
    def typeid(self):
        """:class:`numpy.ndarray`: Types."""
        if not self.has_typeid():
            self._typeid = numpy.ones(self.N, dtype=numpy.int32)
        return self._typeid

    @typeid.setter
    def typeid(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.int32)
            if v.shape != (self.N,):
                raise TypeError('Type must be a size N array')
            if not self.has_typeid():
                self._typeid = numpy.ones(self.N, dtype=numpy.int32)
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
            self._charge = numpy.zeros(self.N, dtype=numpy.float64)
        return self._charge

    @charge.setter
    def charge(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
            if v.shape != (self.N,):
                raise TypeError('Charge must be a size N array')
            if not self.has_charge():
                self._charge = numpy.zeros(self.N, dtype=numpy.float64)
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
            self._mass = numpy.ones(self.N, dtype=numpy.float64)
        return self._mass

    @mass.setter
    def mass(self, value):
        if value is not None:
            v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
            if v.shape != (self.N,):
                raise TypeError('Mass must be a size N array')
            if not self.has_mass():
                self._mass = numpy.ones(self.N, dtype=numpy.float64)
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
            if (sorted_order[0] != 0 or sorted_order[-1] != self.N-1
                or not numpy.all(sorted_order[1:] == sorted_order[:-1]+1)):
                raise ValueError('New order must be an array from 0 to N-1')

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

class DataFile:
    """LAMMPS data file.

    Parameters
    ----------
    filename : str
        Path to data file.
    atom_style : str
        Atom style to use for data file. Defaults to ``None``, which means the
        style should be read from the file.

    Attributes
    ----------
    filename : str
        Path to the file.
    atom_style : str
        Atom style for the file.
    known_headers : list
        Data file headers that can be processed.
    unknown_headers : list
        Data file headers that will be ignored.
    known_bodies : list
        Data file body sections that can be processed.
    unknown_bodies : list
        Data file body sections that will be ignored.

    """
    def __init__(self, filename, atom_style=None):
        self.filename = filename
        self.atom_style = atom_style

    known_headers = ('atoms','atom types','xlo xhi','ylo yhi','zlo zhi','xy xz yz')
    unknown_headers = (
        'bonds','angles','dihedrals','impropers',
        'bond types','angle types','dihedral types','improper types',
        'extra bond per atom','extra angle per atom','extra dihedral per atom',
        'extra improper per atom',
        'ellipsoids','lines','triangles','bodies'
        )
    known_bodies = ('Atoms','Velocities','Masses')
    unknown_bodies = (
        'Ellipsoids','Lines','Triangles','Bodies',
        'Bonds','Angles','Dihedrals','Impropers'
        )

    @classmethod
    def create(cls, filename, snapshot, atom_style=None):
        """Create a LAMMPS data file from a snapshot.

        Parameters
        ----------
        filename : str
            Path to data file.
        snapshot : :class:`Snapshot`
            Snapshot to write to file.
        atom_style : str
            Atom style to use for data file. Defaults to ``None``, which means the
            style should be inferred from the contents of ``snapshot``.

        Returns
        -------
        :class:`DataFile`
            The object representing the new data file.

        Raises
        ------
        ValueError
            If all masses are not the same for a given type.

        """
        # extract number of types
        num_types = numpy.amax(numpy.unique(snapshot.typeid))

        # extract mass by type
        if snapshot.has_mass():
            masses = numpy.empty(num_types)
            for i in range(num_types):
                mi = snapshot.mass[snapshot.typeid == i+1]
                if not numpy.all(mi == mi[0]):
                    raise ValueError('All masses for a type must be equal')
                elif mi[0] <= 0.:
                    raise ValueError('Type mass must be positive value')
                masses[i] = mi[0]
        else:
            masses = None

        with open(filename, 'w') as f:
            # LAMMPS header
            f.write("LAMMPS {}\n\n".format(filename))
            f.write("{} atoms\n".format(snapshot.N))
            f.write("{} atom types\n".format(num_types))
            f.write("{} {} xlo xhi\n".format(snapshot.box.low[0],snapshot.box.high[0]))
            f.write("{} {} ylo yhi\n".format(snapshot.box.low[1],snapshot.box.high[1]))
            f.write("{} {} zlo zhi\n".format(snapshot.box.low[2],snapshot.box.high[2]))
            if snapshot.box.tilt is not None:
                f.write("{} {} {} xy xz yz\n".format(*snapshot.box.tilt))

            # Atoms section
            # determine style if it is not given
            if atom_style is None:
                if snapshot.has_charge() and snapshot.has_molecule():
                    style = 'full'
                elif snapshot.has_charge():
                    style = 'charge'
                elif snapshot.has_molecule():
                    style = 'molecular'
                else:
                    style = 'atomic'
            else:
                style = atom_style
            # set format string based on style
            if style == 'full':
                style_fmt = '{atomid:d} {molid:d} {typeid:d} {q:.5f} {x:.8f} {y:.8f} {z:.8f}'
            elif style == 'charge':
                style_fmt = '{atomid:d} {typeid:d} {q:.5f} {x:.8f} {y:.8f} {z:.8f}'
            elif style == 'molecular':
                style_fmt = '{atomid:d} {molid:d} {typeid:d} {x:.8f} {y:.8f} {z:.8f}'
            elif style == 'atomic':
                style_fmt = '{atomid:d} {typeid:d} {x:.8f} {y:.8f} {z:.8f}'
            else:
                raise ValueError('Unknown atom style')
            if snapshot.has_image():
                style_fmt += ' {ix:d} {iy:d} {iz:d}'
            # write section
            f.write("\nAtoms # {}\n\n".format(style))
            for i in range(snapshot.N):
                style_args = dict(
                    atomid=snapshot.id[i] if snapshot.has_id() else i+1,
                    typeid=snapshot.typeid[i],
                    x=snapshot.position[i][0],
                    y=snapshot.position[i][1],
                    z=snapshot.position[i][2],
                    q=snapshot.charge[i] if snapshot.has_charge() else 0.,
                    molid=snapshot.molecule[i] if snapshot.has_molecule() else 0
                    )
                if snapshot.has_image():
                    style_args.update(ix=snapshot.image[i][0],
                                      iy=snapshot.image[i][1],
                                      iz=snapshot.image[i][2])

                f.write(style_fmt.format(**style_args) + '\n')

            # Velocities section
            if snapshot.has_velocity():
                f.write("\nVelocities\n\n")
                for i in range(snapshot.N):
                    vel_fmt="{atomid:8d}{vx:16.8f}{vy:16.8f}{vz:16.8f}\n"
                    f.write(vel_fmt.format(
                        atomid=snapshot.id[i] if snapshot.has_id() else i+1,
                        vx=snapshot.velocity[i][0],
                        vy=snapshot.velocity[i][1],
                        vz=snapshot.velocity[i][2]))

            # Masses section
            if masses is not None:
                f.write("\nMasses\n\n")
                for i,mi in enumerate(masses):
                    f.write("{typeid:4d}{m:12}\n".format(typeid=i+1, m=mi))

        return DataFile(filename)

    def read(self):
        """Read the file.

        Returns
        -------
        :class:`Snapshot`
            Snapshot from the data file.

        Raises
        ------
        ValueError
            If :attr:`atom_style` is set but does not match file contents.
        ValueError
            If :attr:`atom_style` is not specified and not set in file.

        """
        with open(self.filename) as f:
            # initialize snapshot from header
            N = None
            box_bounds = [None,None,None,None,None,None]
            box_tilt = None
            num_types = None
            # skip first line
            _readline(f,True)
            line = _readline(f)
            while len(line) > 0:
                line = line.rstrip()

                # skip blank and go to next line
                if len(line) == 0:
                    line = _readline(f)
                    continue

                # check for unknown headers and go to next line
                skip_line = False
                for h in self.unknown_headers:
                    if h in line:
                        skip_line = True
                        break
                if skip_line:
                    line = _readline(f)
                    continue

                # line is not empty but it is not a header, so break and try to make snapshot
                # keep the line so that it can be processed in next step
                done_header = True
                for h in self.known_headers:
                    if h in line:
                        done_header = False
                        break
                if done_header:
                    break

                # process useful header info
                if 'atoms' in line:
                    N = int(line.split()[0])
                elif 'atom types' in line:
                    num_types = int(line.split()[0])
                elif 'xlo xhi' in line:
                    box_bounds[0],box_bounds[3] = [float(x) for x in line.split()[:2]]
                elif 'ylo yhi' in line:
                    box_bounds[1],box_bounds[4] = [float(x) for x in line.split()[:2]]
                elif 'zlo zhi' in line:
                    box_bounds[2],box_bounds[5] = [float(x) for x in line.split()[:2]]
                elif 'xy xz yz' in line:
                    box_tilt = [float(x) for x in line.split()[:3]]
                else:
                    raise RuntimeError('Uncaught header line! Check programming')

                # done here, read next line
                line = _readline(f)

            if N is None:
                raise IOError('Number of particles not read')
            elif num_types is None:
                raise IOError('Number of types not read')
            elif None in box_bounds:
                raise IOError('Box bounds not read')
            box = Box(box_bounds[:3], box_bounds[3:], box_tilt)
            snap = Snapshot(N,box)
            id_map = {}

            # now that snapshot is made, file it in with body sections
            masses = None
            while len(line) > 0:
                if 'Atoms' in line:
                    # use or extract style
                    row = line.split()
                    if len(row) == 3:
                        style = row[-1]
                        if self.atom_style is not None and style != self.atom_style:
                            raise ValueError('Specified style does not match style in file')
                    else:
                        style = self.atom_style
                    if style is None:
                        raise IOError('Atom style not found, specify.')
                    # number of columns to read for style
                    if style == 'full':
                        style_cols = 3
                    elif style == 'charge':
                        style_cols = 2
                    elif style == 'molecular':
                        style_cols = 2
                    elif style == 'atomic':
                        style_cols = 1
                    else:
                        raise ValueError('Unknown atom style')

                    # read atom coordinates
                    _readline(f,True) # blank line
                    for i in range(snap.N):
                        # strip out comments
                        row = _readline(f,True).split()
                        try:
                            comment = row.index('#')
                            row = row[:comment]
                        except ValueError:
                            pass

                        # check that row is correctly sized and process
                        if not (len(row) == style_cols+4 or len(row) == style_cols+7):
                            raise IOError('Expected number of columns not read for atom style')

                        # only save the atom id if it is not in standard order
                        id_ = int(row[0])
                        if id_ != i+1:
                            if id_ not in id_map:
                                id_map[id_] = i
                            idx = id_map[id_]
                            snap.id[idx] = id_
                        else:
                            idx = i

                        if style == 'full':
                            snap.molecule[idx] = int(row[1])
                            snap.typeid[idx] = int(row[2])
                            snap.charge[idx] = float(row[3])
                        elif style == 'charge':
                            snap.typeid[idx] = int(row[1])
                            snap.charge[idx] = float(row[2])
                        elif style == 'molecular':
                            snap.molecule[idx] = int(row[1])
                            snap.typeid[idx] = int(row[2])
                        elif style == 'atomic':
                            snap.typeid[idx] = int(row[1])
                        snap.position[idx] = [float(x) for x in row[style_cols+1:style_cols+4]]
                        if len(row) == style_cols+7:
                            snap.image[idx] = [int(x) for x in row[style_cols+4:style_cols+7]]

                    # sanity check types
                    if numpy.any(numpy.logical_or(snap.typeid < 1, snap.typeid > num_types)):
                        raise ValueError('Invalid type id')
                elif 'Velocities' in line:
                    _readline(f,True) # blank line
                    for i in range(snap.N):
                        row = _readline(f,True).split()
                        if len(row) < 4:
                            raise IOError('Expected number of columns not read for velocity')
                        # parse atom id: need to repeat mapping in case Velocity comes before Atoms
                        id_ = int(row[0])
                        if id_ != i+1:
                            if id_ not in id_map:
                                id_map[id_] = i
                            idx = id_map[id_]
                            snap.id[idx] = id_
                        else:
                            idx = i
                        snap.velocity[idx] = [float(x) for x in row[1:4]]
                elif 'Masses' in line:
                    masses = {}
                    _readline(f,True) # blank line
                    for i in range(num_types):
                        row = _readline(f,True).split()
                        if len(row) < 2:
                            raise IOError('Expected number of columns not read for mass')
                        masses[int(row[0])] = float(row[1])
                else:
                    # silently ignore unknown sections / lines
                    pass

                line = _readline(f)

            # set mass on particles at end, in case sections were out of order in file
            if masses is not None:
                for typeid,m in masses.items():
                    snap.mass[snap.typeid == typeid] = m

        return snap

class DumpFile:
    """LAMMPS dump file.

    The dump file is a flexible file format, so a ``schema`` can be given
    to parse the atom data. The ``schema`` is given as a dictionary of column
    indexes. Valid keys for the schema match the names and shapes in the `Snapshot`.
    The keys requiring only 1 column index are: ``id``, ``typeid``, ``molecule``,
    ``charge``, and ``mass``. The keys requiring 3 column indexes are ``position``,
    ``velocity``, and ``image``. If a ``schema`` is not specified, it will be deduced
    from the ``ITEM: ATOMS`` header.

    The vector-valued fields (``position``, ``velocity``, ``image``) must contain all
    three elements.

    Parameters
    ----------
    filename : str
        Path to dump file.
    schema : dict
        Schema for the contents of the file. Defaults to ``None``, which means to read
        it from the file.
    sort_ids : bool
        If true, sort the particles by ID in each snapshot.
    copy_from : :class:`Snapshot`
        If specified, copy fields that are missing in the dump file but are set in
        a reference :class:`Snapshot`. The fields that can be copied are ``typeid``,
        ``molecule``, ``charge``, and ``mass``.

    """
    def __init__(self, filename, schema=None, sort_ids=True, copy_from=None):
        self.filename = filename
        self.schema = schema
        self._frames = None
        self.sort_ids = sort_ids
        self.copy_from = copy_from

    @classmethod
    def create(cls, filename, schema, snapshots):
        """Create a LAMMPS dump file.

        Parameters
        ----------
        filename : str
            Path to dump file.
        schema : dict
            Schema for the contents of the file.
        snapshots : :class:`Snapshot` or list
            One or more snapshots to write to the dump file.

        Returns
        -------
        :class:`DumpFile`
            The object representing the new dump file.

        """
        # map out the schema into a dump row
        # each entry is a tuple: (column, (attribute, index))
        # the index is None for scalars, otherwise it is the component of the vector
        dump_row = []
        for k, v in schema.items():
            try:
                cols = iter(v)
                for i,col in enumerate(cols):
                    dump_row.append((col, (k, i)))
            except TypeError:
                dump_row.append((v, (k, None)))
        dump_row.sort(key=lambda x : x[0])

        # make snapshots iterable
        try:
            snapshots = iter(snapshots)
        except TypeError:
            snapshots = [snapshots]

        with open(filename, 'w') as f:
            for snap in snapshots:
                f.write('ITEM: TIMESTEP\n')
                f.write('{}\n'.format(snap.step))

                f.write('ITEM: NUMBER OF ATOMS\n')
                f.write('{}\n'.format(snap.N))

                # always assume periodic in all directions
                box_header = 'ITEM: BOX BOUNDS pp pp pp'
                if snap.box.tilt is not None:
                    xy, xz, yz = snap.box.tilt
                    box_header += ' {xy:f} {xz:f} {yz:f}'.format(xy=xy, xz=xz, yz=yz)
                    lo = [
                        snap.box.low[0] + min([0.0, xy, xz, xy+xz]),
                        snap.box.low[1] + min([0.0, yz]),
                        snap.box.low[2]
                        ]
                    hi = [
                        snap.box.high[0] + max([0.0, xy, xz, xy+xz]),
                        snap.box.high[1] + max([0.0, yz]),
                        snap.box.high[2]
                        ]
                else:
                    lo = snap.box.low
                    hi = snap.box.high
                f.write(box_header + '\n')
                for i in range(3):
                    f.write('{lo:f} {hi:f}\n'.format(lo=lo[i], hi=hi[i]))

                # mapping from lammpsio to LAMMPS dump keys
                lammps_fields = {
                    'id': 'id',
                    'molecule': 'mol',
                    'typeid': 'type',
                    'mass': 'mass',
                    'position': ('x', 'y', 'z'),
                    'image': ('ix', 'iy', 'iz'),
                    'velocity': ('vx', 'vy', 'vz'),
                    'charge': 'q'}
                schema_header = []
                for _, (key, key_idx) in dump_row:
                    field = lammps_fields[key]
                    if key_idx is not None:
                        field = field[key_idx]
                    schema_header.append(field)
                schema_header = ' '.join(schema_header)

                f.write('ITEM: ATOMS ' + schema_header + '\n')
                for i in range(snap.N):
                    line = ''
                    for col, (key, key_idx) in dump_row:
                        if key == 'id':
                            val = snap.id[i] if snap.has_id() else i+1
                        else:
                            val = getattr(snap, key)[i]
                            if key_idx is not None:
                                val = val[key_idx]

                        if key in ('id', 'typeid', 'molecule', 'image'):
                            fmt = '{:d}'
                        elif key in ('position', 'velocity'):
                            fmt = '{:.8f}'
                        else:
                            fmt = '{:f}'

                        if col > 0:
                            fmt = ' ' + fmt
                        line += fmt.format(val)
                    line = line.strip()
                    f.write(line + '\n')

        filename_path = pathlib.Path(filename)
        if filename_path.suffix == '.gz':
            tmp = pathlib.Path(filename).with_suffix(filename_path.suffix + '.tmp')
            with open(filename, 'rb') as src, gzip.open(tmp, 'wb') as dest:
                dest.writelines(src)
            os.replace(tmp, filename)

        return DumpFile(filename, schema)

    @property
    def copy_from(self):
        return self._copy_from

    @copy_from.setter
    def copy_from(self, value):
        if value is not None and not isinstance(value, Snapshot):
            raise TypeError('Dump file can only copy from Snapshot')
        self._copy_from = value

    @property
    def filename(self):
        """str: Path to the file."""
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        self._gzip = pathlib.Path(value).suffix == '.gz'

        # configure section labels of dump file
        self._section= {
            'step': 'ITEM: TIMESTEP',
            'natoms': 'ITEM: NUMBER OF ATOMS',
            'box': 'ITEM: BOX BOUNDS',
            'atoms': 'ITEM: ATOMS'
            }
        if self._gzip:
            for key,val in self._section.items():
                self._section[key] = val.encode()

    @property
    def schema(self):
        """dict: Data schema."""
        return self._schema

    @schema.setter
    def schema(self, value):
        if value is not None:
            # validate schema
            if 'position' in value and len(value['position']) != 3:
                raise ValueError('Position must be a 3-tuple')
            if 'velocity' in value and len(value['velocity']) != 3:
                raise ValueError('Velocity must be a 3-tuple')
            if 'image' in value and len(value['image']) != 3:
                raise ValueError('Image must be a 3-tuple')
        self._schema = value

    def _open(self):
        """Open the file handle for reading."""
        if self._gzip:
            f = gzip.open(self.filename, 'rb')
        else:
            f = open(self.filename, 'r')
        return f

    def _find_frames(self):
        """Seek line numbers for each frame."""
        self._frames = []
        with self._open() as f:
            line = _readline(f)
            line_num = 0
            while len(line) > 0:
                if self._section['step'] in line:
                    self._frames.append(line_num)
                line = _readline(f)
                line_num += 1

    def __len__(self):
        if self._frames is None:
            self._find_frames()
        return len(self._frames)

    def __iter__(self):
        with self._open() as f:
            state = 0
            line = _readline(f)
            while len(line) > 0:
                # timestep line first
                if state == 0 and self._section['step'] in line:
                    state += 1
                    step = int(_readline(f,True))

                # number of particles second
                if state == 1 and self._section['natoms'] in line:
                    state += 1
                    N = int(_readline(f,True))

                # box size third
                if state == 2 and self._section['box'] in line:
                    state += 1
                    box_header = line.split()
                    # check for triclinic
                    if len(box_header) == 9:
                        box_tilt = [float(x) for x in box_header[6:9]]
                    elif len(box_header) == 6:
                        box_tilt = None
                    else:
                        raise IOError('Incorrectly formed box bound header')
                    box_x = _readline(f,True)
                    box_y = _readline(f,True)
                    box_z = _readline(f,True)
                    x_lo, x_hi = [float(x) for x in box_x.split()]
                    y_lo, y_hi = [float(y) for y in box_y.split()]
                    z_lo, z_hi = [float(z) for z in box_z.split()]
                    if box_tilt is not None:
                        xy, xz, yz = box_tilt
                        lo = [
                            x_lo - min([0.0, xy, xz, xy+xz]),
                            y_lo - min([0.0, yz]),
                            z_lo
                            ]
                        hi = [
                            x_hi - max([0.0, xy, xz, xy+xz]),
                            y_hi - max([0.0, yz]),
                            z_hi
                            ]
                        box = Box(lo, hi, box_tilt)
                    else:
                        box = Box([x_lo, y_lo, z_lo], [x_hi, y_hi, z_hi])

                # atoms come fourth
                if state == 3 and self._section['atoms'] in line:
                    state += 1

                    # extract the scehma
                    if self.schema is None:
                        # mapping from LAMMPS dump keys to lammpsio
                        lammps_fields = {
                            'id': ('id', None),
                            'mol': ('molecule', None),
                            'type': ('typeid', None),
                            'mass': ('mass', None),
                            'x': ('position', 0),
                            'y': ('position', 1),
                            'z': ('position', 2),
                            'ix': ('image', 0),
                            'iy': ('image', 1),
                            'iz': ('image', 2),
                            'vx': ('velocity', 0),
                            'vy': ('velocity', 1),
                            'vz': ('velocity', 2),
                            'q': ('charge', None)}
                        schema = {}
                        schema_header = line.split()[2:]
                        for i, field in enumerate(schema_header):
                            if self._gzip:
                                field = field.decode()
                            if field in lammps_fields:
                                key, key_idx = lammps_fields[field]
                                if key_idx is not None:
                                    if key not in schema:
                                        schema[key] = [None, None, None]
                                    schema[key][key_idx] = i
                                else:
                                    schema[key] = i
                        # validate tuple
                        for key in ('position', 'velocity',' image'):
                            if key in schema and any(x is None for x in schema[key]):
                                raise IOError('lammpsio requires 3-element vectors')
                        self.schema = schema

                    snap = Snapshot(N,box,step)
                    for i in range(snap.N):
                        atom = _readline(f,True)
                        atom = atom.split()

                        if 'id' in self.schema:
                            id_ = int(atom[self.schema['id']])
                            if id_ != i+1:
                                snap.id[i] = id_
                        if 'position' in self.schema:
                            snap.position[i] = [float(atom[j]) for j in self.schema['position']]
                        if 'velocity' in self.schema:
                            snap.velocity[i] = [float(atom[j]) for j in self.schema['velocity']]
                        if 'image' in self.schema:
                            snap.image[i] = [int(atom[j]) for j in self.schema['image']]
                        if 'molecule' in self.schema:
                            snap.molecule[i] = int(atom[self.schema['molecule']])
                        if 'typeid' in self.schema:
                            snap.typeid[i] = int(atom[self.schema['typeid']])
                        if 'charge' in self.schema:
                            snap.charge[i] = float(atom[self.schema['charge']])
                        if 'mass' in self.schema:
                            snap.mass[i] = float(atom[self.schema['mass']])

                # final processing stage for the frame
                if state == 4:
                    # optionally sort the particles by ID
                    if self.sort_ids and snap.has_id():
                        snap.reorder(numpy.argsort(snap.id), check_order=False)

                    # optionally copy reference data by ID / index
                    if self._copy_from is not None:
                        if snap.N != self._copy_from.N:
                            raise ValueError('Cannot copy from a Snapshot with a different size')

                        if self._copy_from.has_id():
                            copy_id_map = {id_: i for i, id_ in enumerate(self._copy_from.id)}
                        else:
                            copy_id_map = {i+1: i for i in range(self._copy_from.N)}

                        if snap.has_id():
                            copy_id = [copy_id_map[id_] for id_ in snap.id]
                        else:
                            copy_id = [copy_id_map[id_] for id_ in range(1, snap.N+1)]

                        if not snap.has_typeid() and self._copy_from.has_typeid():
                            snap.typeid = self._copy_from.typeid[copy_id]
                        if not snap.has_molecule() and self._copy_from.has_molecule():
                            snap.molecule = self._copy_from.molecule[copy_id]
                        if not snap.has_charge() and self._copy_from.has_charge():
                            snap.charge = self._copy_from.charge[copy_id]
                        if not snap.has_mass() and self._copy_from.has_mass():
                            snap.mass = self._copy_from.mass[copy_id]

                    yield snap
                    del snap,N,box,step
                    state = 0

                line = _readline(f)
