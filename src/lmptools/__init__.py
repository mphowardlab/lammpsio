import gzip
import numpy

class Box:
    def __init__(self, low, high, tilt=None):
        self.low = low
        self.high = high
        self.tilt = tilt

    @classmethod
    def cast(cls, value):
        if isinstance(value,Box):
            return value
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape == (9,):
            return Box(v[:3],v[3:6],v[6:])
        elif v.shape == (6,):
            return Box(v[:3],v[3:])
        else:
            raise TypeError('Unable to cast boxlike object with shape {}'.format(v.shape))

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape != (3,):
            raise TypeError('Low must be a 3-tuple')
        self._low = v

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape != (3,):
            raise TypeError('High must be a 3-tuple')
        self._high = v

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        v = value
        if v is not None:
            v = numpy.array(v, ndmin=1, copy=False, dtype=numpy.float64)
            if v.shape != (3,):
                raise TypeError('Tilt must be a 3-tuple')
        self._tilt = v

class Snapshot:
    def __init__(self, N, box, step=None):
        self._N = N
        self._box = Box.cast(box)
        self.step = step

        self._position = None
        self._velocity = None
        self._image = None
        self._molecule = None
        self._typeid = None
        self._charge = None
        self._mass = None

    @property
    def N(self):
        return self._N

    @property
    def box(self):
        return self._box

    @property
    def position(self):
        if not self.has_position():
            self._position = numpy.zeros((self.N,3),dtype=numpy.float64)
        return self._position

    @position.setter
    def position(self, value):
        v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.float64)
        if v.shape != (self.N,3):
            raise TypeError('Positions must be an Nx3 array')
        self._position = v

    def has_position(self):
        return self._position is not None

    @property
    def image(self):
        if not self.has_image():
            self._image = numpy.zeros((self.N,3),dtype=numpy.int32)
        return self._image

    @image.setter
    def image(self, value):
        v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.int32)
        if v.shape != (self.N,3):
            raise TypeError('Images must be an Nx3 array')
        self._image = v

    def has_image(self):
        return self._image is not None

    @property
    def velocity(self):
        if not self.has_velocity():
            self._velocity = numpy.zeros((self.N,3),dtype=numpy.float64)
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        v = numpy.array(value, ndmin=2, copy=False, dtype=numpy.float64)
        if v.shape != (self.N,3):
            raise TypeError('Velocities must be an Nx3 array')
        self._velocity = v

    def has_velocity(self):
        return self._velocity is not None

    @property
    def molecule(self):
        if not self.has_molecule():
            self._molecule = numpy.zeros(self.N,dtype=numpy.int32)
        return self._molecule

    @molecule.setter
    def molecule(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.int32)
        if v.shape != (self.N,):
            raise TypeError('Molecules must be a size N array')
        self._molecule = v

    def has_molecule(self):
        return self._molecule is not None

    @property
    def typeid(self):
        if not self.has_typeid():
            self._typeid = numpy.ones(self.N,dtype=numpy.int32)
        return self._typeid

    @typeid.setter
    def typeid(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.int32)
        if v.shape != (self.N,):
            raise TypeError('Type must be a size N array')
        self._typeid = v

    def has_typeid(self):
        return self._typeid is not None

    @property
    def charge(self):
        if not self.has_charge():
            self._charge = numpy.zeros(self.N,dtype=numpy.float64)
        return self._charge

    @charge.setter
    def charge(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape != (self.N,):
            raise TypeError('Charge must be a size N array')
        self._charge = v

    def has_charge(self):
        return self._charge is not None

    @property
    def mass(self):
        if not self.has_mass():
            self._mass = numpy.ones(self.N,dtype=numpy.float64)
        return self._mass

    @mass.setter
    def mass(self, value):
        v = numpy.array(value, ndmin=1, copy=False, dtype=numpy.float64)
        if v.shape != (self.N,):
            raise TypeError('Mass must be a size N array')
        self._mass = v

    def has_mass(self):
        return self._mass is not None

def readline_(file_,require=False):
    line = file_.readline()
    if require and len(line) == 0:
        raise OSError('Could not read line from file')
    return line

class DataFile:
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
        # validate snapshot
        if not snapshot.has_position():
            raise ValueError('Snapshot does not have positions')
        elif not snapshot.has_typeid():
            raise ValueError('Snapshot does not have typeids')

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
                style_fmt = '{atomid:8d}{molid:8d}{typeid:4d}{q:8.5f}{x:16.8f}{y:16.8f}{z:16.8f}'
            elif style == 'charge':
                style_fmt = '{atomid:8d}{typeid:4d}{q:8.5f}{x:16.8f}{y:16.8f}{z:16.8f}'
            elif style == 'molecular':
                style_fmt = '{atomid:8d}{molid:8d}{typeid:4d}{x:16.8f}{y:16.8f}{z:16.8f}'
            elif style == 'atomic':
                style_fmt = '{atomid:8d}{typeid:4d}{x:16.8f}{y:16.8f}{z:16.8f}'
            else:
                raise ValueError('Unknown atom style')
            if snapshot.has_image():
                style_fmt += '{ix:8d}{iy:8d}{iz:8d}'
            # write section
            f.write("\nAtoms # {}\n\n".format(style))
            for i in range(snapshot.N):
                style_args = dict(
                    atomid=i+1,
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
                        atomid=i+1,
                        vx=snapshot.velocity[i][0],
                        vy=snapshot.velocity[i][1],
                        vz=snapshot.velocity[i][2]))

            # Masses section
            if masses is not None:
                f.write("\nMasses\n\n")
                for i,mi in enumerate(masses):
                    f.write("{typeid:4d}{m:12}\n".format(typeid=i+1,m=mi))

        return DataFile(filename)

    def read(self):
        with open(self.filename) as f:
            # initialize snapshot from header
            N = None
            box_bounds = [None,None,None,None,None,None]
            box_tilt = None
            num_types = None
            # skip first line
            readline_(f,True)
            line = readline_(f)
            while len(line) > 0:
                line = line.rstrip()

                # skip blank and go to next line
                if len(line) == 0:
                    line = readline_(f)
                    continue

                # check for unknown headers and go to next line
                skip_line = False
                for h in self.unknown_headers:
                    if h in line:
                        skip_line = True
                        break
                if skip_line:
                    line = readline_(f)
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
                line = readline_(f)

            if N is None:
                raise IOError('Number of particles not read')
            elif num_types is None:
                raise IOError('Number of types not read')
            elif None in box_bounds:
                raise IOError('Box bounds not read')
            box = Box(box_bounds[:3], box_bounds[3:], box_tilt)
            snap = Snapshot(N,box)

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
                    readline_(f,True) # blank line
                    for i in range(snap.N):
                        # strip out comments
                        row = readline_(f,True).split()
                        try:
                            comment = row.index('#')
                            row = row[:comment]
                        except ValueError:
                            pass

                        # check that row is correctly sized and process
                        if not (len(row) == style_cols+4 or len(row) == style_cols+7):
                            raise IOError('Expected number of columns not read for atom style')
                        idx = int(row[0])-1
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
                    readline_(f,True) # blank line
                    for i in range(snap.N):
                        row = readline_(f,True).split()
                        if len(row) < 4:
                            raise IOError('Expected number of columns not read for velocity')
                        idx = int(row[0])-1
                        snap.velocity[idx] = [float(x) for x in row[1:4]]
                elif 'Masses' in line:
                    masses = {}
                    readline_(f,True) # blank line
                    for i in range(num_types):
                        row = readline_(f,True).split()
                        if len(row) < 2:
                            raise IOError('Expected number of columns not read for mass')
                        masses[int(row[0])] = float(row[1])
                else:
                    # silently ignore unknown sections / lines
                    pass

                line = readline_(f)

            # set mass on particles at end, in case sections were out of order in file
            if masses is not None:
                for typeid,m in masses.items():
                    snap.mass[snap.typeid == typeid] = m

        return snap

class DumpFile:
    def __init__(self, filename, schema, mass=None):
        self.filename = filename
        self.schema = schema
        self.mass = mass
        self._frames = None

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        with open(self._filename,'rb') as f:
            self._gzip = f.read(2) == b'\x1f\x8b'

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
        return self._schema

    @schema.setter
    def schema(self, value):
        # validate schema
        if 'id' not in value:
            raise KeyError('Schema must include the particle id')
        if 'position' in value and len(value['position']) != 3:
            raise ValueError('Position must be a 3-tuple')
        if 'velocity' in value and len(value['velocity']) != 3:
            raise ValueError('Velocity must be a 3-tuple')
        if 'image' in value and len(value['image']) != 3:
            raise ValueError('Image must be a 3-tuple')
        self._schema = value

    def _open(self):
        if self._gzip:
            f = gzip.open(self.filename,'r')
        else:
            f = open(self.filename,'r')
        return f

    def _find_frames(self):
        self._frames = []
        with self._open() as f:
            line = readline_(f)
            line_num = 0
            while len(line) > 0:
                if self._section['step'] in line:
                    self._frames.append(line_num)
                line = readline_(f)
                line_num += 1

    def __len__(self):
        if self._frames is None:
            self._find_frames()
        return len(self._frames)

    def __iter__(self):
        with self._open() as f:
            state = 0
            line = readline_(f)
            while len(line) > 0:
                # timestep line first
                if state == 0 and self._section['step'] in line:
                    state += 1
                    step = int(readline_(f,True))

                # number of particles second
                if state == 1 and self._section['natoms'] in line:
                    state += 1
                    N = int(readline_(f,True))

                # box size third
                if state == 2 and self._section['box'] in line:
                    state += 1
                    # check for triclinic
                    box_header = line.split()
                    if len(box_header) >= 5:
                        box_tilt = [xy, xz, yz]
                    else:
                        box_tilt = None
                    box_x = readline_(f,True)
                    box_y = readline_(f,True)
                    box_z = readline_(f,True)
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
                    snap = Snapshot(N,box,step)
                    for i in range(snap.N):
                        atom = readline_(f,True)
                        atom = atom.split()

                        tag = int(atom[self.schema['id']]) - 1

                        if 'position' in self.schema:
                            snap.position[tag] = [float(atom[j]) for j in self.schema['position']]
                        if 'velocity' in self.schema:
                            snap.velocity[tag] = [float(atom[j]) for j in self.schema['velocity']]
                        if 'image' in self.schema:
                            snap.image[tag] = [int(atom[j]) for j in self.schema['image']]
                        if 'molecule' in self.schema:
                            snap.molecule[tag] = int(atom[self.schema['molecule']])
                        elif 'mol' in self.schema:
                            # allow both mol and molecule for backwards compatibility
                            snap.molecule[tag] = int(atom[self.schema['mol']])
                        if 'typeid' in self.schema:
                            snap.typeid[tag] = int(atom[self.schema['typeid']])
                        if 'charge' in self.schema:
                            snap.charge[tag] = float(atom[self.schema['charge']])
                        if self.mass is not None:
                            snap.mass[tag] = self.mass[snap.typeid[tag]]

                # final processing stage for the frame
                if state == 4:
                    yield snap
                    del snap,N,box,step
                    state = 0

                line = readline_(f)
