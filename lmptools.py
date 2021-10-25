# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
import gzip
import numpy as np

class Box:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    @classmethod
    def cast(cls, value):
        if isinstance(value,Box):
            return value
        v = np.array(value, ndmin=1, copy=False, dtype=np.float64)
        if v.shape == (6,):
            return Box(v[:3],v[3:])
        else:
            raise TypeError('Unable to cast boxlike object with shape {}'.format(v.shape))

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.float64)
        if v.shape != (3,):
            raise TypeError('Low must be a 3-tuple')
        self._low = v

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.float64)
        if v.shape != (3,):
            raise TypeError('High must be a 3-tuple')
        self._high = v

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
            self._position = np.zeros((self.N,3),dtype=np.float64)
        return self._position

    @position.setter
    def position(self, value):
        v = np.array(value, ndmin=2, copy=False, dtype=np.float64)
        if v.shape != (self.N,3):
            raise TypeError('Positions must be an Nx3 array')
        self._position = v

    def has_position(self):
        return self._position is not None

    @property
    def velocity(self):
        if not self.has_velocity():
            self._velocity = np.zeros((self.N,3),dtype=np.float64)
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        v = np.array(value, ndmin=2, copy=False, dtype=np.float64)
        if v.shape != (self.N,3):
            raise TypeError('Velocities must be an Nx3 array')
        self._velocity = v

    def has_velocity(self):
        return self._velocity is not None

    @property
    def image(self):
        if not self.has_image():
            self._image = np.zeros((self.N,3),dtype=np.int32)
        return self._image

    @image.setter
    def image(self, value):
        v = np.array(value, ndmin=2, copy=False, dtype=np.int32)
        if v.shape != (self.N,3):
            raise TypeError('Images must be an Nx3 array')
        self._image = v

    def has_image(self):
        return self._image is not None

    @property
    def molecule(self):
        if not self.has_molecule():
            self._molecule = np.zeros(self.N,dtype=np.int32)
        return self._molecule

    @molecule.setter
    def molecule(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.int32)
        if v.shape != (self.N,):
            raise TypeError('Molecules must be a size N array')
        self._molecule = v

    def has_molecule(self):
        return self._molecule is not None

    @property
    def typeid(self):
        if not self.has_typeid():
            self._typeid = np.zeros(self.N,dtype=np.int32)
        return self._typeid

    @typeid.setter
    def typeid(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.int32)
        if v.shape != (self.N,):
            raise TypeError('Type must be a size N array')
        self._typeid = v

    def has_typeid(self):
        return self._typeid is not None

    @property
    def charge(self):
        if not self.has_charge():
            self._charge = np.zeros(self.N,dtype=np.float64)
        return self._charge

    @charge.setter
    def charge(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.float64)
        if v.shape != (self.N,):
            raise TypeError('Charge must be a size N array')
        self._charge = v

    def has_charge(self):
        return self._charge is not None

    @property
    def mass(self):
        if not self.has_mass():
            self._mass = np.zeros(self.N,dtype=np.float64)
        return self._mass

    @mass.setter
    def mass(self, value):
        v = np.array(value, ndmin=1, copy=False, dtype=np.float64)
        if v.shape != (self.N,):
            raise TypeError('Mass must be a size N array')
        self._mass = v

    def has_mass(self):
        return self._mass is not None

class DataFile:
    @classmethod
    def create(cls, filename, snapshot):
        # validate snapshot
        if not snapshot.has_position():
            raise ValueError('Snapshot does not have positions')
        elif not snapshot.has_typeid():
            raise ValueError('Snapshot does not have typeids')

        # extract number of types
        num_types = np.amax(np.unique(snapshot.typeid))

        # extract mass by type
        if snapshot.has_mass():
            masses = np.empty(num_types)
            for i in range(num_types):
                mi = snapshot.mass[snapshot.typeid == i+1]
                if not np.all(mi == mi[0]):
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

            # Atoms section
            if snapshot.has_charge() and snapshot.has_molecule():
                style = 'full'
                style_fmt = '{atomid:8d}{molid:8d}{typeid:4d}{q:8.5f}{x:16.8f}{y:16.8f}{z:16.8f}'
            elif snapshot.has_charge():
                style = 'charge'
                style_fmt = '{atomid:8d}{typeid:4d}{q:8.5f}{x:16.8f}{y:16.8f}{z:16.8f}'
            elif snapshot.has_molecule():
                style = 'molecular'
                style_fmt = '{atomid:8d}{molid:8d}{typeid:4d}{x:16.8f}{y:16.8f}{z:16.8f}'
            else:
                style = 'atomic'
                style_fmt = '{atomid:8d}{typeid:4d}{x:16.8f}{y:16.8f}{z:16.8f}'
            if snapshot.has_image():
                style_fmt += '{ix:8d}{iy:8d}{iz:8d}'
            f.write("\nAtoms # {}\n\n".format(style))
            for i in range(snapshot.N):
                style_args = dict(
                    atomid=i+1,
                    typeid=snapshot.typeid[i],
                    x=snapshot.position[i][0],
                    y=snapshot.position[i][1],
                    z=snapshot.position[i][2])
                if snapshot.has_charge():
                    style_args.update(q=snapshot.charge[i])
                if snapshot.has_molecule():
                    style_args.update(molid=snapshot.molecule[i])
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
            raise KeyError('Scheme must include the particle id')
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

    @staticmethod
    def _readline(f,require=False):
        line = f.readline()
        if require and len(line) == 0:
            raise OSError('Could not read line from file')
        return line

    def _find_frames(self):
        self._frames = []
        with self._open() as f:
            line = self._readline(f)
            line_num = 0
            while len(line) > 0:
                if self._section['step'] in line:
                    self._frames.append(line_num)
                line = self._readline(f)
                line_num += 1

    def __len__(self):
        if self._frames is None:
            self._find_frames()
        return len(self._frames)

    def __iter__(self):
        with self._open() as f:
            state = 0
            line = self._readline(f)
            while len(line) > 0:
                # timestep line first
                if state == 0 and self._section['step'] in line:
                    state += 1
                    step = int(self._readline(f,True))

                # number of particles second
                if state == 1 and self._section['natoms'] in line:
                    state += 1
                    N = int(self._readline(f,True))

                # box size third
                if state == 2 and self._section['box'] in line:
                    state += 1
                    box_x = self._readline(f,True)
                    box_y = self._readline(f,True)
                    box_z = self._readline(f,True)
                    x_lo, x_hi = [float(x) for x in box_x.split()]
                    y_lo, y_hi = [float(y) for y in box_y.split()]
                    z_lo, z_hi = [float(z) for z in box_z.split()]
                    box = Box([x_lo,y_lo,z_lo],[x_hi,y_hi,z_hi])

                # atoms come fourth
                if state == 3 and self._section['atoms'] in line:
                    state += 1
                    snap = Snapshot(N,box,step)
                    for i in range(snap.N):
                        atom = self._readline(f,True)
                        atom = atom.split()

                        tag = int(atom[self.schema['id']]) - 1

                        if 'position' in self.schema:
                            snap.position[tag] = [float(atom[j]) for j in self.schema['position']]
                        if 'velocity' in self.schema:
                            snap.velocity[tag] = [float(atom[j]) for j in self.schema['velocity']]
                        if 'image' in self.schema:
                            snap.image[tag] = [int(atom[j]) for j in self.schema['image']]
                        if 'mol' in self.schema:
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

                line = self._readline(f)
