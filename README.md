# lammpsio

[![PyPI version](https://img.shields.io/pypi/v/lammpsio.svg)](https://pypi.org/project/lammpsio)
[![PyPI downloads](https://img.shields.io/pypi/dm/lammpsio)](https://pypi.org/project/lammpsio)
[![Conda](https://img.shields.io/conda/dn/conda-forge/lammpsio)](https://anaconda.org/conda-forge/lammpsio)

Tools for working with LAMMPS data and dump files.

`lammpsio` is a pure Python package that can be installed using `pip`:

    pip install lammpsio

or `conda`:

    conda install -c conda-forge lammpsio

## Snapshot

The particle configuration is stored in a `Snapshot`. A `Snapshot` holds the
data for *N* particles, the simulation `Box`, and the timestep. The `Box` follows
the LAMMPS conventions for its shape and bounds. Here is a 3-particle
configuration in an orthorhombic box centered at the origin at step 100:

    box = lammpsio.Box((-2,-3,-4), (2,3,4))
    snapshot = lammpsio.Snapshot(3, box, step=100)

These constructor arguments are available as attributes:

- `N`: number of particles (int)
- `box`: bounding box (`Box`)
- `step`: timestep counter (int)
- `num_types`: number of particle types (int). If `num_types is None`, then the number of types is deduced from `typeid`.

The data contained in a `Snapshot` per particle is:

- `id`: (*N*,) array atom IDs (dtype: `int`, default: runs from 1 to *N*)
- `position`: (*N*,3) array of coordinates (dtype: `float`, default: `(0,0,0)`)
- `image`: (*N*,3) array of periodic image indexes (dtype: `int`, default: `(0,0,0)`)
- `velocity`: (*N*,3) array of velocities (dtype: `float`, default: `(0,0,0)`)
- `molecule`: (*N*,) array of molecule indexes (dtype: `int`, default: `0`)
- `typeid`: (*N*,) array of type indexes (dtype: `int`, default: `1`)
- `mass`: (*N*,) array of masses (dtype: `float`, default: `1`)
- `charge`: (*N*,) array of charges (dtype: `float`, default: `0`)

The optional topology data is:

- `type_label`: Labels of particle typeids. (`LabelMap`, default: `None`)
- `bonds`: Bond data (`Bonds`, default: `None`)
- `angles`: Angle data (`Angles`, default: `None`)
- `dihedrals`: Dihedral data (`Dihedrals`, default: `None`)
- `impropers`: Improper data (`Impropers`, default: `None`)

All values of indexes will follow the LAMMPS 1-indexed convention, but the
arrays themselves are 0-indexed.

The `Snapshot` will lazily initialize these per-particle arrays as they are
accessed to save memory. Hence, accessing a per-particle property will allocate
it to default values. If you want to check if an attribute has been set, use the
corresponding `has_` method instead (e.g., `has_position()`):

    snapshot.position = [[0,0,0],[1,-1,1],[1.5,2.5,-3.5]]
    snapshot.typeid[2] = 2
    if not snapshot.has_mass():
        snapshot.mass = [2.,2.,10.]

## Topology

The topology (bond information) can be stored in `Bonds`, `Angles`, `Dihedrals`,
and `Impropers` objects. All these objects function similarly, differing only in the
number of particles that are included in a connection (2 for a bond, 3 for an angle,
4 for a dihedral or improper). Each connection has an associated `id` and `typeid`.

```py
bonds = Bonds(N=3, num_types=2)
angles = Angles(N=2, num_types=1)
```
These constructor arguments are available as attributes:

- `N`: number of connections (int)
- `num_types`: number of connection types (int). If `num_types is None`, then the number of types is deduced from `typeid`.

The data contained per connection is:
- `members`: (*N*, *M*) array of particles IDs in each topology (dtype: `int`, default: `1`),
where *M* is the number of particles in a connection.
- `id`: (*N*,) array topology IDs (dtype: `int`, default: runs from 1 to *N*)
- `typeid`: (*N*,) array of type indexes (dtype: `int`, default: `1`)

A label (type) can be associated with a connection's typeid using a `type_label`.
- `type_label`: Labels of connection typeids. (`LabelMap`, default: `None`)

All values of indexes will follow the LAMMPS 1-indexed convention, but the
arrays themselves are 0-indexed. Lazy array initialization is used as for the `Snapshot`.

## Label maps

A `LabelMap` is effectively a dictionary associating a label (type) with a particle's
or connection's typeid. These labels can be useful for tracking the meaning of
typeids. They are also automatically used when interconverting with
HOOMD GSD files that require such labels.

The keys of the `LabelMap` are the typeids, and the values are the labels. A
`LabelMap` additionally has the following attributes:

- `types`: Types in label map. (dtype: `tuple`, default: `()`)
- `typeids`: Typeids in label map. (dtype: `tuple`, default: `()`)

## Data files

A LAMMPS data file is represented by a `DataFile`. The file must be explicitly
`read()` to get a `Snapshot`:

    f = lammpsio.DataFile("config.data")
    snapshot = f.read()

The `atom_style` will be read from the comment in the Atoms section
of the file. If it is not present, it must be specified in the `DataFile`.
If `atom_style` is specified and also present in the file, the two must match
or an error will be raised.

There are many sections that can be stored in a data file, but `lammpsio` does
not currently understand all of them. You can check `DataFile.known_headers`,
`DataFile.unknown_headers`, `DataFile.known_bodies` and `DataFile.unknown_bodies`
for lists of what is currently supported.

A `Snapshot` can be written using the `create()` method:

    f = lammpsio.DataFile.create("config2.data", snapshot)

A `DataFile` corresponding to the new file is returned by `create()`.

## Dump files

A LAMMPS dump file is represented by a `DumpFile`. The actual file format is
very flexible, but by default embeds a schema that can be read:

    traj = lammpsio.DumpFile(filename="atoms.lammpstrj")

If the schema does not exist for some reason, it can be manually specified as
a dictionary. Valid keys for the schema match the names and shapes in the
`Snapshot`. The keys requiring only 1 column index are: `id`, `typeid`,
`molecule`, `charge`, and `mass`. The keys requiring 3 column indexes are
`position`, `velocity`, and `image`.

LAMMPS will dump particles in an unknown order unless you have used the
`dump_modify sort` option. If you want particles to be ordered by `id` in the
`Snapshot`, use `sort_ids=True` (default).

A `DumpFile` is iterable, so you can use it to go through all the snapshots
of a trajectory:

    for snap in traj:
        print(snap.step)

You can also get the number of snapshots in the `DumpFile`, but this does
require reading the entire file: so use with caution!

    num_frames = len(traj)

Random access to snapshots is not currently implemented, but it may be added
in future. If you want to randomly access snapshots, you should load the
whole file into a list:

    snaps = [snap for snap in traj]
    print(snaps[3].step)

Keep in the mind that the memory requirements for this can be huge!

A `DumpFile` can be created from a list of snapshots:

    t = lammpsio.DumpFile.create("atoms.lammpstrj", schema, snaps)

The object representing the new file is returned and can be used.
