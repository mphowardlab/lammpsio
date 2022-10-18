# lmptools

Tools for working with LAMMPS data and dump files.

`lmptools` is a pure Python package that can be installed using `pip`:

    pip install lmptools

## Snapshot

The particle configuration is stored in a `Snapshot`. A `Snapshot` holds the
data for *N* particles, the simulation `Box`, and the timestep. The `Box` follows
the LAMMPS conventions for its shape and bounds. Here is a 3-particle
configuration in an orthorhombic box centered at the origin at step 100:

    box = lmptools.Box((-2,-3,-4), (2,3,4))
    snapshot = lmptools.Snapshot(3, box, step=100)

These constructor arguments are available as attributes:

- `N`: number of particles (int)
- `box`: bounding box (`Box`)
- `step`: timestep counter (int)

The data contained in a `Snapshot` per particle is:

- `position`: (*N*,3) array of coordinates (dtype: `numpy.float64`)
- `image`: (*N*,3) array of periodic image indexes (dtype: `numpy.int32`)
- `velocity`: (*N*,3) array of velocities (dtype: `numpy.float64`)
- `molecule`: (*N*,) array of molecule indexes (dtype: `numpy.int32`)
- `typeid`: (*N*,) array of type indexes (dtype: `numpy.int32`)
- `mass`: (*N*,) array of masses (dtype: `numpy.float64`)
- `charge`: (*N*,) array of charges (dtype: `numpy.float64`)

All values of indexes will follow the LAMMPS 1-indexed convention, but the
arrays themselves are 0-indexed.

The `Snapshot` will lazily initialize these per-particle arrays as they are
accessed to save memory. Hence, accessing a per-particle property will allocate
it to default values. If you want to check if an attribute has been set, use the
corresponding `has_` method instead (e.g., `has_position()`). The setters are
zero-copy if the assigned values are already NumPy arrays with the proper shape
and data type:

    snapshot.position = [[0,0,0],[1,-1,1],[1.5,2.5,-3.5]]
    snapshot.typeid[2] = 2
    if not snapshot.has_mass():
        snapshot.mass = [2.,2.,10.]

## Data files

A LAMMPS data file is represented by a `DataFile`. The file must be explicitly
`read()` to get a `Snapshot:

    f = lmptools.DataFile("config.data")
    snapshot = f.read()

The `atom_style` will be read from the comment in the Atoms section
of the file. If it is not present, it must be specified in the `DataFile`.
If `atom_style` is specified and also present in the file, the two must match
or an error will be raised.

There are many sections that can be stored in a data file, but `lmptools` does
not currently understand all of them. You can check `DataFile.known_headers`,
`DataFile.unknown_headers`, `DataFile.known_bodies` and `DataFile.unknown_bodies`
for lists of what is currently supported.

A `Snapshot` can be written using the `create()` method:

    f = lmptools.DataFile.create("config2.data", snapshot)

A `DataFile` corresponding to the new file is returned by `create()`.

## Dump files

A LAMMPS dump file is represented by a `DumpFile`. The actual file format is
very flexible, so a schema needs to be specified to parse it.

    traj = lmptools.DumpFile(
            filename="particles.lammpstrj",
            schema={"id": 0, "typeid": 1, "position": (2, 3, 4)}
            )

Valid keys for the schema match the names and shapes in the `Snapshot`. The
keys requiring only 1 column index are: `id`, `typeid`, `molecule`, `charge`,
and `mass`. The keys requiring 3 column indexes are `position`, `velocity`,
and `image`.
