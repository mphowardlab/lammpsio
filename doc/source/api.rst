API
---

Particle data
=============

The particle data is stored in the `Snapshot`, `Box` and `LabelMap` objects.
`Snapshot` contains information about the particles, their positions, velocities, and other properties.
The `Box` object defines the simulation box dimensions and shape.
The `LabelMap` object is used to map particle labels (types) with a particle's or connection's typeid.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    lammpsio.Box
    lammpsio.Snapshot
    lammpsio.LabelMap

Topology
========

The topology (bond information) can be stored in `Bonds`, `Angles`, `Dihedrals`,
and `Impropers` objects. All these objects function similarly, differing only in
the number of particles that are included in a connection (2 for a bond, 3 for
an angle, 4 for a dihedral or improper).

.. autosummary::
    :nosignatures:
    :toctree: generated/

    lammpsio.Angles
    lammpsio.Bonds
    lammpsio.Dihedrals
    lammpsio.Impropers

File formats
============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    lammpsio.DataFile
    lammpsio.DumpFile
