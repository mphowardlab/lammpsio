API
---

Particles
=========

The particle configuration can be stored in a `Snapshot`, including the simulation `Box`
and timestep, per-particle properties like position and velocity, and topology
information (see next).

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

Particle data can be read and written using the following LAMMPS file formats:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    lammpsio.DataFile
    lammpsio.DumpFile
