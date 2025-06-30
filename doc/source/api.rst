API
---

Particle data
=============

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
