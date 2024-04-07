import numpy
import pytest

import lammpsio


def test_bonds(snap_top):
    snap_top.bonds = lammpsio.topology.Bonds(N=6, num_types=2)
    id = [1, 2, 3, 4, 5, 6]
    typeid = [1, 1, 1, 2, 2, 2]
    members = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
    ]

    snap_top.bonds.id = id
    snap_top.bonds.typeid = typeid
    snap_top.bonds.members = members

    assert numpy.allclose(snap_top.bonds.id, id)
    assert numpy.allclose(snap_top.bonds.typeid, typeid)
    assert numpy.allclose(snap_top.bonds.members, members)

    # test cases that raise errors
    with pytest.raises(TypeError):
        snap_top.bonds = lammpsio.topology.Angles(N=6, num_types=2)

    id = [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(TypeError):
        snap_top.bonds.id = id

    typeid = [1, 1, 1, 2, 2, 2, 3]
    with pytest.raises(TypeError):
        snap_top.bonds.typeid = typeid

    members = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9]]
    with pytest.raises(TypeError):
        snap_top.bonds.members = members


def test_angles(snap_top):
    snap_top.angles = lammpsio.topology.Angles(N=4, num_types=2)
    id = [
        1,
        2,
        3,
        4,
    ]
    typeid = [1, 1, 2, 2]
    members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
    ]

    snap_top.angles.id = id
    snap_top.angles.typeid = typeid
    snap_top.angles.members = members

    assert numpy.allclose(snap_top.angles.id, id)
    assert numpy.allclose(snap_top.angles.typeid, typeid)
    assert numpy.allclose(snap_top.angles.members, members)

    # test cases that raise errors
    with pytest.raises(TypeError):
        snap_top.angles = lammpsio.topology.Bonds(N=4, num_types=2)

    id = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        snap_top.angles.id = id

    typeid = [1, 1, 2, 2, 3]
    with pytest.raises(TypeError):
        snap_top.angles.typeid = typeid

    members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_top.angles.members = members


def test_dihedrals(snap_top):
    snap_top.dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)
    id = [1, 2]
    typeid = [1, 2]
    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    snap_top.dihedrals.id = id
    snap_top.dihedrals.typeid = typeid
    snap_top.dihedrals.members = members

    assert numpy.allclose(snap_top.dihedrals.id, id)
    assert numpy.allclose(snap_top.dihedrals.typeid, typeid)
    assert numpy.allclose(snap_top.dihedrals.members, members)

    # test cases that raise errors
    with pytest.raises(TypeError):
        snap_top.dihedrals = lammpsio.topology.Angles(N=2, num_types=2)

    id = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_top.dihedrals.id = id

    typeid = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_top.dihedrals.typeid = typeid

    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_top.dihedrals.members = members


def test_impropers(snap_top):
    snap_top.impropers = lammpsio.topology.Impropers(N=2, num_types=2)
    id = [1, 2]
    typeid = [1, 2]
    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    snap_top.impropers.id = id
    snap_top.impropers.typeid = typeid
    snap_top.impropers.members = members

    assert numpy.allclose(snap_top.impropers.id, id)
    assert numpy.allclose(snap_top.impropers.typeid, typeid)
    assert numpy.allclose(snap_top.impropers.members, members)

    # test cases that raise errors
    with pytest.raises(TypeError):
        snap_top.impropers = lammpsio.topology.Angles(N=2, num_types=2)

    id = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_top.impropers.id = id

    typeid = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_top.impropers.typeid = typeid

    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_top.impropers.members = members
