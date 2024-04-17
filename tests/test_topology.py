import pytest

import lammpsio


def test_bonds_wrong_shape(snap_8):
    snap_8.bonds = lammpsio.topology.Bonds(N=6, num_types=2)

    # check that error is raised if array is the wrong shape
    id = [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(TypeError):
        snap_8.bonds.id = id

    typeid = [1, 1, 1, 2, 2, 2, 3]
    with pytest.raises(TypeError):
        snap_8.bonds.typeid = typeid

    members = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9]]
    with pytest.raises(TypeError):
        snap_8.bonds.members = members


def test_angles_wrong_shape(snap_8):
    snap_8.angles = lammpsio.topology.Angles(N=4, num_types=2)

    # check that error is raised if array is the wrong shape
    id = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        snap_8.angles.id = id

    typeid = [1, 1, 2, 2, 3]
    with pytest.raises(TypeError):
        snap_8.angles.typeid = typeid

    members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_8.angles.members = members


def test_dihedrals_wrong_shape(snap_8):
    snap_8.dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)

    # check that error is raised if array is the wrong shape
    id = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_8.dihedrals.id = id

    typeid = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_8.dihedrals.typeid = typeid

    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_8.dihedrals.members = members


def test_impropers_wrong_shape(snap_8):
    snap_8.impropers = lammpsio.topology.Impropers(N=2, num_types=2)

    # check that error is raised if array is the wrong shape
    id = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_8.impropers.id = id

    typeid = [1, 2, 3]
    with pytest.raises(TypeError):
        snap_8.impropers.typeid = typeid

    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    with pytest.raises(TypeError):
        snap_8.impropers.members = members
