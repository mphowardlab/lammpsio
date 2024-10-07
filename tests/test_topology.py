import pytest

import lammpsio


def test_bonds(snap_8):
    # default is no bonds
    assert snap_8.bonds is None
    assert not snap_8.has_bonds()

    # empty bonds set still means we don't have bonds
    bonds = lammpsio.Bonds(N=0)
    snap_8.bonds = bonds
    assert snap_8.bonds is bonds
    assert not snap_8.has_bonds()

    # one bond counts as bonds
    bonds = lammpsio.Bonds(N=1)
    snap_8.bonds = bonds
    assert snap_8.bonds is bonds
    assert snap_8.has_bonds()

    # make sure can set back to None
    snap_8.bonds = None
    assert snap_8.bonds is None
    assert not snap_8.has_bonds()

    # make sure other types cannot be set and nothing changes
    with pytest.raises(TypeError):
        snap_8.bonds = lammpsio.Angles(N=0)
    assert snap_8.bonds is None
    assert not snap_8.has_bonds()


def test_angles(snap_8):
    # default is no angles
    assert snap_8.angles is None
    assert not snap_8.has_angles()

    # empty angles set still means we don't have angles
    angles = lammpsio.Angles(N=0)
    snap_8.angles = angles
    assert snap_8.angles is angles
    assert not snap_8.has_angles()

    # one bond counts as angles
    angles = lammpsio.Angles(N=1)
    snap_8.angles = angles
    assert snap_8.angles is angles
    assert snap_8.has_angles()

    # make sure can set back to None
    snap_8.angles = None
    assert snap_8.angles is None
    assert not snap_8.has_angles()

    # make sure other types cannot be set and nothing changes
    with pytest.raises(TypeError):
        snap_8.angles = lammpsio.Bonds(N=0)
    assert snap_8.angles is None
    assert not snap_8.has_angles()


def test_dihedrals(snap_8):
    # default is no dihedrals
    assert snap_8.dihedrals is None
    assert not snap_8.has_dihedrals()

    # empty dihedrals set still means we don't have dihedrals
    dihedrals = lammpsio.Dihedrals(N=0)
    snap_8.dihedrals = dihedrals
    assert snap_8.dihedrals is dihedrals
    assert not snap_8.has_dihedrals()

    # one bond counts as dihedrals
    dihedrals = lammpsio.Dihedrals(N=1)
    snap_8.dihedrals = dihedrals
    assert snap_8.dihedrals is dihedrals
    assert snap_8.has_dihedrals()

    # make sure can set back to None
    snap_8.dihedrals = None
    assert snap_8.dihedrals is None
    assert not snap_8.has_dihedrals()

    # make sure other types cannot be set and nothing changes
    with pytest.raises(TypeError):
        snap_8.dihedrals = lammpsio.Angles(N=0)
    assert snap_8.dihedrals is None
    assert not snap_8.has_dihedrals()


def test_impropers(snap_8):
    # default is no impropers
    assert snap_8.impropers is None
    assert not snap_8.has_impropers()

    # empty impropers set still means we don't have impropers
    impropers = lammpsio.Impropers(N=0)
    snap_8.impropers = impropers
    assert snap_8.impropers is impropers
    assert not snap_8.has_impropers()

    # one bond counts as impropers
    impropers = lammpsio.Impropers(N=1)
    snap_8.impropers = impropers
    assert snap_8.impropers is impropers
    assert snap_8.has_impropers()

    # make sure can set back to None
    snap_8.impropers = None
    assert snap_8.impropers is None
    assert not snap_8.has_impropers()

    # make sure other types cannot be set and nothing changes
    with pytest.raises(TypeError):
        snap_8.impropers = lammpsio.Angles(N=0)
    assert snap_8.impropers is None
    assert not snap_8.has_impropers()


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


def test_LabelMap():
    # create a simple label map
    label = lammpsio.topology.LabelMap({1: "typeA", 2: "typeB"})

    # check the types
    assert label.types == ("typeA", "typeB")
    # check the typeids
    assert label.typeid == (1, 2)
    # get
    assert label[2] == "typeB"
    # set
    label[3] = "typeC"
    assert label[3] == "typeC"
    # delete
    del label[3]
    assert 3 not in label
