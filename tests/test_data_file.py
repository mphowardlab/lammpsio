import numpy
import pytest

import lammpsio


@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("atom_style", ["atomic", "molecular", "charge", "full"])
def test_data_file_min(snap, atom_style, shuffle_ids, tmp_path):
    if shuffle_ids:
        snap.id = [2, 0, 1]
    # write the data file with default values
    filename = tmp_path / "atoms.data"
    data = lammpsio.DataFile.create(filename, snap, atom_style)
    assert filename.exists

    # read it back in and check
    snap_2 = data.read()
    assert snap_2.N == snap.N
    assert snap_2.step is None
    assert numpy.allclose(snap_2.box.low, snap.box.low)
    assert numpy.allclose(snap_2.box.high, snap.box.high)
    if snap.box.tilt is not None:
        assert numpy.allclose(snap_2.box.tilt, snap.box.tilt)
    else:
        assert snap_2.box.tilt is None
    if shuffle_ids:
        assert snap_2.has_id()
        assert numpy.allclose(snap_2.id, snap.id)
    else:
        assert not snap_2.has_id()
    assert snap_2.has_position()
    assert numpy.allclose(snap_2.position, 0)
    assert not snap_2.has_image()
    assert not snap_2.has_velocity()
    assert snap_2.has_typeid()
    assert numpy.allclose(snap_2.typeid, 1)
    assert not snap_2.has_mass()
    if atom_style in ("molecular", "full"):
        assert snap_2.has_molecule()
        assert numpy.allclose(snap_2.molecule, 0)
    else:
        assert not snap_2.has_molecule()
    if atom_style in ("charge", "full"):
        assert snap_2.has_charge()
        assert numpy.allclose(snap_2.charge, 0)
    else:
        assert not snap_2.has_charge()


@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("set_style", [True, False])
@pytest.mark.parametrize("atom_style", ["atomic", "molecular", "charge", "full"])
def test_data_file_all(snap, atom_style, set_style, shuffle_ids, tmp_path):
    if shuffle_ids:
        snap.id = [2, 0, 1]
    # write the data file with nondefault values
    snap.position = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [0.7, 0.8, 0.9]]
    snap.image = [[1, 2, 3], [-4, -5, -6], [7, 8, 9]]
    snap.velocity = [[-3, -2, -1], [6, 5, 4], [9, 8, 7]]
    snap.typeid = [2, 1, 2]
    snap.mass = [3, 2, 3]
    if atom_style in ("molecular", "full"):
        snap.molecule = [2, 0, 1]
    if atom_style in ("charge", "full"):
        snap.charge = [-1, 0, 1]
    filename = tmp_path / "atoms.data"
    data = lammpsio.DataFile.create(filename, snap, atom_style if set_style else None)
    assert filename.exists

    # read it back in and check
    snap_2 = data.read()
    assert snap_2.N == snap.N
    assert snap_2.step is None
    assert numpy.allclose(snap_2.box.low, snap.box.low)
    assert numpy.allclose(snap_2.box.high, snap.box.high)
    if snap.box.tilt is not None:
        assert numpy.allclose(snap_2.box.tilt, snap.box.tilt)
    else:
        assert snap_2.box.tilt is None
    if shuffle_ids:
        assert snap_2.has_id()
        assert numpy.allclose(snap_2.id, snap.id)
    else:
        assert not snap_2.has_id()
    assert snap_2.has_position()
    assert numpy.allclose(snap_2.position, snap.position)
    assert snap_2.has_velocity()
    assert numpy.allclose(snap_2.velocity, snap.velocity)
    assert snap_2.has_image()
    assert numpy.allclose(snap_2.image, snap.image)
    assert snap_2.has_typeid()
    assert numpy.allclose(snap_2.typeid, snap.typeid)
    assert snap_2.has_mass()
    assert numpy.allclose(snap_2.mass, snap.mass)
    if atom_style in ("molecular", "full"):
        assert snap_2.has_molecule()
        assert numpy.allclose(snap_2.molecule, snap.molecule)
    else:
        assert not snap_2.has_molecule()
    if atom_style in ("charge", "full"):
        assert snap_2.has_charge()
        assert numpy.allclose(snap_2.charge, snap.charge)
    else:
        assert not snap_2.has_charge()
