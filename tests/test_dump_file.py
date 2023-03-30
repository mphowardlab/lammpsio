import copy

import numpy
import pytest

import lammpsio


@pytest.mark.parametrize("sort_ids", [False, True])
@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("use_gzip", [False, True])
def test_dump_file_min(snap, use_gzip, shuffle_ids, sort_ids, tmp_path):
    # create file with 2 snapshots with defaults, changing N & step
    snap_2 = lammpsio.Snapshot(snap.N + 2, snap.box, snap.step + 1)
    snaps = [snap, snap_2]
    if shuffle_ids:
        for s in snaps:
            s.id = s.id[::-1]
    if use_gzip:
        filename = tmp_path / "atoms.lammpstrj.gz"
    else:
        filename = tmp_path / "atoms.lammpstrj"
    schema = {"id": 0, "position": (1, 2, 3)}
    f = lammpsio.DumpFile.create(filename, schema, snaps)
    assert filename.exists
    assert len(f) == 2

    # read it back in and check snapshots
    f2 = lammpsio.DumpFile(filename, sort_ids=sort_ids)
    read_snaps = [s for s in f2]
    for i in range(2):
        assert read_snaps[i].N == snaps[i].N
        assert read_snaps[i].step == snaps[i].step
        assert numpy.allclose(read_snaps[i].box.low, snaps[i].box.low)
        assert numpy.allclose(read_snaps[i].box.high, snaps[i].box.high)
        if snaps[i].box.tilt is not None:
            assert numpy.allclose(read_snaps[i].box.tilt, snaps[i].box.tilt)
        else:
            assert read_snaps[i].box.tilt is None
        if shuffle_ids:
            assert read_snaps[i].has_id()
            if sort_ids:
                assert numpy.allclose(read_snaps[i].id, numpy.arange(1, snaps[i].N + 1))
            else:
                assert numpy.allclose(
                    read_snaps[i].id, numpy.arange(1, snaps[i].N + 1)[::-1]
                )
        else:
            assert not read_snaps[i].has_id()
        assert read_snaps[i].has_position()
        assert numpy.allclose(read_snaps[i].position, 0)
        assert not read_snaps[i].has_image()
        assert not read_snaps[i].has_velocity()
        assert not read_snaps[i].has_typeid()
        assert not read_snaps[i].has_mass()
        assert not read_snaps[i].has_molecule()
        assert not read_snaps[i].has_charge()


@pytest.mark.parametrize("sort_ids", [False, True])
@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("use_gzip", [False, True])
def test_dump_file_all(snap, use_gzip, shuffle_ids, sort_ids, tmp_path):
    snap.position = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [0.7, 0.8, 0.9]]
    snap.image = [[1, 2, 3], [-4, -5, -6], [7, 8, 9]]
    snap.velocity = [[-3, -2, -1], [6, 5, 4], [9, 8, 7]]
    snap.typeid = [2, 1, 2]
    snap.mass = [3, 2, 3]
    snap.molecule = [2, 0, 1]
    snap.charge = [-1, 0, 1]

    snap_2 = lammpsio.Snapshot(snap.N, snap.box, snap.step + 1)
    snap_2.position = snap.position[::-1]
    snap_2.image = snap.image[::-1]
    snap_2.velocity = snap.velocity[::-1]
    snap_2.typeid = snap.typeid[::-1]
    snap_2.mass = snap.mass[::-1]
    snap_2.molecule = snap.molecule[::-1]
    snap_2.charge = snap.charge[::-1]
    snaps = [snap, snap_2]

    if shuffle_ids:
        for s in snaps:
            s.id = s.id[::-1]
        if sort_ids:
            order = numpy.arange(snap.N)[::-1]
        else:
            order = numpy.arange(snap.N)
    else:
        order = numpy.arange(snap.N)

    # create file with 2 snapshots
    schema = {
        "id": 13,
        "position": (11, 10, 12),
        "image": (9, 8, 7),
        "velocity": (4, 5, 6),
        "typeid": 3,
        "mass": 2,
        "molecule": 1,
        "charge": 0,
    }
    if use_gzip:
        filename = tmp_path / "atoms.lammpstrj.gz"
    else:
        filename = tmp_path / "atoms.lammpstrj"
    f = lammpsio.DumpFile.create(filename, schema, snaps)
    assert filename.exists
    assert len(f) == 2

    # read it back in and check snapshots
    f2 = lammpsio.DumpFile(filename, sort_ids=sort_ids)
    read_snaps = [s for s in f2]
    for i, s in enumerate(f):
        assert read_snaps[i].N == snaps[i].N
        assert read_snaps[i].step == snaps[i].step
        assert numpy.allclose(read_snaps[i].box.low, snaps[i].box.low)
        assert numpy.allclose(read_snaps[i].box.high, snaps[i].box.high)
        if snaps[i].box.tilt is not None:
            assert numpy.allclose(read_snaps[i].box.tilt, snaps[i].box.tilt)
        else:
            assert read_snaps[i].box.tilt is None
        if shuffle_ids:
            assert read_snaps[i].has_id()
            if sort_ids:
                assert numpy.allclose(read_snaps[i].id, numpy.arange(1, snaps[i].N + 1))
            else:
                assert numpy.allclose(
                    read_snaps[i].id, numpy.arange(1, snaps[i].N + 1)[::-1]
                )
        else:
            assert not read_snaps[i].has_id()
        assert read_snaps[i].has_position()
        assert numpy.allclose(read_snaps[i].position, snaps[i].position[order])
        assert read_snaps[i].has_image()
        assert numpy.allclose(read_snaps[i].image, snaps[i].image[order])
        assert read_snaps[i].has_velocity()
        assert numpy.allclose(read_snaps[i].velocity, snaps[i].velocity[order])
        assert read_snaps[i].has_typeid()
        assert numpy.allclose(read_snaps[i].typeid, snaps[i].typeid[order])
        assert read_snaps[i].has_mass()
        assert numpy.allclose(read_snaps[i].mass, snaps[i].mass[order])
        assert read_snaps[i].has_molecule()
        assert numpy.allclose(read_snaps[i].molecule, snaps[i].molecule[order])
        assert read_snaps[i].has_charge()
        assert numpy.allclose(read_snaps[i].charge, snaps[i].charge[order])


def test_copy_from(snap, tmp_path):
    ref_snap = copy.deepcopy(snap)
    ref_snap.id = [12, 0, 1]
    ref_snap.typeid = [2, 1, 2]
    ref_snap.mass = [3, 2, 3]
    ref_snap.molecule = [2, 0, 1]
    ref_snap.charge = [-1, 0, 1]

    snap.id = [0, 1, 12]
    snap.position = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [0.7, 0.8, 0.9]]

    filename = tmp_path / "atoms.lammpstrj"
    schema = {"id": 0, "position": (1, 2, 3)}
    lammpsio.DumpFile.create(filename, schema, snap)
    assert filename.exists

    f = lammpsio.DumpFile(filename, schema, copy_from=ref_snap)
    read_snap = [s for s in f][0]

    assert read_snap.N == snap.N
    assert read_snap.step == snap.step
    assert numpy.allclose(read_snap.box.low, snap.box.low)
    assert numpy.allclose(read_snap.box.high, snap.box.high)
    if snap.box.tilt is not None:
        assert numpy.allclose(read_snap.box.tilt, snap.box.tilt)
    else:
        assert read_snap.box.tilt is None
    assert read_snap.has_id()
    assert numpy.allclose(read_snap.id, snap.id)
    assert read_snap.has_position()
    assert numpy.allclose(read_snap.position, snap.position)
    assert not read_snap.has_image()
    assert not read_snap.has_velocity()
    assert read_snap.has_typeid()
    assert numpy.all(read_snap.typeid == [1, 2, 2])
    assert read_snap.has_mass()
    assert numpy.allclose(read_snap.mass, [2, 3, 3])
    assert read_snap.has_molecule()
    assert numpy.all(read_snap.molecule == [0, 1, 2])
    assert read_snap.has_charge()
    assert numpy.allclose(read_snap.charge, [0, 1, -1])
