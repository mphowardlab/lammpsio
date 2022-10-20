import gzip
import numpy
import pytest

import lmptools

@pytest.mark.parametrize('use_gzip', [False, True])
def test_dump_file_min(snap, use_gzip, tmp_path):
    # minimum file info
    snap.position = [[0.1,0.2,0.3],[-0.4,-0.5,-0.6],[0.7,0.8,0.9]]
    snap_2 = lmptools.Snapshot(snap.N, snap.box, snap.step+1)
    snap_2.position = snap.position[::-1]
    snaps = [snap, snap_2]

    # create file with 2 snapshots
    if use_gzip:
        filename = tmp_path / "atoms.lammpstrj.gz"
    else:
        filename = tmp_path / "atoms.lammpstrj"
    schema = {'id': 0, 'position': (1, 2, 3)}
    f = lmptools.DumpFile.create(filename, schema, snaps)
    assert len(f) == 2

    # read it back in and check snapshots
    read_snaps = [s for s in f]
    for i,s in enumerate(f):
        assert read_snaps[i].N == snaps[i].N
        assert read_snaps[i].step == snaps[i].step
        assert numpy.allclose(read_snaps[i].box.low, snaps[i].box.low)
        assert numpy.allclose(read_snaps[i].box.high, snaps[i].box.high)
        if snaps[i].box.tilt is not None:
            assert numpy.allclose(read_snaps[i].box.tilt, snaps[i].box.tilt)
        else:
            assert read_snaps[i].box.tilt is None
        assert snaps[i].has_position()
        assert numpy.allclose(read_snaps[i].position, snaps[i].position)
        assert not snaps[i].has_image()
        assert not snaps[i].has_velocity()
        assert not snaps[i].has_typeid()
        assert not snaps[i].has_mass()
        assert not snaps[i].has_molecule()
        assert not snaps[i].has_charge()

@pytest.mark.parametrize('use_gzip', [False, True])
def test_dump_file_all(snap, use_gzip, tmp_path):
    snap.position = [[0.1,0.2,0.3],[-0.4,-0.5,-0.6],[0.7,0.8,0.9]]
    snap.image = [[1,2,3],[-4,-5,-6],[7,8,9]]
    snap.velocity = [[-3,-2,-1],[6,5,4],[9,8,7]]
    snap.typeid = [2,1,2]
    snap.mass = [3,2,3]
    snap.molecule = [2,0,1]
    snap.charge = [-1,0,1]

    snap_2 = lmptools.Snapshot(snap.N, snap.box, snap.step+1)
    snap_2.position = snap.position[::-1]
    snap_2.image = snap.image[::-1]
    snap_2.velocity = snap.velocity[::-1]
    snap_2.typeid = snap.typeid[::-1]
    snap_2.mass = snap.mass[::-1]
    snap_2.molecule = snap.molecule[::-1]
    snap_2.charge = snap.charge[::-1]
    snaps = [snap, snap_2]

    # create file with 2 snapshots
    schema = {
            'id': 13,
            'position': (11, 10, 12),
            'image': (9, 8, 7),
            'velocity': (4, 5, 6),
            'typeid': 3,
            'mass': 2,
            'molecule': 1,
            'charge': 0
            }
    if use_gzip:
        filename = tmp_path / "atoms.lammpstrj.gz"
    else:
        filename = tmp_path / "atoms.lammpstrj"
    f = lmptools.DumpFile.create(filename, schema, snaps)

    assert len(f) == 2

    # read it back in and check snapshots
    read_snaps = [s for s in f]
    for i,s in enumerate(f):
        assert read_snaps[i].N == snaps[i].N
        assert read_snaps[i].step == snaps[i].step
        assert numpy.allclose(read_snaps[i].box.low, snaps[i].box.low)
        assert numpy.allclose(read_snaps[i].box.high, snaps[i].box.high)
        if snaps[i].box.tilt is not None:
            assert numpy.allclose(read_snaps[i].box.tilt, snaps[i].box.tilt)
        else:
            assert read_snaps[i].box.tilt is None
        assert snaps[i].has_position()
        assert numpy.allclose(read_snaps[i].position, snaps[i].position)
        assert snaps[i].has_image()
        assert numpy.allclose(read_snaps[i].image, snaps[i].image)
        assert snaps[i].has_velocity()
        assert numpy.allclose(read_snaps[i].velocity, snaps[i].velocity)
        assert snaps[i].has_typeid()
        assert numpy.allclose(read_snaps[i].typeid, snaps[i].typeid)
        assert snaps[i].has_mass()
        assert numpy.allclose(read_snaps[i].mass, snaps[i].mass)
        assert snaps[i].has_molecule()
        assert numpy.allclose(read_snaps[i].molecule, snaps[i].molecule)
        assert snaps[i].has_charge()
        assert numpy.allclose(read_snaps[i].charge, snaps[i].charge)
