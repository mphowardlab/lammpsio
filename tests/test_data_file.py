import os

import numpy
import pytest

import lammpsio

try:
    import lammps

    has_lammps = True
except ImportError:
    has_lammps = False


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


@pytest.mark.parametrize("shuffle_ids", [False, True])
def test_data_file_topology(snap_8, tmp_path, shuffle_ids):
    # set ids to be assigned
    if shuffle_ids:
        particle_id = [1, 5, 2, 6, 3, 7, 4, 8]
        bond_id = [1, 4, 2, 5, 3, 6]
        angle_id = [1, 3, 2, 4]
        dihedral_id = [2, 1]
        improper_id = [2, 1]
    else:
        particle_id = [1, 2, 3, 4, 5, 6, 7, 8]
        bond_id = [1, 2, 3, 4, 5, 6]
        angle_id = [1, 2, 3, 4]
        dihedral_id = [1, 2]
        improper_id = [1, 2]

    # particle information
    snap_8.id = particle_id
    snap_8.typeid = [1, 1, 1, 1, 2, 2, 2, 2]
    snap_8.position = [
        [0, 0, 0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [1, 1, 1],
        [1.1, 1.1, 1.1],
        [1.2, 1.2, 1.2],
        [1.3, 1.3, 1.3],
    ]
    snap_8.mass = [1, 1, 1, 1, 2, 2, 2, 2]

    # bond information
    snap_8.bonds = lammpsio.topology.Bonds(N=6, num_types=2)
    snap_8.bonds.id = bond_id
    snap_8.bonds.typeid = [1, 2, 1, 2, 1, 2]
    snap_8.bonds.members = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
    ]

    # angle information
    snap_8.angles = lammpsio.topology.Angles(N=4, num_types=2)
    snap_8.angles.id = angle_id
    snap_8.angles.typeid = [1, 2, 2, 1]
    snap_8.angles.members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
    ]

    # dihedral information
    snap_8.dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)
    snap_8.dihedrals.id = dihedral_id
    snap_8.dihedrals.typeid = [1, 2]
    snap_8.dihedrals.members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    # improper information
    snap_8.impropers = lammpsio.topology.Impropers(N=2, num_types=2)
    snap_8.impropers.id = improper_id
    snap_8.impropers.typeid = [1, 2]
    snap_8.impropers.members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    filename = tmp_path / "atoms.data"
    data = lammpsio.DataFile.create(filename, snap_8)
    assert filename.exists

    snap_2 = data.read()
    # test none topology features of a snapshot with topology
    assert snap_2.N == snap_8.N
    if shuffle_ids:
        assert snap_2.has_id()
        assert numpy.allclose(snap_2.id, snap_8.id)
    else:
        assert not snap_2.has_id()
    assert numpy.allclose(snap_2.id, snap_8.id)
    assert snap_2.has_typeid()
    assert numpy.allclose(snap_2.typeid, snap_8.typeid)
    assert snap_2.has_position()
    assert numpy.allclose(snap_2.position, snap_8.position)
    assert snap_2.has_mass()
    assert numpy.allclose(snap_2.mass, snap_8.mass)

    # test bonds
    assert snap_2.bonds.N == snap_8.bonds.N
    if shuffle_ids:
        assert snap_2.bonds.has_id()
        assert numpy.allclose(snap_2.bonds.id, snap_8.bonds.id)
    else:
        assert not snap_2.bonds.has_id()
    assert numpy.allclose(snap_2.bonds.id, snap_8.bonds.id)
    assert snap_2.bonds.has_typeid()
    assert numpy.allclose(snap_2.bonds.typeid, snap_8.bonds.typeid)
    assert snap_2.bonds.has_members()
    assert numpy.allclose(snap_2.bonds.members, snap_8.bonds.members)

    # test angles
    assert snap_2.angles.N == snap_8.angles.N
    if shuffle_ids:
        assert snap_2.angles.has_id()
        assert numpy.allclose(snap_2.angles.id, snap_8.angles.id)
    else:
        assert not snap_2.angles.has_id()
    assert numpy.allclose(snap_2.angles.id, snap_8.angles.id)
    assert snap_2.angles.has_typeid()
    assert numpy.allclose(snap_2.angles.typeid, snap_8.angles.typeid)
    assert snap_2.angles.has_members()
    assert numpy.allclose(snap_2.angles.members, snap_8.angles.members)

    # test dihedrals
    assert snap_2.dihedrals.N == snap_8.dihedrals.N
    if shuffle_ids:
        assert snap_2.dihedrals.has_id()
        assert numpy.allclose(snap_2.dihedrals.id, snap_8.dihedrals.id)
    else:
        assert not snap_2.dihedrals.has_id()
    assert numpy.allclose(snap_2.dihedrals.id, snap_8.dihedrals.id)
    assert snap_2.dihedrals.has_typeid()
    assert numpy.allclose(snap_2.dihedrals.typeid, snap_8.dihedrals.typeid)
    assert snap_2.dihedrals.has_members()
    assert numpy.allclose(snap_2.dihedrals.members, snap_8.dihedrals.members)

    # test impropers
    assert snap_2.impropers.N == snap_8.impropers.N
    if shuffle_ids:
        assert snap_2.impropers.has_id()
        assert numpy.allclose(snap_2.impropers.id, snap_8.impropers.id)
    else:
        assert not snap_2.impropers.has_id()
    assert numpy.allclose(snap_2.impropers.id, snap_8.impropers.id)
    assert snap_2.impropers.has_typeid()
    assert numpy.allclose(snap_2.impropers.typeid, snap_8.impropers.typeid)
    assert snap_2.impropers.has_members()
    assert numpy.allclose(snap_2.impropers.members, snap_8.impropers.members)


@pytest.mark.skipif(not has_lammps, reason="lammps not installed")
@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("atom_style", ["atomic", "molecular", "charge", "full"])
def test_data_file_min_lammps(tmp_path, snap, atom_style, shuffle_ids):
    if shuffle_ids:
        snap.id = [3, 1, 2]

    # write the data file with default values using lammpsio
    filename = tmp_path / "atoms.data"
    assert not snap.has_image()
    assert not snap.has_velocity()
    lammpsio.DataFile.create(filename, snap, atom_style)
    assert os.path.isfile(filename)

    # read it in LAMMPS and write it out
    lmps = lammps.lammps(cmdargs=["-log", "none"])
    cmds = [f"atom_style {atom_style}"]
    cmds += [f"read_data {filename}"]
    cmds += ["mass 1 1.0"]
    cmds += [f"write_data {filename} nocoeff nofix"]
    lmps.commands_list(cmds)
    lmps.close()

    # read it back in and check
    snap_2 = lammpsio.DataFile(filename).read()
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
    assert snap_2.has_typeid()
    assert numpy.allclose(snap_2.typeid, 1)

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


@pytest.mark.skipif(not has_lammps, reason="lammps not installed")
@pytest.mark.parametrize("shuffle_ids", [False, True])
@pytest.mark.parametrize("set_style", [True, False])
@pytest.mark.parametrize("atom_style", ["atomic", "molecular", "charge", "full"])
def test_data_file_all_lammps(snap, atom_style, set_style, shuffle_ids, tmp_path):
    if shuffle_ids:
        snap.id = [3, 1, 2]

    # write the data file with nondefault values using lammpsio
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

    # create the data file using lammpsio
    lammpsio.DataFile.create(filename, snap, atom_style if set_style else None)
    assert os.path.isfile(filename)

    lmps = lammps.lammps(cmdargs=["-log", "none"])
    cmds = [f"atom_style {atom_style}"]
    cmds += [f"read_data {filename}"]
    cmds += [f"write_data {filename}"]
    lmps.commands_list(cmds)
    lmps.close()

    # read it back in and check
    snap_2 = lammpsio.DataFile(filename).read()
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


@pytest.mark.skipif(not has_lammps, reason="lammps not installed")
@pytest.mark.parametrize("shuffle_ids", [False, True])
def test_data_file_topology_lammps(snap_8, tmp_path, shuffle_ids):
    # set ids to be assigned
    # Note: LAMMPS reorders the ids of topology objects when ids are shuffled
    # We check that topology objects typeid and members survive the round trip
    if shuffle_ids:
        particle_id = [1, 5, 2, 6, 3, 7, 4, 8]
        bond_id = [1, 4, 2, 5, 3, 6]
        angle_id = [1, 3, 2, 4]
        dihedral_id = [2, 1]
        improper_id = [2, 1]
    else:
        particle_id = [1, 2, 3, 4, 5, 6, 7, 8]
        bond_id = [1, 2, 3, 4, 5, 6]
        angle_id = [1, 2, 3, 4]
        dihedral_id = [1, 2]
        improper_id = [1, 2]

    # particle information
    snap_8.id = particle_id
    snap_8.typeid = [1, 1, 1, 1, 2, 2, 2, 2]
    snap_8.position = [
        [0, 0, 0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [1, 1, 1],
        [1.1, 1.1, 1.1],
        [1.2, 1.2, 1.2],
        [1.3, 1.3, 1.3],
    ]
    snap_8.mass = [1, 1, 1, 1, 2, 2, 2, 2]

    # bond information
    snap_8.bonds = lammpsio.topology.Bonds(N=6, num_types=2)
    snap_8.bonds.id = bond_id
    snap_8.bonds.typeid = [1, 2, 1, 2, 1, 2]
    snap_8.bonds.members = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
    ]

    # angle information
    snap_8.angles = lammpsio.topology.Angles(N=4, num_types=2)
    snap_8.angles.id = angle_id
    snap_8.angles.typeid = [1, 2, 2, 1]
    snap_8.angles.members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
    ]

    # dihedral information
    snap_8.dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)
    snap_8.dihedrals.id = dihedral_id
    snap_8.dihedrals.typeid = [1, 2]
    snap_8.dihedrals.members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    # improper information
    snap_8.impropers = lammpsio.topology.Impropers(N=2, num_types=2)
    snap_8.impropers.id = improper_id
    snap_8.impropers.typeid = [1, 2]
    snap_8.impropers.members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    if shuffle_ids:
        bond_id_lammpsio = [b - 1 for b in bond_id]
        angle_id_lammpsio = [a - 1 for a in angle_id]
        snap_8.bonds.reorder(bond_id_lammpsio, check_order=True)
        snap_8.angles.reorder(angle_id_lammpsio, check_order=True)

    filename = tmp_path / "atoms.data"
    # create the data file using lammpsio
    lammpsio.DataFile.create(filename, snap_8)

    # read it in LAMMPS and write it out
    lmps = lammps.lammps(cmdargs=["-log", "none"])
    cmds = ["atom_style molecular"]
    cmds += [f"read_data {filename}"]
    cmds += [f"write_data {filename}"]
    lmps.commands_list(cmds)
    lmps.close()

    # read it back in and check
    snap_2 = lammpsio.DataFile(filename).read()
    # test none topology features of a snapshot with topology
    assert snap_2.N == snap_8.N
    if shuffle_ids:
        assert snap_2.has_id()
        assert numpy.allclose(snap_2.id, snap_8.id)
    else:
        assert not snap_2.has_id()

    assert numpy.allclose(snap_2.id, snap_8.id)
    assert snap_2.has_typeid()
    assert numpy.allclose(snap_2.typeid, snap_8.typeid)
    assert snap_2.has_position()
    assert numpy.allclose(snap_2.position, snap_8.position)
    assert snap_2.has_mass()
    assert numpy.allclose(snap_2.mass, snap_8.mass)

    # test bonds
    assert snap_2.bonds.N == snap_8.bonds.N

    # Create arrays with bond members and typeid
    assert snap_2.bonds.has_typeid()
    assert snap_2.bonds.has_members()
    snap_8_bond_data = numpy.column_stack((snap_8.bonds.members, snap_8.bonds.typeid))
    snap_2_bond_data = numpy.column_stack((snap_2.bonds.members, snap_2.bonds.typeid))
    # test that types and bond members survive round trip
    # Note: This sorting only works because each member only appears in the first
    # column one time.
    assert numpy.allclose(
        snap_2_bond_data[snap_2_bond_data[:, 0].argsort()],
        snap_8_bond_data[snap_8_bond_data[:, 0].argsort()],
    )

    # test angles
    assert snap_2.angles.N == snap_8.angles.N

    # Create arrays with angle members and typeid
    assert snap_2.angles.has_typeid()
    assert snap_2.angles.has_members()
    snap_8_angle_data = numpy.column_stack(
        (snap_8.angles.members, snap_8.angles.typeid)
    )
    snap_2_angle_data = numpy.column_stack(
        (snap_2.angles.members, snap_2.angles.typeid)
    )
    # test that types and angle members survive round trip
    assert numpy.allclose(
        snap_2_angle_data[snap_2_angle_data[:, 0].argsort()],
        snap_8_angle_data[snap_8_angle_data[:, 0].argsort()],
    )

    # test dihedrals
    assert snap_2.dihedrals.N == snap_8.dihedrals.N

    # Create arrays with dihedral members and typeid
    assert snap_2.dihedrals.has_typeid()
    assert snap_2.dihedrals.has_members()
    snap_8_dihedral_data = numpy.column_stack(
        (snap_8.dihedrals.members, snap_8.dihedrals.typeid)
    )
    snap_2_dihedral_data = numpy.column_stack(
        (snap_2.dihedrals.members, snap_2.dihedrals.typeid)
    )
    # test that types and dihedral members survive round trip
    assert numpy.allclose(
        snap_2_dihedral_data[snap_2_dihedral_data[:, 0].argsort()],
        snap_8_dihedral_data[snap_8_dihedral_data[:, 0].argsort()],
    )

    # test impropers
    assert snap_2.impropers.N == snap_8.impropers.N

    # Create arrays with improper members and typeid
    assert snap_2.impropers.has_typeid()
    assert snap_2.impropers.has_members()
    snap_8_improper_data = numpy.column_stack(
        (snap_8.impropers.members, snap_8.impropers.typeid)
    )
    snap_2_improper_data = numpy.column_stack(
        (snap_2.impropers.members, snap_2.impropers.typeid)
    )
    # test that types and improper members survive round trip
    assert numpy.allclose(
        snap_2_improper_data[snap_2_improper_data[:, 0].argsort()],
        snap_8_improper_data[snap_8_improper_data[:, 0].argsort()],
    )
