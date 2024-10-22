import copy

import numpy
import pytest
from packaging import version

import lammpsio

try:
    import gsd.hoomd

    has_gsd = True
except ModuleNotFoundError:
    has_gsd = False


def test_create(snap):
    assert snap.N == 3
    assert isinstance(snap.box, lammpsio.Box)
    assert snap.step == 10


@pytest.mark.skipif(not has_gsd, reason="gsd not installed")
def test_gsd_conversion():
    # make a GSD frame
    try:
        gsd_version = gsd.version.version
    except AttributeError:
        gsd_version = gsd.__version__
    if version.Version(gsd_version) >= version.Version("2.8.0"):
        frame = gsd.hoomd.Frame()
    else:
        frame = gsd.hoomd.Snapshot()
    frame.configuration.step = 3
    frame.configuration.box = [4, 5, 6, 0.1, 0.2, 0.3]
    frame.particles.N = 2
    frame.particles.position = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    frame.particles.image = [[1, -1, 0], [0, 2, -2]]
    frame.particles.velocity = [[1, 2, 3], [-4, -5, -6]]
    frame.particles.body = [0, -1]
    frame.particles.types = ["A", "B"]
    frame.particles.typeid = [1, 0]
    frame.particles.mass = [3, 2]
    frame.particles.charge = [-1, 1]

    # make Snapshot from GSD
    snap, type_map = lammpsio.Snapshot.from_hoomd_gsd(frame)
    assert snap.step == 3
    assert numpy.allclose(snap.box.low, [-2, -2.5, -3])
    assert numpy.allclose(snap.box.high, [2, 2.5, 3])
    assert numpy.allclose(snap.box.tilt, [0.5, 1.2, 1.8])
    assert snap.N == 2
    assert numpy.allclose(snap.position, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    assert numpy.all(snap.image == [[1, -1, 0], [0, 2, -2]])
    assert numpy.allclose(snap.velocity, [[1, 2, 3], [-4, -5, -6]])
    assert numpy.all(snap.molecule == [1, 0])
    assert numpy.all(snap.typeid == [2, 1])
    assert snap.type_label == {1: "A", 2: "B"}
    assert numpy.allclose(snap.mass, [3, 2])
    assert numpy.allclose(snap.charge, [-1, 1])
    assert type_map == {1: "A", 2: "B"}

    # go back to GSD frame
    frame2 = snap.to_hoomd_gsd()
    assert frame2.configuration.step == frame.configuration.step
    assert numpy.allclose(frame2.configuration.box, frame.configuration.box)
    assert frame2.particles.N == frame.particles.N
    assert numpy.allclose(frame2.particles.position, frame.particles.position)
    assert numpy.all(frame2.particles.image == frame.particles.image)
    assert numpy.allclose(frame2.particles.velocity, frame.particles.velocity)
    assert numpy.all(frame2.particles.types == tuple(frame.particles.types))
    assert numpy.all(frame2.particles.typeid == frame.particles.typeid)
    assert numpy.allclose(frame2.particles.mass, frame.particles.mass)
    assert numpy.allclose(frame2.particles.charge, frame.particles.charge)
    assert numpy.all(frame2.particles.body == frame.particles.body)

    # do the same thing, but use explicit type map (should give warning)
    with pytest.warns(DeprecationWarning):
        frame3 = snap.to_hoomd_gsd({1: "C", 2: "D"})
    assert numpy.all(frame3.particles.types == ("C", "D"))
    assert numpy.all(frame3.particles.typeid == [1, 0])

    # do the same thing, but with different type_label
    snap.type_label = lammpsio.topology.LabelMap({1: "C", 2: "D"})
    frame4 = snap.to_hoomd_gsd()
    assert numpy.all(frame4.particles.types == tuple(["C", "D"]))
    assert numpy.all(frame4.particles.typeid == [1, 0])

    # check for warning on floppy molecules
    frame.particles.body = [-2, -1]
    with pytest.warns():
        lammpsio.Snapshot.from_hoomd_gsd(frame)

    # check for barebones Snapshot going to GSD
    snap2 = lammpsio.Snapshot(N=2, box=lammpsio.Box([-5, -5, -5], [5, 5, 5]))
    snap2.to_hoomd_gsd()
    # check again with id remapping to make sure order is preserved
    snap2.id = [2, 1]
    snap2.to_hoomd_gsd()
    assert numpy.all(snap2.id == [2, 1])

    # check for error out on bad box
    snap2.box.low = [-10, -10, -10]
    with pytest.raises(ValueError):
        snap2.to_hoomd_gsd()


@pytest.mark.skipif(not has_gsd, reason="gsd not installed")
def test_minimal_gsd_conversion():
    # make a GSD frame
    try:
        gsd_version = gsd.version.version
    except AttributeError:
        gsd_version = gsd.__version__
    if version.Version(gsd_version) >= version.Version("2.8.0"):
        frame = gsd.hoomd.Frame()
    else:
        frame = gsd.hoomd.Snapshot()

    frame.configuration.box = [4, 5, 6, 0.1, 0.2, 0.3]
    frame.particles.N = 2
    frame.particles.types = ["A", "B"]
    frame.particles.typeid = [1, 0]
    frame.particles.position = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]

    # make Snapshot from GSD with minimial information
    snap, type_map = lammpsio.Snapshot.from_hoomd_gsd(frame)
    # check that unspecified properties are not set
    assert snap.has_velocity() is False
    assert snap.has_image() is False
    assert snap.has_charge() is False
    assert snap.has_mass() is False
    assert snap.has_molecule() is False
    assert snap.has_bonds() is False
    assert snap.has_angles() is False
    assert snap.has_dihedrals() is False
    assert snap.has_impropers() is False


@pytest.mark.skipif(not has_gsd, reason="gsd not installed")
def test_gsd_conversion_topology():
    # make a GSD frame
    try:
        gsd_version = gsd.version.version
    except AttributeError:
        gsd_version = gsd.__version__
    if version.Version(gsd_version) >= version.Version("2.8.0"):
        frame = gsd.hoomd.Frame()
    else:
        frame = gsd.hoomd.Snapshot()

    # set box
    frame.configuration.box = [10, 10, 10, 1, 1, 1]

    # particles
    position = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [3.0, 3.0, 0.0],
        [3.0, 4.0, 0.0],
        [4.0, 3.0, 0.0],
        [4.0, 4.0, 0.0],
    ]
    frame.particles.N = 8
    frame.particles.types = ["A", "B"]
    frame.particles.typeid = [0, 0, 0, 0, 1, 1, 1, 1]
    frame.particles.position = position

    # bonds
    frame.bonds.N = 6
    frame.bonds.types = ["bondA", "bondB"]
    frame.bonds.typeid = [0, 1, 0, 1, 0, 1]
    frame.bonds.group = [
        [0, 1],
        [1, 2],
        [2, 3],
        [4, 5],
        [5, 6],
        [6, 7],
    ]

    # angles
    frame.angles.N = 4
    frame.angles.types = ["angleA", "angleB"]
    frame.angles.typeid = [0, 1, 0, 1]
    frame.angles.group = [
        [0, 1, 2],
        [1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
    ]

    # dihedrals
    frame.dihedrals.N = 2
    frame.dihedrals.types = ["dihedralA", "dihedralB"]
    frame.dihedrals.typeid = [0, 1]
    frame.dihedrals.group = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]

    # impropers
    frame.impropers.N = 2
    frame.impropers.types = ["improperA", "improperB"]
    frame.impropers.typeid = [0, 1]
    frame.impropers.group = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]

    # make Snapshot from GSD
    snap, type_map = lammpsio.Snapshot.from_hoomd_gsd(frame)
    # particles
    assert snap.N == 8
    assert numpy.allclose(snap.position, position)
    assert numpy.all(snap.typeid == [1, 1, 1, 1, 2, 2, 2, 2])
    assert type_map == {1: "A", 2: "B"}
    assert snap.type_label == {1: "A", 2: "B"}
    # bonds
    assert snap.bonds.N == 6
    assert numpy.all(snap.bonds.typeid == [1, 2, 1, 2, 1, 2])
    assert numpy.all(
        snap.bonds.members == [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]]
    )
    assert snap.bonds.type_label == {1: "bondA", 2: "bondB"}
    # angles
    assert snap.angles.N == 4
    assert numpy.all(snap.angles.typeid == [1, 2, 1, 2])
    assert numpy.all(
        snap.angles.members == [[1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]]
    )
    assert snap.angles.type_label == {1: "angleA", 2: "angleB"}
    # dihedrals
    assert snap.dihedrals.N == 2
    assert numpy.all(snap.dihedrals.typeid == [1, 2])
    assert numpy.all(snap.dihedrals.members == [[1, 2, 3, 4], [5, 6, 7, 8]])
    assert snap.dihedrals.type_label == {1: "dihedralA", 2: "dihedralB"}
    # impropers
    assert snap.impropers.N == 2
    assert numpy.all(snap.impropers.typeid == [1, 2])
    assert numpy.all(snap.impropers.members == [[1, 2, 3, 4], [5, 6, 7, 8]])
    assert snap.impropers.type_label == {1: "improperA", 2: "improperB"}

    # go back to GSD frame
    frame2 = snap.to_hoomd_gsd()
    assert numpy.allclose(frame2.configuration.box, frame.configuration.box)
    # particles
    assert frame2.particles.N == frame.particles.N
    assert numpy.allclose(frame2.particles.position, frame.particles.position)
    assert numpy.all(frame2.particles.types == tuple(frame.particles.types))
    assert numpy.all(frame2.particles.typeid == frame.particles.typeid)
    # bonds
    assert frame2.bonds.N == frame.bonds.N
    assert numpy.all(frame2.bonds.types == tuple(frame.bonds.types))
    assert numpy.all(frame2.bonds.typeid == frame.bonds.typeid)
    assert numpy.all(frame2.bonds.group == frame.bonds.group)
    # angles
    assert frame2.angles.N == frame.angles.N
    assert numpy.all(frame2.angles.types == tuple(frame.angles.types))
    assert numpy.all(frame2.angles.typeid == frame.angles.typeid)
    assert numpy.all(frame2.angles.group == frame.angles.group)
    # dihedrals
    assert frame2.dihedrals.N == frame.dihedrals.N
    assert numpy.all(frame2.dihedrals.types == tuple(frame.dihedrals.types))
    assert numpy.all(frame2.dihedrals.typeid == frame.dihedrals.typeid)
    assert numpy.all(frame2.dihedrals.group == frame.dihedrals.group)
    # impropers
    assert frame2.impropers.N == frame.impropers.N
    assert numpy.all(frame2.impropers.types == tuple(frame.impropers.types))
    assert numpy.all(frame2.impropers.typeid == frame.impropers.typeid)
    assert numpy.all(frame2.impropers.group == frame.impropers.group)


def test_position(snap):
    assert not snap.has_position()
    assert numpy.allclose(snap.position, 0.0)
    position = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [100, -200, 300]]
    snap.position = position
    assert numpy.allclose(snap.position, position)
    with pytest.raises(TypeError):
        snap.position = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(TypeError):
        snap.position = [0, 0, 0]


def test_image(snap):
    assert not snap.has_image()
    assert numpy.allclose(snap.image, 0)
    image = [[1, 2, 3], [-1, 0, -2], [100, -200, 300]]
    snap.image = image
    assert numpy.allclose(snap.image, image)
    with pytest.raises(TypeError):
        snap.image = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(TypeError):
        snap.image = [0, 0, 0]


def test_velocity(snap):
    assert not snap.has_velocity()
    assert numpy.allclose(snap.velocity, 0.0)
    velocity = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [100, -200, 300]]
    snap.velocity = velocity
    assert numpy.allclose(snap.velocity, velocity)
    with pytest.raises(TypeError):
        snap.velocity = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(TypeError):
        snap.velocity = [0, 0, 0]


def test_typeid(snap):
    assert not snap.has_typeid()
    assert numpy.allclose(snap.typeid, 1)
    typeid = [2, 1, 3]
    snap.typeid = typeid
    assert numpy.allclose(snap.typeid, typeid)
    with pytest.raises(TypeError):
        snap.typeid = [1, 1]


def test_molecule(snap):
    assert not snap.has_molecule()
    assert numpy.allclose(snap.molecule, 0)
    molecule = [2, 0, 1]
    snap.molecule = molecule
    assert numpy.allclose(snap.molecule, molecule)
    with pytest.raises(TypeError):
        snap.molecule = [0, 0]


def test_mass(snap):
    assert not snap.has_mass()
    assert numpy.allclose(snap.mass, 1)
    # also set typeid, so that types and masses are consistent as LAMMPS requires
    # removing this line doesn't actually do anything, though
    snap.typeid = [1, 2, 1]
    mass = [2, 3, 2]
    snap.mass = mass
    assert numpy.allclose(snap.mass, mass)
    with pytest.raises(TypeError):
        snap.mass = [1, 1]


def test_bonds(snap_8):
    snap_8.bonds = lammpsio.topology.Bonds(N=6, num_types=2)
    id = [1, 2, 3, 4, 5, 6]
    typeid = [1, 2, 1, 2, 1, 2]
    members = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
    ]

    snap_8.bonds.id = id
    snap_8.bonds.typeid = typeid
    snap_8.bonds.members = members

    assert numpy.allclose(snap_8.bonds.id, id)
    assert numpy.allclose(snap_8.bonds.typeid, typeid)
    assert numpy.allclose(snap_8.bonds.members, members)


def test_angles(snap_8):
    snap_8.angles = lammpsio.topology.Angles(N=4, num_types=2)
    id = [
        1,
        2,
        3,
        4,
    ]
    typeid = [1, 2, 1, 2]
    members = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
    ]

    snap_8.angles.id = id
    snap_8.angles.typeid = typeid
    snap_8.angles.members = members

    assert numpy.allclose(snap_8.angles.id, id)
    assert numpy.allclose(snap_8.angles.typeid, typeid)
    assert numpy.allclose(snap_8.angles.members, members)


def test_dihedrals(snap_8):
    snap_8.dihedrals = lammpsio.topology.Dihedrals(N=2, num_types=2)
    id = [1, 2]
    typeid = [1, 2]
    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    snap_8.dihedrals.id = id
    snap_8.dihedrals.typeid = typeid
    snap_8.dihedrals.members = members

    assert numpy.allclose(snap_8.dihedrals.id, id)
    assert numpy.allclose(snap_8.dihedrals.typeid, typeid)
    assert numpy.allclose(snap_8.dihedrals.members, members)


def test_impropers(snap_8):
    snap_8.impropers = lammpsio.topology.Impropers(N=2, num_types=2)
    id = [1, 2]
    typeid = [1, 2]
    members = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    snap_8.impropers.id = id
    snap_8.impropers.typeid = typeid
    snap_8.impropers.members = members

    assert numpy.allclose(snap_8.impropers.id, id)
    assert numpy.allclose(snap_8.impropers.typeid, typeid)
    assert numpy.allclose(snap_8.impropers.members, members)


def test_charge(snap):
    assert not snap.has_charge()
    assert numpy.allclose(snap.charge, 0)
    charge = [-1, 0, 1]
    snap.charge = charge
    assert numpy.allclose(snap.charge, charge)
    with pytest.raises(TypeError):
        snap.typeid = [0, 0]


def test_copy(snap):
    snap.id = [2, 0, 1]
    snap.position = [[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6], [0.7, 0.8, 0.9]]
    snap.image = [[1, 2, 3], [-4, -5, -6], [7, 8, 9]]
    snap.velocity = [[-3, -2, -1], [6, 5, 4], [9, 8, 7]]
    snap.typeid = [2, 1, 2]
    snap.mass = [3, 2, 3]
    snap.molecule = [2, 0, 1]
    snap.charge = [-1, 0, 1]

    # check shallow copy works
    snap_shallow = copy.copy(snap)
    assert numpy.all(snap_shallow.id == snap.id)
    assert numpy.allclose(snap_shallow.position, snap.position)
    assert numpy.all(snap_shallow.image == snap.image)
    assert numpy.allclose(snap_shallow.velocity, snap.velocity)
    assert numpy.all(snap_shallow.typeid == snap.typeid)
    assert numpy.allclose(snap_shallow.mass, snap.mass)
    assert numpy.all(snap_shallow.molecule == snap.molecule)
    assert numpy.allclose(snap_shallow.charge, snap.charge)

    # check deep copy works
    snap_deep = copy.deepcopy(snap)
    assert numpy.all(snap_deep.id == snap.id)
    assert numpy.allclose(snap_deep.position, snap.position)
    assert numpy.all(snap_deep.image == snap.image)
    assert numpy.allclose(snap_deep.velocity, snap.velocity)
    assert numpy.all(snap_deep.typeid == snap.typeid)
    assert numpy.allclose(snap_deep.mass, snap.mass)
    assert numpy.all(snap_deep.molecule == snap.molecule)
    assert numpy.allclose(snap_deep.charge, snap.charge)

    # change the original snapshot and check shallow changes, but deep does not
    old_ids = list(snap.id)
    snap.id = [3, 4, 5]
    assert numpy.all(snap_shallow.id == snap.id)
    assert numpy.all(snap_deep.id == old_ids)
