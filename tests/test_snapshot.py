import numpy
import pytest

import lammpsio

def test_create(snap):
    assert snap.N == 3
    assert isinstance(snap.box, lammpsio.Box)
    assert snap.step == 10

def test_position(snap):
    assert not snap.has_position()
    assert numpy.allclose(snap.position, 0.)
    position = [[0.1,0.2,0.3],[-0.4,-0.5,-0.6],[100,-200,300]]
    snap.position = position
    assert numpy.allclose(snap.position, position)
    with pytest.raises(TypeError):
        snap.position = [[0,0,0],[0,0,0]]
    with pytest.raises(TypeError):
        snap.position = [0,0,0]

def test_image(snap):
    assert not snap.has_image()
    assert numpy.allclose(snap.image, 0)
    image = [[1,2,3],[-1,0,-2],[100,-200,300]]
    snap.image = image
    assert numpy.allclose(snap.image, image)
    with pytest.raises(TypeError):
        snap.image = [[0,0,0],[0,0,0]]
    with pytest.raises(TypeError):
        snap.image = [0,0,0]

def test_velocity(snap):
    assert not snap.has_velocity()
    assert numpy.allclose(snap.velocity, 0.)
    velocity = [[0.1,0.2,0.3],[-0.4,-0.5,-0.6],[100,-200,300]]
    snap.velocity = velocity
    assert numpy.allclose(snap.velocity, velocity)
    with pytest.raises(TypeError):
        snap.velocity = [[0,0,0],[0,0,0]]
    with pytest.raises(TypeError):
        snap.velocity = [0,0,0]

def test_typeid(snap):
    assert not snap.has_typeid()
    assert numpy.allclose(snap.typeid, 1)
    typeid = [2,1,3]
    snap.typeid = typeid
    assert numpy.allclose(snap.typeid, typeid)
    with pytest.raises(TypeError):
        snap.typeid = [1,1]

def test_molecule(snap):
    assert not snap.has_molecule()
    assert numpy.allclose(snap.molecule, 0)
    molecule = [2,0,1]
    snap.molecule = molecule
    assert numpy.allclose(snap.molecule, molecule)
    with pytest.raises(TypeError):
        snap.molecule = [0,0]

def test_mass(snap):
    assert not snap.has_mass()
    assert numpy.allclose(snap.mass, 1)
    # also set typeid, so that types and masses are consistent as LAMMPS requires
    # removing this line doesn't actually do anything, though
    snap.typeid = [1,2,1]
    mass = [2,3,2]
    snap.mass = mass
    assert numpy.allclose(snap.mass, mass)
    with pytest.raises(TypeError):
        snap.mass = [1,1]

def test_charge(snap):
    assert not snap.has_charge()
    assert numpy.allclose(snap.charge, 0)
    charge = [-1,0,1]
    snap.charge = charge
    assert numpy.allclose(snap.charge, charge)
    with pytest.raises(TypeError):
        snap.typeid = [0,0]
    