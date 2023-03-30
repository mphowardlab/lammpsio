import numpy
import pytest


def test_orthorhombic(orthorhombic):
    box = orthorhombic
    assert numpy.allclose(box.low, [-5, -10, 0])
    assert numpy.allclose(box.high, [1, 10, 8])
    assert box.tilt is None

    box.low = [0, 0, 0]
    assert numpy.allclose(box.low, [0, 0, 0])
    with pytest.raises(TypeError):
        box.low = [0, 0]

    box.high = [6, 20, 8]
    assert numpy.allclose(box.high, [6, 20, 8])
    with pytest.raises(TypeError):
        box.high = [0, 0]


def test_triclinic(triclinic):
    box = triclinic
    assert numpy.allclose(box.low, [-5, -10, 0])
    assert numpy.allclose(box.high, [1, 10, 8])
    assert numpy.allclose(box.tilt, [1.0, -2.0, 0.5])

    box.low = [0, 0, 0]
    assert numpy.allclose(box.low, [0, 0, 0])
    with pytest.raises(TypeError):
        box.low = [0, 0]

    box.high = [6, 20, 8]
    assert numpy.allclose(box.high, [6, 20, 8])
    with pytest.raises(TypeError):
        box.high = [0, 0]

    box.tilt = [0, 0, 0]
    assert numpy.allclose(box.tilt, [0, 0, 0])
    with pytest.raises(TypeError):
        box.tilt = [0, 0]
