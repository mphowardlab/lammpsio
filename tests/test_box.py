import numpy
import pytest
from pytest_lazy_fixtures import lf

import lammpsio


def test_orthorhombic(orthorhombic):
    box = orthorhombic
    assert numpy.allclose(box.low, [-5, -10, -1])
    assert numpy.allclose(box.high, [2, 10, 8])
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
    assert numpy.allclose(box.low, [-5, -10, -1])
    assert numpy.allclose(box.high, [2, 10, 8])
    assert numpy.allclose(box.tilt, [2.0, -2.0, 0.5])

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


@pytest.mark.parametrize("box", [lf("orthorhombic"), lf("triclinic")])
@pytest.mark.parametrize("force_triclinic", [True, False])
def test_from_matrix(box, force_triclinic):
    lx, ly, lz = box.high - box.low
    xy, xz, yz = box.tilt if box.tilt is not None else (0, 0, 0)
    matrix = numpy.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    new_box = lammpsio.Box.from_matrix(box.low, matrix, force_triclinic=force_triclinic)

    assert numpy.allclose(new_box.low, box.low)
    assert numpy.allclose(new_box.high, box.high)
    if force_triclinic:
        assert new_box.tilt is not None
        if box.tilt is not None:
            assert numpy.allclose(new_box.tilt, box.tilt)
        else:
            assert numpy.allclose(new_box.tilt, [0, 0, 0])
    else:
        if box.tilt is not None:
            assert numpy.allclose(new_box.tilt, box.tilt)
        else:
            assert new_box.tilt is None

    # test with invalid low
    with pytest.raises(TypeError):
        lammpsio.Box.from_matrix([0, 0], matrix)

    # test with invalid matrix shape
    invalid_matrix_shape = numpy.array([[lx, xy], [ly, yz]])
    with pytest.raises(TypeError):
        lammpsio.Box.from_matrix(box.low, invalid_matrix_shape)

    # test with invalid matrix values
    invalid_matrix = numpy.array([[lx, xy, xz], [ly, 0, yz], [lz, 0, 0]])
    with pytest.raises(ValueError):
        lammpsio.Box.from_matrix(box.low, invalid_matrix)


@pytest.mark.parametrize("box", [lf("orthorhombic"), lf("triclinic")])
def test_to_matrix(box):
    low, matrix = box.to_matrix()

    lx, ly, lz = box.high - box.low
    xy, xz, yz = box.tilt if box.tilt is not None else (0, 0, 0)
    expected_matrix = numpy.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    assert numpy.allclose(matrix, expected_matrix)
    assert numpy.allclose(low, box.low)


@pytest.mark.parametrize("box", [lf("orthorhombic"), lf("triclinic")])
def test_convert_convention(box):
    # Convert to HOOMD-blue convention
    hoomd_box = box.to_hoomd_convention()
    assert isinstance(hoomd_box, numpy.ndarray)
    assert hoomd_box.shape == (6,)
    assert numpy.allclose(hoomd_box[:3], box.high - box.low)
    if box.tilt is not None:
        assert numpy.allclose(
            hoomd_box[3:],
            [
                box.tilt[0] / (box.high[1] - box.low[1]),
                box.tilt[1] / (box.high[2] - box.low[2]),
                box.tilt[2] / (box.high[2] - box.low[2]),
            ],
        )
    else:
        assert numpy.allclose(hoomd_box[3:], [0, 0, 0])

    # Convert back from HOOMD-blue convention
    new_box = lammpsio.Box.from_hoomd_convention(hoomd_box, low=box.low)
    assert numpy.allclose(new_box.low, box.low)
    assert numpy.allclose(new_box.high, box.high)
    if box.tilt is not None:
        assert numpy.allclose(new_box.tilt, box.tilt)
    else:
        assert new_box.tilt is None

    # test two dimension box
    hoomd_box_2d = box.to_hoomd_convention()
    # set Lz to zero for 2D
    hoomd_box_2d[2] = 0

    lammps_box_2d = lammpsio.Box.from_hoomd_convention(hoomd_box_2d)
    # check that Lz is set to 1.0 and xz & yz are 0
    assert numpy.allclose(lammps_box_2d.high[2] - lammps_box_2d.low[2], 1.0)
    if box.tilt is not None:
        assert numpy.allclose(lammps_box_2d.tilt, [box.tilt[0], 0, 0])
    else:
        assert lammps_box_2d.tilt is None
