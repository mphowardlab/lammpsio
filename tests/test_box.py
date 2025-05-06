import tempfile

import numpy
import pytest

import lammpsio

try:
    import lammps

    has_lammps = True
except ImportError:
    has_lammps = False


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
    assert numpy.allclose(box.high, [1, 10, 8])
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


@pytest.mark.skipif(not has_lammps, reason="lammps not installed")
def test_orthorhombic_lammps(orthorhombic):
    _tmp = tempfile.TemporaryDirectory()
    directory = _tmp.name
    filename = directory + "/orthorhombic_box.data"

    box = orthorhombic
    lmp = lammps.lammps(cmdargs=["-log", f"{directory}/log.lammps"])
    cmds = ["units lj", "dimension 3", "boundary p p p", "atom_style atomic"]

    cmds += [
        f"region box block {box.low[0]} {box.high[0]} "
        f"{box.low[1]} {box.high[1]} "
        f"{box.low[2]} {box.high[2]}"
    ]
    cmds += ["create_box 1 box"]
    cmds += ["create_atoms 1 single 0.0 0.0 0.0"]
    cmds += ["mass 1 1.0"]
    cmds += [f"write_data {filename}"]
    lmp.commands_list(cmds)
    lmp.close()

    # Read the data file back in
    data_file = lammpsio.DataFile(filename).read()
    box_read = data_file.box
    assert numpy.allclose(box_read.low, box.low)
    assert numpy.allclose(box_read.high, box.high)
    _tmp.cleanup()


@pytest.mark.skipif(not has_lammps, reason="lammps not installed")
def test_triclinic_lammps(triclinic):
    _tmp = tempfile.TemporaryDirectory()
    directory = _tmp.name
    filename = directory + "/triclinic_box.data"

    box = triclinic
    lmp = lammps.lammps(cmdargs=["-log", f"{directory}/log.lammps"])
    cmds = ["units lj", "dimension 3", "boundary p p p", "atom_style atomic"]

    cmds += [
        f"region box prism {box.low[0]} {box.high[0]} "
        f"{box.low[1]} {box.high[1]} "
        f"{box.low[2]} {box.high[2]} "
        f"{box.tilt[0]} {box.tilt[1]} {box.tilt[2]}"
    ]
    cmds += ["create_box 1 box"]
    cmds += ["create_atoms 1 single 0.0 0.0 0.0"]
    cmds += ["mass 1 1.0"]
    cmds += [f"write_data {filename}"]
    lmp.commands_list(cmds)
    lmp.close()

    # Read the data file back in
    data_file = lammpsio.DataFile(filename).read()
    box_read = data_file.box
    assert numpy.allclose(box_read.low, box.low)
    assert numpy.allclose(box_read.high, box.high)
    assert numpy.allclose(box_read.tilt, box.tilt)
    _tmp.cleanup()
