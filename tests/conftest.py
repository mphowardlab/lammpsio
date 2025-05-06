import pytest
from pytest_lazy_fixtures import lf

import lammpsio


@pytest.fixture
def orthorhombic():
    return lammpsio.Box([-5.0, -10.0, -1.0], [2.0, 10.0, 8.0])


@pytest.fixture
def triclinic():
    return lammpsio.Box([-5.0, -10.0, -1.0], [1.0, 10.0, 8.0], [2, -2.0, 0.5])


@pytest.fixture(params=[lf("orthorhombic"), lf("triclinic")])
def snap(request):
    return lammpsio.Snapshot(3, request.param, 10)


@pytest.fixture(params=[lf("orthorhombic"), lf("triclinic")])
def snap_8(request):
    return lammpsio.Snapshot(8, request.param, 10)
