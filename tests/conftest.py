import pytest
from pytest_lazyfixture import lazy_fixture

import lammpsio

@pytest.fixture
def orthorhombic():
    return lammpsio.Box([-5.,-10.,0.],[1.,10.,8.])

@pytest.fixture
def triclinic():
    return lammpsio.Box([-5.,-10.,0.],[1.,10.,8.],[1.0,-2.0,0.5])

@pytest.fixture(params=[
    lazy_fixture('orthorhombic'),
    lazy_fixture('triclinic')
    ])
def snap(request):
    return lammpsio.Snapshot(3, request.param, 10)
