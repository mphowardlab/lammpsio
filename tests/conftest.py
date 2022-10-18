import pytest

import lmptools

# @pytest.fixture(scope="session")
# def data_file()

@pytest.fixture
def orthorhombic():
    return lmptools.Box([-5.,-10.,0.],[1.,10.,8.])

@pytest.fixture
def triclinic():
    return lmptools.Box([-5.,-10.,0.],[1.,10.,8.],[1.0,-2.0,0.5])
