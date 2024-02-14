class Topology:
    """Particle topology.

    Parameters
    ----------
    N : int
        Number of connection type in configuration.
    num_members : int
        The number of members in a connection (e.g., 2 for bonds, 3 for angles
        and 4 for dihedrals).

    """

    def __init__(self, N, num_members):
        self._N = N
        self._num_members = num_members


class Bonds(Topology):
    def __init__(self, N):
        super().__init__(N, num_members=2)
