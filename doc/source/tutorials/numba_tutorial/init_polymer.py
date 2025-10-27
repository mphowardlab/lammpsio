import lammpsio
import numpy as np

# Create a 10 bead polymer chain
n_beads = 10

# Initialize positions for a linear chain
positions = np.zeros((n_beads, 3))
for i in range(n_beads):
    positions[i] = [(i - (n_beads - 1) / 2) * 1, 0, 0]  # Space beads 1 unit apart along x-axis

# Create atom IDs and types
atom_ids = np.arange(1, n_beads + 1)
atom_types = np.ones(n_beads, dtype=int)

# Create bonds between consecutive beads
bonds = []
for i in range(n_beads - 1):
    bonds.append([i + 1, i + 2])  # Bond between bead i and i+1
bonds = np.array(bonds)
bond_types = np.ones(len(bonds), dtype=int)

# Create the system
snapshot = lammpsio.Snapshot(N=n_beads,
                             box=lammpsio.Box([-10, -10, -10], [10, 10, 10]),
                             step=100)

snapshot.position = positions
snapshot.id = atom_ids
snapshot.typeid = atom_types
snapshot.mass = np.array([1.0] * n_beads)
# Add bonds
snapshot.bonds = lammpsio.topology.Bonds(N=len(bonds),num_types=1)
snapshot.bonds.id = np.arange(1, len(bonds)+1)
snapshot.bonds.typeid = bond_types  # lammpsio uses 0-indexed types
snapshot.bonds.members = bonds  # lammpsio uses 0-indexed atom IDs

# Write to LAMMPS data file
data = lammpsio.DataFile.create('init.data', snapshot)

