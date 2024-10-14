import numpy

from .box import Box
from .snapshot import Snapshot
from .topology import Angles, Bonds, Dihedrals, Impropers


def _readline(file_, require=False):
    """Read and require a line."""
    line = file_.readline()
    if require and len(line) == 0:
        raise OSError("Could not read line from file")
    return line


class DataFile:
    """LAMMPS data file.

    Parameters
    ----------
    filename : str
        Path to data file.
    atom_style : str
        Atom style to use for data file. Defaults to ``None``, which means the
        style should be read from the file.

    Attributes
    ----------
    filename : str
        Path to the file.
    atom_style : str
        Atom style for the file.
    known_headers : list
        Data file headers that can be processed.
    unknown_headers : list
        Data file headers that will be ignored.
    known_bodies : list
        Data file body sections that can be processed.
    unknown_bodies : list
        Data file body sections that will be ignored.

    """

    def __init__(self, filename, atom_style=None):
        self.filename = filename
        self.atom_style = atom_style

    known_headers = (
        "atoms",
        "atom types",
        "xlo xhi",
        "ylo yhi",
        "zlo zhi",
        "xy xz yz",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
    )
    unknown_headers = (
        "extra bond per atom",
        "extra angle per atom",
        "extra dihedral per atom",
        "extra improper per atom",
        "ellipsoids",
        "lines",
        "triangles",
        "bodies",
    )
    known_bodies = ("Atoms", "Velocities", "Masses")
    unknown_bodies = (
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
    )

    @classmethod
    def create(cls, filename, snapshot, atom_style=None):
        """Create a LAMMPS data file from a snapshot.

        Parameters
        ----------
        filename : str
            Path to data file.
        snapshot : :class:`Snapshot`
            Snapshot to write to file.
        atom_style : str
            Atom style to use for data file. Defaults to ``None``, which means the
            style should be inferred from the contents of ``snapshot``.

        Returns
        -------
        :class:`DataFile`
            The object representing the new data file.

        Raises
        ------
        ValueError
            If all masses are not the same for a given type.

        """

        # extract mass by type
        if snapshot.has_mass():
            masses = numpy.ones(snapshot.num_types)
            for i in range(snapshot.num_types):
                mi = snapshot.mass[snapshot.typeid == i + 1]
                if not numpy.all(mi == mi[0]):
                    raise ValueError("All masses for a type must be equal")
                elif mi[0] <= 0.0:
                    raise ValueError("Type mass must be positive value")
                masses[i] = mi[0]
        else:
            masses = None

        with open(filename, "w") as f:
            # LAMMPS header
            f.write(f"LAMMPS {filename}\n\n" f"{snapshot.N} atoms\n")

            if snapshot.has_bonds():
                f.write(f"{snapshot.bonds.N} bonds\n")

            if snapshot.has_angles():
                f.write(f"{snapshot.angles.N} angles\n")

            if snapshot.has_dihedrals():
                f.write(f"{snapshot.dihedrals.N} dihedrals\n")

            if snapshot.has_impropers():
                f.write(f"{snapshot.impropers.N} impropers\n")

            f.write(f"{snapshot.num_types} atom types\n")

            if snapshot.has_bonds():
                f.write(f"{snapshot.bonds.num_types} bond types\n")

            if snapshot.has_angles():
                f.write(f"{snapshot.angles.num_types} angle types\n")

            if snapshot.has_dihedrals():
                f.write(f"{snapshot.dihedrals.num_types} dihedral types\n")

            if snapshot.has_impropers():
                f.write(f"{snapshot.impropers.num_types} improper types\n")

            f.write(
                f"{snapshot.box.low[0]} {snapshot.box.high[0]} xlo xhi\n"
                f"{snapshot.box.low[1]} {snapshot.box.high[1]} ylo yhi\n"
                f"{snapshot.box.low[2]} {snapshot.box.high[2]} zlo zhi\n"
            )
            if snapshot.box.tilt is not None:
                f.write("{} {} {} xy xz yz\n".format(*snapshot.box.tilt))

            # Atoms section
            # determine style if it is not given
            has_topology = (
                snapshot.has_bonds()
                or snapshot.has_angles()
                or snapshot.has_dihedrals()
                or snapshot.has_impropers()
            )
            if atom_style is None:
                if snapshot.has_molecule() or has_topology:
                    if snapshot.has_charge():
                        style = "full"
                    else:
                        style = "molecular"
                else:
                    if snapshot.has_charge():
                        style = "charge"
                    else:
                        style = "atomic"
            else:
                style = atom_style
            # set format string based on style
            if style == "full":
                style_fmt = (
                    "{atomid:d} {molid:d} {typeid:d} {q:.5f} {x:.8f} {y:.8f} {z:.8f}"
                )
            elif style == "charge":
                style_fmt = "{atomid:d} {typeid:d} {q:.5f} {x:.8f} {y:.8f} {z:.8f}"
            elif style == "molecular":
                style_fmt = "{atomid:d} {molid:d} {typeid:d} {x:.8f} {y:.8f} {z:.8f}"
            elif style == "atomic":
                style_fmt = "{atomid:d} {typeid:d} {x:.8f} {y:.8f} {z:.8f}"
            else:
                raise ValueError("Unknown atom style")
            if snapshot.has_image():
                style_fmt += " {ix:d} {iy:d} {iz:d}"
            # write section
            f.write(f"\nAtoms # {style}\n\n")
            for i in range(snapshot.N):
                style_args = dict(
                    atomid=snapshot.id[i] if snapshot.has_id() else i + 1,
                    typeid=snapshot.typeid[i],
                    x=snapshot.position[i][0],
                    y=snapshot.position[i][1],
                    z=snapshot.position[i][2],
                    q=snapshot.charge[i] if snapshot.has_charge() else 0.0,
                    molid=snapshot.molecule[i] if snapshot.has_molecule() else 0,
                )
                if snapshot.has_image():
                    style_args.update(
                        ix=snapshot.image[i][0],
                        iy=snapshot.image[i][1],
                        iz=snapshot.image[i][2],
                    )

                f.write(style_fmt.format(**style_args) + "\n")

            # Velocities section
            if snapshot.has_velocity():
                f.write("\nVelocities\n\n")
                for i in range(snapshot.N):
                    vel_fmt = "{atomid:8d}{vx:16.8f}{vy:16.8f}{vz:16.8f}\n"
                    f.write(
                        vel_fmt.format(
                            atomid=snapshot.id[i] if snapshot.has_id() else i + 1,
                            vx=snapshot.velocity[i][0],
                            vy=snapshot.velocity[i][1],
                            vz=snapshot.velocity[i][2],
                        )
                    )

            # Masses section
            if masses is not None:
                f.write("\nMasses\n\n")
                for i, mi in enumerate(masses):
                    f.write("{typeid:4d}{m:12}\n".format(typeid=i + 1, m=mi))

            # Bonds section
            if snapshot.has_bonds():
                f.write("\nBonds\n\n")
                for i in range(snapshot.bonds.N):
                    f.write(
                        "{id} {typeid} {m1} {m2}\n".format(
                            id=snapshot.bonds.id[i],
                            typeid=snapshot.bonds.typeid[i],
                            m1=snapshot.bonds.members[i, 0],
                            m2=snapshot.bonds.members[i, 1],
                        )
                    )

            # Angles section
            if snapshot.has_angles():
                f.write("\nAngles\n\n")
                for i in range(snapshot.angles.N):
                    f.write(
                        "{id} {typeid} {m1} {m2} {m3}\n".format(
                            id=snapshot.angles.id[i],
                            typeid=snapshot.angles.typeid[i],
                            m1=snapshot.angles.members[i, 0],
                            m2=snapshot.angles.members[i, 1],
                            m3=snapshot.angles.members[i, 2],
                        )
                    )

            # Dihedrals section
            if snapshot.has_dihedrals():
                f.write("\nDihedrals\n\n")
                for i in range(snapshot.dihedrals.N):
                    f.write(
                        "{id} {typeid} {m1} {m2} {m3} {m4}\n".format(
                            id=snapshot.dihedrals.id[i],
                            typeid=snapshot.dihedrals.typeid[i],
                            m1=snapshot.dihedrals.members[i, 0],
                            m2=snapshot.dihedrals.members[i, 1],
                            m3=snapshot.dihedrals.members[i, 2],
                            m4=snapshot.dihedrals.members[i, 3],
                        )
                    )

            # Impropers section
            if snapshot.has_impropers():
                f.write("\nImpropers\n\n")
                for i in range(snapshot.impropers.N):
                    f.write(
                        "{id} {typeid} {m1} {m2} {m3} {m4}\n".format(
                            id=snapshot.impropers.id[i],
                            typeid=snapshot.impropers.typeid[i],
                            m1=snapshot.impropers.members[i, 0],
                            m2=snapshot.impropers.members[i, 1],
                            m3=snapshot.impropers.members[i, 2],
                            m4=snapshot.impropers.members[i, 3],
                        )
                    )
        return DataFile(filename)

    def read(self):
        """Read the file.

        Returns
        -------
        :class:`Snapshot`
            Snapshot from the data file.

        Raises
        ------
        ValueError
            If :attr:`atom_style` is set but does not match file contents.
        ValueError
            If :attr:`atom_style` is not specified and not set in file.

        """
        with open(self.filename) as f:
            # initialize snapshot from header
            N = None
            N_bonds = None
            N_angles = None
            N_dihedrals = None
            N_impropers = None
            num_types = None
            num_bond_types = None
            num_angle_types = None
            num_dihedral_types = None
            num_improper_types = None
            box_bounds = [None, None, None, None, None, None]
            box_tilt = None

            # skip first line
            _readline(f, True)
            line = _readline(f)
            while len(line) > 0:
                line = line.rstrip()

                # skip blank and go to next line
                if len(line) == 0:
                    line = _readline(f)
                    continue

                # check for unknown headers and go to next line
                skip_line = False
                for h in self.unknown_headers:
                    if h in line:
                        skip_line = True
                        break
                if skip_line:
                    line = _readline(f)
                    continue

                # line is not empty but it is not a header, so break and try to
                # make snapshot keep the line so that it can be processed in
                # next step
                done_header = True
                for h in self.known_headers:
                    if h in line:
                        done_header = False
                        break
                if done_header:
                    break

                # process useful header info
                if "atoms" in line:
                    N = int(line.split()[0])
                elif "bonds" in line:
                    N_bonds = int(line.split()[0])
                elif "angles" in line:
                    N_angles = int(line.split()[0])
                elif "dihedrals" in line:
                    N_dihedrals = int(line.split()[0])
                elif "impropers" in line:
                    N_impropers = int(line.split()[0])
                elif "atom types" in line:
                    num_types = int(line.split()[0])
                elif "bond types" in line:
                    num_bond_types = int(line.split()[0])
                elif "angle types" in line:
                    num_angle_types = int(line.split()[0])
                elif "dihedral types" in line:
                    num_dihedral_types = int(line.split()[0])
                elif "improper types" in line:
                    num_improper_types = int(line.split()[0])
                elif "xlo xhi" in line:
                    box_bounds[0], box_bounds[3] = [float(x) for x in line.split()[:2]]
                elif "ylo yhi" in line:
                    box_bounds[1], box_bounds[4] = [float(x) for x in line.split()[:2]]
                elif "zlo zhi" in line:
                    box_bounds[2], box_bounds[5] = [float(x) for x in line.split()[:2]]
                elif "xy xz yz" in line:
                    box_tilt = [float(x) for x in line.split()[:3]]
                else:
                    raise RuntimeError("Uncaught header line! Check programming")

                # done here, read next line
                line = _readline(f)

            if N is None:
                raise IOError("Number of particles not read")
            elif num_types is None:
                raise IOError("Number of types not read")
            elif None in box_bounds:
                raise IOError("Box bounds not read")
            box = Box(box_bounds[:3], box_bounds[3:], box_tilt)
            snap = Snapshot(N, box, num_types=num_types)
            id_map = {}

            # now that snapshot is made, file it in with body sections
            masses = None
            while len(line) > 0:
                if "Atoms" in line:
                    # use or extract style
                    row = line.split()
                    if len(row) == 3:
                        style = row[-1]
                        if self.atom_style is not None and style != self.atom_style:
                            raise ValueError(
                                "Specified style does not match style in file"
                            )
                    else:
                        style = self.atom_style
                    if style is None:
                        raise IOError("Atom style not found, specify.")
                    # number of columns to read for style
                    if style == "full":
                        style_cols = 3
                    elif style == "charge":
                        style_cols = 2
                    elif style == "molecular":
                        style_cols = 2
                    elif style == "atomic":
                        style_cols = 1
                    else:
                        raise ValueError("Unknown atom style")

                    # read atom coordinates
                    _readline(f, True)  # blank line
                    for i in range(snap.N):
                        # strip out comments
                        row = _readline(f, True).split()
                        try:
                            comment = row.index("#")
                            row = row[:comment]
                        except ValueError:
                            pass

                        # check that row is correctly sized and process
                        if not (
                            len(row) == style_cols + 4 or len(row) == style_cols + 7
                        ):
                            raise IOError(
                                "Expected number of columns not read for atom style"
                            )

                        # only save the atom id if it is not in standard order
                        id_ = int(row[0])
                        if id_ != i + 1:
                            if id_ not in id_map:
                                id_map[id_] = i
                            idx = id_map[id_]
                            snap.id[idx] = id_
                        else:
                            idx = i

                        if style == "full":
                            snap.molecule[idx] = int(row[1])
                            snap.typeid[idx] = int(row[2])
                            snap.charge[idx] = float(row[3])
                        elif style == "charge":
                            snap.typeid[idx] = int(row[1])
                            snap.charge[idx] = float(row[2])
                        elif style == "molecular":
                            snap.molecule[idx] = int(row[1])
                            snap.typeid[idx] = int(row[2])
                        elif style == "atomic":
                            snap.typeid[idx] = int(row[1])
                        snap.position[idx] = [
                            float(x) for x in row[style_cols + 1 : style_cols + 4]
                        ]
                        if len(row) == style_cols + 7:
                            snap.image[idx] = [
                                int(x) for x in row[style_cols + 4 : style_cols + 7]
                            ]

                    # sanity check types
                    if numpy.any(
                        numpy.logical_or(snap.typeid < 1, snap.typeid > num_types)
                    ):
                        raise ValueError("Invalid type id")
                elif "Velocities" in line:
                    _readline(f, True)  # blank line
                    for i in range(snap.N):
                        row = _readline(f, True).split()
                        if len(row) < 4:
                            raise IOError(
                                "Expected number of columns not read for velocity"
                            )
                        # parse atom id: need to repeat mapping in case
                        # Velocity comes before Atoms
                        id_ = int(row[0])
                        if id_ != i + 1:
                            if id_ not in id_map:
                                id_map[id_] = i
                            idx = id_map[id_]
                            snap.id[idx] = id_
                        else:
                            idx = i
                        snap.velocity[idx] = [float(x) for x in row[1:4]]
                elif "Masses" in line:
                    masses = {}
                    _readline(f, True)  # blank line
                    for i in range(num_types):
                        row = _readline(f, True).split()
                        if len(row) < 2:
                            raise IOError(
                                "Expected number of columns not read for mass"
                            )
                        masses[int(row[0])] = float(row[1])
                elif "Bonds" in line:
                    if N_bonds is not None:
                        snap.bonds = Bonds(N_bonds, num_bond_types)
                    _readline(f, True)  # blank line
                    for i in range(snap.bonds.N):
                        row = _readline(f, True).split()
                        if len(row) < 4:
                            raise IOError(
                                "Expected number of columns not read for bonds"
                            )
                        row = [int(x) for x in row]
                        # only write ID if it is out of default order
                        id = row[0]
                        if id != i + 1:
                            snap.bonds.id[i] = id
                        snap.bonds.typeid[i] = row[1]
                        snap.bonds.members[i] = row[2:4]

                    # sanity check
                    if numpy.any(snap.bonds.num_types < 1) or numpy.any(
                        snap.bonds.typeid > snap.bonds.num_types
                    ):
                        raise ValueError("Invalid bond type id")
                elif "Angles" in line:
                    if N_angles is not None:
                        snap.angles = Angles(N_angles, num_angle_types)
                    _readline(f, True)  # blank line
                    for i in range(snap.angles.N):
                        row = _readline(f, True).split()
                        if len(row) < 5:
                            raise IOError(
                                "Expected number of columns not read for angles"
                            )
                        row = [int(x) for x in row]
                        # only write ID if it is out of default order
                        id = row[0]
                        if id != i + 1:
                            snap.angles.id[i] = id
                        snap.angles.typeid[i] = row[1]
                        snap.angles.members[i] = row[2:5]

                    # sanity check
                    if numpy.any(snap.angles.num_types < 1) or numpy.any(
                        snap.angles.typeid > snap.angles.num_types
                    ):
                        raise ValueError("Invalid angle type id")
                elif "Dihedrals" in line:
                    if N_dihedrals is not None:
                        snap.dihedrals = Dihedrals(N_dihedrals, num_dihedral_types)
                    _readline(f, True)  # blank line
                    for i in range(snap.dihedrals.N):
                        row = _readline(f, True).split()
                        if len(row) < 6:
                            raise IOError(
                                "Expected number of columns not read for dihedrals"
                            )
                        row = [int(x) for x in row]
                        # only write ID if it is out of default order
                        id = row[0]
                        if id != i + 1:
                            snap.dihedrals.id[i] = id
                        snap.dihedrals.typeid[i] = row[1]
                        snap.dihedrals.members[i] = row[2:6]

                    # sanity check
                    if numpy.any(snap.dihedrals.num_types < 1) or numpy.any(
                        snap.dihedrals.typeid > snap.dihedrals.num_types
                    ):
                        raise ValueError("Invalid dihedral type id")
                elif "Impropers" in line:
                    if N_impropers is not None:
                        snap.impropers = Impropers(N_impropers, num_improper_types)
                    _readline(f, True)  # blank line
                    for i in range(snap.impropers.N):
                        row = _readline(f, True).split()
                        if len(row) < 6:
                            raise IOError(
                                "Expected number of columns not read for impropers"
                            )
                        row = [int(x) for x in row]
                        # only write ID if it is out of default order
                        id = row[0]
                        if id != i + 1:
                            snap.impropers.id[i] = id
                        snap.impropers.typeid[i] = row[1]
                        snap.impropers.members[i] = row[2:6]

                    # sanity check
                    if numpy.any(snap.impropers.num_types < 1) or numpy.any(
                        snap.impropers.typeid > snap.impropers.num_types
                    ):
                        raise ValueError("Invalid improper type id")
                else:
                    # silently ignore unknown sections / lines
                    pass

                line = _readline(f)

            # set mass on particles at end, in case sections were out of order in file
            if masses is not None:
                for typeid, m in masses.items():
                    snap.mass[snap.typeid == typeid] = m

        return snap
