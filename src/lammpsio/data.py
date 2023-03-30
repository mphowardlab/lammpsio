import numpy

from .box import Box
from .snapshot import Snapshot


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

    known_headers = ("atoms", "atom types", "xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz")
    unknown_headers = (
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
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
        # extract number of types
        num_types = numpy.amax(numpy.unique(snapshot.typeid))

        # extract mass by type
        if snapshot.has_mass():
            masses = numpy.empty(num_types)
            for i in range(num_types):
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
            f.write(
                f"LAMMPS {filename}\n\n"
                f"{snapshot.N} atoms\n"
                f"{num_types} atom types\n"
                f"{snapshot.box.low[0]} {snapshot.box.high[0]} xlo xhi\n"
                f"{snapshot.box.low[1]} {snapshot.box.high[1]} ylo yhi\n"
                f"{snapshot.box.low[2]} {snapshot.box.high[2]} zlo zhi\n"
            )
            if snapshot.box.tilt is not None:
                f.write("{} {} {} xy xz yz\n".format(*snapshot.box.tilt))

            # Atoms section
            # determine style if it is not given
            if atom_style is None:
                if snapshot.has_charge() and snapshot.has_molecule():
                    style = "full"
                elif snapshot.has_charge():
                    style = "charge"
                elif snapshot.has_molecule():
                    style = "molecular"
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
            box_bounds = [None, None, None, None, None, None]
            box_tilt = None
            num_types = None
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
                elif "atom types" in line:
                    num_types = int(line.split()[0])
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
            snap = Snapshot(N, box)
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
                else:
                    # silently ignore unknown sections / lines
                    pass

                line = _readline(f)

            # set mass on particles at end, in case sections were out of order in file
            if masses is not None:
                for typeid, m in masses.items():
                    snap.mass[snap.typeid == typeid] = m

        return snap
