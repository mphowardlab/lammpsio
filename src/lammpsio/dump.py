import gzip
import os
import pathlib

import numpy

from . import _compatibility
from .box import Box
from .data import _readline
from .snapshot import Snapshot

if _compatibility.pyzstd_version is not None:
    import pyzstd


class DumpFile:
    """LAMMPS dump file.

    The dump file is a flexible file format, so a ``schema`` can be given
    to parse the atom data. The ``schema`` is given as a dictionary of column
    indexes. Valid keys for the schema match the names and shapes in the `Snapshot`.
    The keys requiring only 1 column index are: ``id``, ``typeid``, ``molecule``,
    ``charge``, and ``mass``. The keys requiring 3 column indexes are ``position``,
    ``velocity``, and ``image``. If a ``schema`` is not specified, it will be deduced
    from the ``ITEM: ATOMS`` header.

    The vector-valued fields (``position``, ``velocity``, ``image``) must contain all
    three elements.

    Parameters
    ----------
    filename : str
        Path to dump file.
    schema : dict
        Schema for the contents of the file. Defaults to ``None``, which means to read
        it from the file.
    sort_ids : bool
        If true, sort the particles by ID in each snapshot.
    copy_from : :class:`Snapshot`
        If specified, copy fields that are missing in the dump file but are set in
        a reference :class:`Snapshot`. The fields that can be copied are ``typeid``,
        ``molecule``, ``charge``, and ``mass``.

    """

    def __init__(self, filename, schema=None, sort_ids=True, copy_from=None):
        self.filename = filename
        self.schema = schema
        self._frames = None
        self.sort_ids = sort_ids
        self.copy_from = copy_from

    @classmethod
    def create(cls, filename, schema, snapshots):
        """Create a LAMMPS dump file.

        Parameters
        ----------
        filename : str
            Path to dump file.
        schema : dict
            Schema for the contents of the file.
        snapshots : :class:`Snapshot` or list
            One or more snapshots to write to the dump file.

        Returns
        -------
        :class:`DumpFile`
            The object representing the new dump file.

        """
        # map out the schema into a dump row
        # each entry is a tuple: (column, (attribute, index))
        # the index is None for scalars, otherwise it is the component of the vector
        dump_row = []
        for k, v in schema.items():
            try:
                cols = iter(v)
                for i, col in enumerate(cols):
                    dump_row.append((col, (k, i)))
            except TypeError:
                dump_row.append((v, (k, None)))
        dump_row.sort(key=lambda x: x[0])

        # make snapshots iterable
        try:
            snapshots = iter(snapshots)
        except TypeError:
            snapshots = [snapshots]

        with open(filename, "w") as f:
            for snap in snapshots:
                f.write("ITEM: TIMESTEP\n" f"{snap.step}\n")

                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{snap.N}\n")

                # always assume periodic in all directions
                if snap.box.tilt is not None:
                    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                    xy, xz, yz = snap.box.tilt
                    lo = [
                        snap.box.low[0] + min([0.0, xy, xz, xy + xz]),
                        snap.box.low[1] + min([0.0, yz]),
                        snap.box.low[2],
                    ]
                    hi = [
                        snap.box.high[0] + max([0.0, xy, xz, xy + xz]),
                        snap.box.high[1] + max([0.0, yz]),
                        snap.box.high[2],
                    ]
                    for i in range(3):
                        f.write(f"{lo[i]:f} {hi[i]:f} {snap.box.tilt[i]:f}\n")
                else:
                    f.write("ITEM: BOX BOUNDS pp pp pp\n")
                    lo = snap.box.low
                    hi = snap.box.high
                    for i in range(3):
                        f.write(f"{lo[i]:f} {hi[i]:f}\n")

                # mapping from lammpsio to LAMMPS dump keys
                lammps_fields = {
                    "id": "id",
                    "molecule": "mol",
                    "typeid": "type",
                    "mass": "mass",
                    "position": ("x", "y", "z"),
                    "image": ("ix", "iy", "iz"),
                    "velocity": ("vx", "vy", "vz"),
                    "charge": "q",
                }
                schema_header = []
                for _, (key, key_idx) in dump_row:
                    field = lammps_fields[key]
                    if key_idx is not None:
                        field = field[key_idx]
                    schema_header.append(field)
                schema_header = " ".join(schema_header)

                f.write("ITEM: ATOMS " + schema_header + "\n")
                for i in range(snap.N):
                    line = ""
                    for col, (key, key_idx) in dump_row:
                        if key == "id":
                            val = snap.id[i] if snap.has_id() else i + 1
                        else:
                            val = getattr(snap, key)[i]
                            if key_idx is not None:
                                val = val[key_idx]

                        if key in ("id", "typeid", "molecule", "image"):
                            fmt = "{:d}"
                        elif key in ("position", "velocity"):
                            fmt = "{:.8f}"
                        else:
                            fmt = "{:f}"

                        if col > 0:
                            fmt = " " + fmt
                        line += fmt.format(val)
                    line = line.strip()
                    f.write(line + "\n")

        filename_path = pathlib.Path(filename)
        compression = cls._compression_from_suffix(filename_path.suffix)
        if compression:
            tmp = pathlib.Path(filename).with_suffix(filename_path.suffix + ".tmp")
            with open(filename, "rb") as src, compression.open(tmp, "wb") as dest:
                dest.writelines(src)
            os.replace(tmp, filename)

        return DumpFile(filename, schema)

    @staticmethod
    def _compression_from_suffix(suffix):
        if suffix == ".gz":
            return gzip
        elif suffix == ".zst":
            if _compatibility.pyzstd_version is None:
                raise ModuleNotFoundError("pyzstd needed for zstd compression")
            return pyzstd
        else:
            return None

    @property
    def copy_from(self):
        return self._copy_from

    @copy_from.setter
    def copy_from(self, value):
        if value is not None and not isinstance(value, Snapshot):
            raise TypeError("Dump file can only copy from Snapshot")
        self._copy_from = value

    @property
    def filename(self):
        """str: Path to the file."""
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        self._compression = self._compression_from_suffix(pathlib.Path(value).suffix)

        # configure section labels of dump file
        self._section = {
            "step": "ITEM: TIMESTEP",
            "natoms": "ITEM: NUMBER OF ATOMS",
            "box": "ITEM: BOX BOUNDS",
            "atoms": "ITEM: ATOMS",
        }
        if self._compression:
            for key, val in self._section.items():
                self._section[key] = val.encode()

    @property
    def schema(self):
        """dict: Data schema."""
        return self._schema

    @schema.setter
    def schema(self, value):
        if value is not None:
            # validate schema
            if "position" in value and len(value["position"]) != 3:
                raise ValueError("Position must be a 3-tuple")
            if "velocity" in value and len(value["velocity"]) != 3:
                raise ValueError("Velocity must be a 3-tuple")
            if "image" in value and len(value["image"]) != 3:
                raise ValueError("Image must be a 3-tuple")
        self._schema = value

    def _open(self):
        """Open the file handle for reading."""
        if self._compression:
            f = self._compression.open(self.filename, "rb")
        else:
            f = open(self.filename, "r")
        return f

    def _find_frames(self):
        """Seek line numbers for each frame."""
        self._frames = []
        with self._open() as f:
            line = _readline(f)
            line_num = 0
            while len(line) > 0:
                if self._section["step"] in line:
                    self._frames.append(line_num)
                line = _readline(f)
                line_num += 1

    def __len__(self):
        if self._frames is None:
            self._find_frames()
        return len(self._frames)

    def __iter__(self):
        with self._open() as f:
            state = 0
            line = _readline(f)
            while len(line) > 0:
                # timestep line first
                if state == 0 and self._section["step"] in line:
                    state += 1
                    step = int(_readline(f, True))

                # number of particles second
                if state == 1 and self._section["natoms"] in line:
                    state += 1
                    N = int(_readline(f, True))

                # box size third
                if state == 2 and self._section["box"] in line:
                    state += 1
                    box_header = line.split()
                    # check for triclinic
                    if len(box_header) == 9:
                        is_triclinic = True
                    elif len(box_header) == 6:
                        is_triclinic = False
                    else:
                        raise IOError("Incorrectly formed box bound header")
                    box_ = [
                        [float(v) for v in _readline(f, True).split()]
                        for line_ in range(3)
                    ]
                    x_lo, x_hi = box_[0][:2]
                    y_lo, y_hi = box_[1][:2]
                    z_lo, z_hi = box_[2][:2]
                    if is_triclinic:
                        xy, xz, yz = [row[2] for row in box_]
                        lo = [
                            x_lo - min([0.0, xy, xz, xy + xz]),
                            y_lo - min([0.0, yz]),
                            z_lo,
                        ]
                        hi = [
                            x_hi - max([0.0, xy, xz, xy + xz]),
                            y_hi - max([0.0, yz]),
                            z_hi,
                        ]
                        box = Box(lo, hi, [xy, xz, yz])
                    else:
                        box = Box([x_lo, y_lo, z_lo], [x_hi, y_hi, z_hi])

                # atoms come fourth
                if state == 3 and self._section["atoms"] in line:
                    state += 1

                    # extract the schema
                    if self.schema is None:
                        # mapping from LAMMPS dump keys to lammpsio
                        lammps_fields = {
                            "id": ("id", None),
                            "mol": ("molecule", None),
                            "type": ("typeid", None),
                            "mass": ("mass", None),
                            "x": ("position", 0),
                            "y": ("position", 1),
                            "z": ("position", 2),
                            "ix": ("image", 0),
                            "iy": ("image", 1),
                            "iz": ("image", 2),
                            "vx": ("velocity", 0),
                            "vy": ("velocity", 1),
                            "vz": ("velocity", 2),
                            "q": ("charge", None),
                        }
                        schema = {}
                        schema_header = line.split()[2:]
                        for i, field in enumerate(schema_header):
                            if self._compression:
                                field = field.decode()
                            if field in lammps_fields:
                                key, key_idx = lammps_fields[field]
                                if key_idx is not None:
                                    if key not in schema:
                                        schema[key] = [None, None, None]
                                    schema[key][key_idx] = i
                                else:
                                    schema[key] = i
                        # validate tuple
                        for key in ("position", "velocity", "image"):
                            if key in schema and any(x is None for x in schema[key]):
                                raise IOError("lammpsio requires 3-element vectors")
                        self.schema = schema

                    snap = Snapshot(N, box, step)
                    for i in range(snap.N):
                        atom = _readline(f, True)
                        atom = atom.split()

                        if "id" in self.schema:
                            id_ = int(atom[self.schema["id"]])
                            if id_ != i + 1:
                                snap.id[i] = id_
                        if "position" in self.schema:
                            snap.position[i] = [
                                float(atom[j]) for j in self.schema["position"]
                            ]
                        if "velocity" in self.schema:
                            snap.velocity[i] = [
                                float(atom[j]) for j in self.schema["velocity"]
                            ]
                        if "image" in self.schema:
                            snap.image[i] = [int(atom[j]) for j in self.schema["image"]]
                        if "molecule" in self.schema:
                            snap.molecule[i] = int(atom[self.schema["molecule"]])
                        if "typeid" in self.schema:
                            snap.typeid[i] = int(atom[self.schema["typeid"]])
                        if "charge" in self.schema:
                            snap.charge[i] = float(atom[self.schema["charge"]])
                        if "mass" in self.schema:
                            snap.mass[i] = float(atom[self.schema["mass"]])

                # final processing stage for the frame
                if state == 4:
                    # optionally sort the particles by ID
                    if self.sort_ids and snap.has_id():
                        snap.reorder(numpy.argsort(snap.id), check_order=False)

                    # optionally copy reference data by ID / index
                    if self._copy_from is not None:
                        if snap.N != self._copy_from.N:
                            raise ValueError(
                                "Cannot copy from a Snapshot with a different size"
                            )

                        if self._copy_from.has_id():
                            copy_id_map = {
                                id_: i for i, id_ in enumerate(self._copy_from.id)
                            }
                        else:
                            copy_id_map = {i + 1: i for i in range(self._copy_from.N)}

                        if snap.has_id():
                            copy_id = [copy_id_map[id_] for id_ in snap.id]
                        else:
                            copy_id = [copy_id_map[id_] for id_ in range(1, snap.N + 1)]

                        if not snap.has_typeid() and self._copy_from.has_typeid():
                            snap.typeid = self._copy_from.typeid[copy_id]
                        if not snap.has_molecule() and self._copy_from.has_molecule():
                            snap.molecule = self._copy_from.molecule[copy_id]
                        if not snap.has_charge() and self._copy_from.has_charge():
                            snap.charge = self._copy_from.charge[copy_id]
                        if not snap.has_mass() and self._copy_from.has_mass():
                            snap.mass = self._copy_from.mass[copy_id]
                        if self._copy_from.has_bonds():
                            snap.bonds = self._copy_from.bonds
                        if self._copy_from.has_angles():
                            snap.angles = self._copy_from.angles
                        if self._copy_from.has_dihedrals():
                            snap.dihedrals = self._copy_from.dihedrals
                        if self._copy_from.has_impropers():
                            snap.impropers = self._copy_from.impropers

                    yield snap
                    del snap, N, box, step
                    state = 0

                line = _readline(f)
