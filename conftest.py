import numpy

try:
    import sybil
    import sybil.parsers.rest

except ImportError:
    sybil = None

import lammpsio

try:
    import gsd.hoomd

    has_gsd = True
except ModuleNotFoundError:
    has_gsd = False

try:
    import lammps

    has_lammps = True
except ImportError:
    has_lammps = False


def setup_sybil_tests(namespace):
    """Sybil setup function."""
    # Common imports.
    namespace["numpy"] = numpy
    namespace["lammpsio"] = lammpsio

    if has_lammps:
        namespace["lammps"] = lammps
    else:
        namespace["lammps"] = None

    if has_gsd:
        namespace["gsd"] = gsd
        namespace["gsd.hoomd"] = gsd.hoomd
    else:
        namespace["gsd.hoomd"] = None


if sybil is not None:
    pytest_collect_file = sybil.Sybil(
        parsers=[
            sybil.parsers.rest.PythonCodeBlockParser(),
            sybil.parsers.rest.SkipParser(),
        ],
        pattern="*.py",
        setup=setup_sybil_tests,
        fixtures=["tmp_path"],
    ).pytest()
