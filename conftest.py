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


def setup_sybil_tests(namespace):
    """Sybil setup function."""
    # Common imports.
    namespace["numpy"] = numpy
    namespace["lammpsio"] = lammpsio
    if has_gsd:
        namespace["frame"] = gsd.hoomd.Frame()
    else:
        namespace["frame"] = 0


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
