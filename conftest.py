import numpy

try:
    import sybil
    import sybil.parsers.rest

except ImportError:
    sybil = None

import lammpsio


def setup_sybil_tests(namespace):
    """Sybil setup function."""
    # Common imports.
    namespace["numpy"] = numpy
    namespace["lammpsio"] = lammpsio

    try:
        import gsd.hoomd
    except ImportError:
        gsd.hoomd = None


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
