import numpy
import sybil
import sybil.parsers.rest

import lammpsio


def setup_sybil_tests(namespace):
    """Sybil setup function."""
    # Common imports.
    namespace["numpy"] = numpy
    namespace["lammpsio"] = lammpsio


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
