import numpy
import packaging.version

try:
    import gsd

    try:
        gsd_version = gsd.version.version
    except AttributeError:
        gsd_version = gsd.__version__
    gsd_version = packaging.version.Version(gsd_version)
except ModuleNotFoundError:
    gsd_version = None

try:
    import pyzstd

    pyzstd_version = packaging.version.Version(pyzstd.__version__)
except ModuleNotFoundError:
    pyzstd_version = None

# https://github.com/scipy/scipy/pull/20172
numpy_version = packaging.version.Version(numpy.__version__)
if numpy_version >= packaging.version.Version("2.0.0"):
    numpy_copy_if_needed = None
else:
    numpy_copy_if_needed = False
