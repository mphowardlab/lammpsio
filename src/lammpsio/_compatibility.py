import numpy
import packaging.version

# GSD - optional import
try:
    import gsd
    import gsd.hoomd

    # determine how GSD stores its version
    try:
        gsd_version = gsd.version.version
    except AttributeError:
        gsd_version = gsd.__version__
    gsd_version = packaging.version.Version(gsd_version)

    # GSD >= 2.8.0 deprecated Snapshot in favor of Frame
    if gsd_version >= packaging.version.Version("2.8.0"):
        gsd_frame_class = gsd.hoomd.Frame
    else:
        gsd_frame_class = gsd.hoomd.Snapshot
except ModuleNotFoundError:
    gsd_version = None

# pyzstd - optional import
try:
    import pyzstd

    pyzstd_version = packaging.version.Version(pyzstd.__version__)
except ModuleNotFoundError:
    pyzstd_version = None

# numpy
# copy behavior changed in 2.0.0, see https://github.com/scipy/scipy/pull/20172
numpy_version = packaging.version.Version(numpy.__version__)
if numpy_version >= packaging.version.Version("2.0.0"):
    numpy_copy_if_needed = None
else:
    numpy_copy_if_needed = False
