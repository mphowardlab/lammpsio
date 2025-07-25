# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2025-07-19
### Added
- Complete sphinx documentation is hosted on
  [Read the Docs](https://lammpsio.readthedocs.io).
- Unit tests that run against LAMMPS.
- Convenience methods for working with boxes: to/from a matrix and to/from a
  HOOMD box definition.

### Fixed
- Lists of known/unknown headers and bodies for `DataFile` are now current.
- Particle positions are now correctly recentered when converting a `Snapshot`
  to a GSD `Frame` if the `Box` is not centered at the origin.

## [0.7.0] - 2024-12-10
### Added
- Initial support for type labels of particle and topology data through the
  `LabelMap` object.
- Interconversion of topology data with GSD `Frame`.
- Support for Python 3.13.
- Copying of topology data from an existing `Snapshot` when reading a dump file.

### Fixed
- Only GSD properties that have been assigned are converted to a `Snapshot`.
- Issues with writing topology information to data files.
- Deduction of atom type from information in `Snapshot`.

### Removed
- Support for Python 3.8.

## [0.6.1] - 2024-06-24
### Added
- Backwards compatible support for NumPy 2.0.

### Fixed
- Test dependencies for Python 3.8.

## [0.6.0] - 2024-05-28
### Added
- Support for reading and writing dump files with zstd compression.

### Fixed
- Reading and writing dump files with triclinic boxes.
- Validation of image in user-specified dump schema.

## [0.5.0] - 2024-04-29
### Added
- Basic support for molecular topology data. These data are exposed as `Bonds`,
`Angles`, `Dihedrals` and `Impropers` objects that can be included in a
`Snapshot` and read/written to a `DataFile`. Some features are not yet fully
supported, such as conversion to/from GSD format and as an option to `copy_from`
for a `DumpFile`.
- Testing for Python 3.12.

### Changed
- Bumped license year to 2024.

## [0.4.1] - 2023-07-20
### Added
- `lammpsio` is available for download on conda-forge. Installation directions
have been updated to include this option.

### Fixed
- Compatibility with GSD 3.

### Changed
- Bumped license year to 2023.

## [0.4.0] - 2023-03-31
### Added
- `Snapshot` can be created from and converted to a GSD HOOMD frame.
- Package version is embedded in `__version__`.

### Changed
- Python 3.11 is supported and tested.
- Code style is enforced using `black` and `flake8`. Developers should install
`requirements-dev.txt` and configure `pre-commit`.
- Classes are broken into modules for readability. The user API does not change.
- NumPy arrays use `float` and `int` as data types instead of specified precision.

## [0.3.0] - 2022-11-06
### Added
- Dump file defaults to reading schema from atoms header.

### Fixed
- Dump file now reads/writes box bounds header correctly.

## [0.2.0] - 2022-10-26
### Added
- Snapshots store particle IDs. Files can contain noncompact particle ID ranges.
- Dump file can copy fields from another snapshot during reading.

## [0.1.1] - 2022-10-21
### Fixed
- Typos in the README documentation.

## [0.1.0] - 2022-10-20
### Added
- Initial official release of all tools.
- Packaging support for PyPI.
- Unit tests for all code.
- Use GitHub Actions for testing and publishing.
- Create changelog and code of conduct files.

### Changed
- The package has been renamed `lammpsio` for consistency with PyPI.

[Unreleased]: https://github.com/mphowardlab/lammpsio/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/mphowardlab/lammpsio/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/mphowardlab/lammpsio/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/mphowardlab/lammpsio/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/mphowardlab/lammpsio/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/mphowardlab/lammpsio/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/mphowardlab/lammpsio/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/mphowardlab/lammpsio/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mphowardlab/lammpsio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mphowardlab/lammpsio/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mphowardlab/lammpsio/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mphowardlab/lammpsio/releases/tag/v0.1.0
