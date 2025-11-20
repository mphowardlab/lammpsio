Tutorials
---------

.. toctree::
    :hidden:

    ./tutorials/dimer_lattice_tutorial
    ./tutorials/analysis_tutorial/analysis_tutorial
    ./tutorials/numba_tutorial/numba_tutorial
    ./tutorials/ethanol-water-tutorial/ethanol_water_mixture

.. grid:: 2

    .. grid-item-card:: Packing a Dimer on a Cubic Lattice
        :link: ./tutorials/dimer_lattice_tutorial
        :link-type: doc

        Create a LAMMPS data file and also convert it to HOOMD-blue's GSD
        format.

    .. grid-item-card:: Analyzing a Lennard-Jones Fluid
        :link: ./tutorials/analysis_tutorial/analysis_tutorial
        :link-type: doc

        Read a LAMMPS dump file for a Lennard-Jones fluid and analyze it using
        ``freud``.

    .. grid-item-card:: Initializing an Ethanol-Water Mixture
        :link: ./tutorials/ethanol-water-tutorial/ethanol_water_mixture
        :link-type: doc

        Create a LAMMPS data file for an atomistic simulation of ethanol and
        water using PACKMOL and LAMMPS molecule templates.

    .. grid-item-card:: Speeding Up Analysis
        :link: ./tutorials/numba_tutorial/numba_tutorial
        :link-type: doc

        Accelerate the analysis of a LAMMPS dump file for a linear polymer using
        NumPy and Numba.
