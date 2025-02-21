============
Installation
============

The easiest way to get ``lammpsio`` is from PyPI using ``pip``:

.. code:: bash

    pip install lammpsio


Building from source
====================

If you want to build from source, you can also use ``pip``:

.. code:: bash

    pip install .

If you are developing new code, include the ``-e`` option for an editable build.
You should then also install the developer tools:

.. code:: bash

    pip install -r requirements-dev.txt -r doc/requirements.txt

A suite of unit tests is provided with the source code and can be run with
``pytest``:

.. code:: bash

    python -m pytest

You can build the documentation from source with:

.. code:: bash

    cd doc
    make html
