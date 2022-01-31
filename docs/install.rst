.. _install:

Installation
============

.. note:: ``sigpyproc`` requires Python 3.8 and later.

Using pip
---------

The recommended method of installing *sigpyproc* is with `pip
<https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U git+https://github.com/FRBs/sigpyproc3


Some of the required dependencies are:

- `numpy <https://numpy.org>`_
- `astropy <https://www.astropy.org>`_
- `numba <https://numba.pydata.org>`_
- `bottleneck <https://bottleneck.readthedocs.io>`_
- `attrs <https://attrs.org>`_
- `rich <https://rich.readthedocs.io>`_


.. _source:

From Source
-----------

The source code for *sigpyproc* can be downloaded and installed `from GitHub
<https://github.com/FRBs/sigpyproc3>`_ by running

.. code-block:: bash

    git clone https://github.com/FRBs/sigpyproc3.git
    cd sigpyproc3
    python -m pip install -e .


Testing
-------

To run the unit tests with `pytest <https://docs.pytest.org>`_,
first install the testing dependencies using pip:

.. code-block:: bash

    python -m pip install -e ".[tests]"

and then execute:

.. code-block:: bash

    python -m pytest -v tests

Normally, all of the tests should pass.