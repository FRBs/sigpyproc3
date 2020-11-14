.. _install:

Installation
============

You need Python 3.6 or later to run sigpyproc. Additionally,
`FFTW3 <http://www.fftw.org/>`_ and `OpenMP <https://www.openmp.org/>`_
should be installed/enabled on your system.

From source
-----------

sigpyproc is developed on `GitHub <https://github.com/FRBs/sigpyproc3>`_, so to
try the latest stuff, you can install via pip:

.. code-block:: bash

    pip install git+https://github.com/FRBs/sigpyproc3

or clone the source repository and install from there

.. code-block:: bash

    git clone https://github.com/FRBs/sigpyproc3.git
    cd sigpyproc3
    python -m pip install -e .


Test the installation
---------------------

You can execute some unit and benchmark tests using
`pytest <https://docs.pytest.org>`_ to make sure that the installation went
alright. In the root directory of the source code execute the following command:

.. code-block:: bash

    python -m pytest -v tests