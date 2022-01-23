sigpyproc
=========

**sigpyproc** is a FRB/pulsar data analysis library for Python. It provides an
OOP approach to pulsar data handling through the use of objects representing
different data types (e.g.
`SIGPROC filterbank <http://sigproc.sourceforge.net>`_,
`PSRFITS <https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html>`_,
time series, fourier series, etc.).

As pulsar data processing is often time critical, speed is maintained through
the use of compiled C++ code that are accessed via the excellent
`pybind11 <https://pybind11.readthedocs.io/>`_ library.
Additional performance increases are obtained via the use of multi-threading
with OpenMP, a threading library standard to most linux and mac systems.

.. image:: https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg
  :target: https://github.com/FRBs/sigpyproc3/actions
.. image:: https://readthedocs.org/projects/sigpyproc3/badge/?version=latest
  :target: https://sigpyproc3.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://codecov.io/gh/FRBs/sigpyproc3/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/FRBs/sigpyproc3
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black


.. toctree::
  :maxdepth: 2
  :caption: Documentation

  modules
  dev

.. toctree::
  :maxdepth: 2
  :caption: Tutorials

  tutorials/quickstart.ipynb
