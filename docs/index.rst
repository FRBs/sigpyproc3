sigpyproc
=========

**sigpyproc** is a FRB/pulsar data analysis library for Python. It provides an
OOP approach to pulsar data handling through the use of objects representing
different data types (e.g.
`SIGPROC filterbank <http://sigproc.sourceforge.net>`_,
`PSRFITS <https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html>`_,
time series, fourier series, etc.).

As pulsar data processing is often time critical, speed is maintained using
the excellent `numba <https://numba.pydata.org/>`_ library.

`sigpyproc` is intended to be an Python alternative for the
`SIGPROC filterbank <http://sigproc.sourceforge.net>`_ pulsar signal processing toolbox.
Over time it has also developed and become an independent project in its own right.
Unlike `SIGPROC <http://sigproc.sourceforge.net>`_ and `PRESTO <https://github.com/scottransom/presto>`_,
`sigpyproc` does not currently have full capabilities as a piece of FRB/pulsar searching software.
Instead, `sigpyproc` provides data manipulation routines which are well suited to preprocessing
and micro-management of pulsar data. The structure of the package also makes it an ideal development
environment, with a simple plug-and-play system with new modules and extensions.

.. image:: https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg
  :target: https://github.com/FRBs/sigpyproc3/actions
.. image:: https://readthedocs.org/projects/sigpyproc3/badge/?version=latest
  :target: https://sigpyproc3.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://codecov.io/gh/FRBs/sigpyproc3/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/FRBs/sigpyproc3
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black


Contents
--------

.. toctree::
  :maxdepth: 1
  :caption: User Guide

  install
  modules
  dev
  changes

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  tutorials/quickstart.ipynb
