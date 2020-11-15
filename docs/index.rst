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


Basic Usage
-----------

.. code-block:: python

  from sigpyproc.Readers import FilReader
  myFil = FilReader("tutorial.fil")

  from sigpyproc.Readers import FitsReader
  myFits = FitsReader("tutorial.fits")


Why sigpyproc?
--------------
**sigpyproc** was initially intended to be an Python wrapper for the
`SIGPROC <http://sigproc.sourceforge.net>`_ pulsar signal processing toolbox,
but over time it has developed and become an independent project in its own right.
Unlike `SIGPROC <http://sigproc.sourceforge.net>`_ and
`PRESTO <https://github.com/scottransom/presto>`_, **sigpyproc** does not currently
have full capabilities as a piece of FRB/pulsar searching software.
Instead, **sigpyproc** provides data manipulation routines which are well suited to
preprocessing and micro-management of pulsar data. The structure of the package also
makes it an ideal development environment, with a simple plug-and-play system with
new modules and extensions.



.. toctree::
  :maxdepth: 2
  :caption: Contents

  install
  modules

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart.ipynb
