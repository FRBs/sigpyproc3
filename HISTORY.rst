1.1.0 (2024-02-20)
++++++++++++++++++

- Adds a pulse extractor class. (`#28 <https://github.com/FRBs/sigpyproc3/pull/28>`_)
- Adds RFI masking module, PSRFITS reading module. (`#23 <https://github.com/FRBs/sigpyproc3/pull/23>`_)
- Adds DMT transform.
- Fixes bugs related to tfactor, types and apps.


1.0.0 (2022-02-01)
++++++++++++++++++

- First stable release of the new API.
- Adds Multifile support for ``SIGPROC Filterbank`` files. (`#13 <https://github.com/FRBs/sigpyproc3/pull/13>`_)
- Adds Numba backend for kernel functions in place of the earlier `pybind11 <https://github.com/pybind/pybind11>`_. (`#17 <https://github.com/FRBs/sigpyproc3/pull/17>`_)
- Removes dependence on `pybind11 <https://github.com/pybind/pybind11>`_ and `fftw3 <http://www.fftw.org/>`_.
- Adds Packaging structure as for the new API. (`#16 <https://github.com/FRBs/sigpyproc3/pull/16>`_)
- Adds ``io`` module for reading and writing of ``SIGPROC`` files.
- Adds fast median and mean filter (`bottleneck <https://github.com/pydata/bottleneck>`_) and Higher-Order running stats.
- Support for command-line utilities.
- Adds CHIME telescope code to ``params``.


0.5.5 (2021-03-31)
++++++++++++++++++

- Fixes major installation and Compilation errors on Mac.
- Adds SRT telescope code to ``params``.
- Adds more tests to increase code covergae.


0.5.0 (2020-11-18)
++++++++++++++++++

- Major update: replacing ctypes with `pybind11 <https://github.com/pybind/pybind11>`_.
- Adds coverage tests and online docs on the readthedocs platform.
- Fixes the cyclic import in modules.


0.1.1 (2020-08-14)
++++++++++++++++++

- First python3 stable release.
- Adds an arg ``only_valid_samples`` to ``FilterbankBlock.dedisperse()`` (`#2 <https://github.com/FRBs/sigpyproc3/pull/2>`_)
- Adds a new reader ``FilReader.readDedispersedBlock()``  (`#1 <https://github.com/FRBs/sigpyproc3/pull/1>`_)
- Adds coverage tests and progress bar using `tqdm <https://github.com/tqdm/tqdm>`_.
- Adds new methods ``removeBandpass``, ``removeZeroDM``, ``splitToBands``.
- Adds more stable and accurate ``getStats`` (`ref <https://doi.org/10.2172/1028931>`_).
