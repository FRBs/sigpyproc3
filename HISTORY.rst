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
