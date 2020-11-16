.. _dev:

Developer guide
===============

Here we will cover all the steps required to add new functionality to the
``sigpyproc`` package. To do this, we will first consider adding a new function
to the :class:`~sigpyproc.Filterbank.Filterbank` class, before going on to extend
the function to deal with an arbitrary data format.


Adding a new function: ``bandpass()``
-------------------------------------
**Aim:** Add a new function called bandpass, which will return the total power
as a function of frequency for our observation.

**Files to be modified:** ``sigpyproc/Filterbank.py``, ``c_src/kernels.hpp``
, ``c_src/bindings.cpp``.

Step 1: Write the Python part
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first step is to write the Python side of the function. As this function
will run on data with both time and frequency resolution,
it belongs in the :class:`~sigpyproc.Filterbank.Filterbank` class.

.. code-block:: python

    def bandpass(self, gulp=512, **kwargs):
        bpass_ar = np.zeros(self.header.nchans, dtype="float64")
        num_samples = 0
        for nsamps, ii, data in self.readPlan(gulp, **kwargs):
            lib.getBpass(data, bpass_ar, self.header.nchans, nsamps)
            num_samples += nsamps
        bpass_ar = bpass_ar / num_samples
        return TimeSeries(bpass_ar, self.header.newHeader({"nchans": 1}))

Looking at the important lines, we have:
Return an instance of the :class:`~sigpyproc.TimeSeries.TimeSeries` class.
The :class:`~sigpyproc.TimeSeries.TimeSeries` class takes two arguments,
an instance of :py:obj:`numpy.ndarray` and an instance of
:class:`~sigpyproc.Header.Header`.

Now we have something similar to a normal :py:obj:`numpy.ndarray`,
which exports several other methods for convenience.

Step 2: Write the C++ part
^^^^^^^^^^^^^^^^^^^^^^^^^^
We called a C++ library function named getBpass. This function belongs in the
``c_src/kernels.hpp`` file (or anywhere you please, as long as it is properly
binded with the `pybind11 <https://pybind11.readthedocs.io/>`_ library).
In ``kernels.hpp``, our function looks like:

.. code-block:: C++

    template <class T>
    void getBpass(T* inbuffer, double* outbuffer, int nchans,
                int nsamps) {
    #pragma omp parallel for default(shared)
        for (int jj = 0; jj < nchans; jj++) {
            for (int ii = 0; ii < nsamps; ii++) {
                outbuffer[jj] += inbuffer[(nchans * ii) + jj];
            }
        }
    }

This function receives a block of data and sums that block along the time axis.
We use an optional C pre-processor directive to enable OpenMP threading.
Realistically this call is not required here, but it is left here of a clear
example of how to quicly multi-thread a system of loops.


Reporting an issue
------------------

`Post an issue on the GitHub repository
<https://github.com/FRBs/sigpyproc3/issues>`_. When you post an issue,
please provide the details to reproduce the issue.


Contributing code or documentation
----------------------------------

