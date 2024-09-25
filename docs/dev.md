# Developer documentation

Here we will cover all the steps required to add new functionality to the
`sigpyproc` package. To do this, we will first consider adding a new function
to the {py:class}`~sigpyproc.base.Filterbank` class.

## Adding a new function: `bandpass()`

**Aim:** Add a new function called `bandpass`, which will return the total power
as a function of frequency for the filterbank data.

**Files to be modified:** `sigpyproc/base.py`, `sigpyproc/core/kernels.py`.

### Step 1: Write the API part

The first step is to write the user API of the function. As this function will run on
data with both time and frequency resolution, it belongs in the
{py:class}`~sigpyproc.base.Filterbank` class.

```python
def bandpass(self, gulp=16384, **plan_kwargs):
    bpass_ar = np.zeros(self.header.nchans, dtype=np.float32)
    num_samples = 0
    for nsamps, ii, data in self.read_plan(**plan_kwargs):
        kernels.extract_bpass(data, bpass_ar, self.header.nchans, nsamps)
        num_samples += nsamps
    bpass_ar /= num_samples
    hdr_changes = {"nchans": 1, "nsamples": len(bpass_ar)}
    return TimeSeries(bpass_ar, self.header.new_header(hdr_changes))
```

Looking at the important lines, we have returned an instance of the
{py:class}`~sigpyproc.timeseries.TimeSeries` class.
The {py:class}`~sigpyproc.timeseries.TimeSeries` class takes two arguments, an instance
of {py:obj}`numpy.ndarray` and an instance of {py:class}`~sigpyproc.header.Header`.
It exposes several other methods for convenience.

### Step 2: Write the core Numba kernel

We called a kernel function named `extract_bpass`. This function belongs in the
`sigpyproc/core/kernels.py` file. The function looks like:

```python
@njit(
    ["void(u1[:], f4[:], i4, i4)", "void(f4[:], f4[:], i4, i4)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def extract_bpass(inarray, outarray, nchans, nsamps):
    for ichan in prange(nchans):
        for isamp in range(nsamps):
            outarray[ichan] += inarray[nchans * isamp + ichan]
```

This function receives a block of data and sums that block along the time axis.
We use the jit compiler directive `parallel` to enable OpenMP threading.

## Reporting an issue

[Post an issue on the GitHub repository](https://github.com/FRBs/sigpyproc3/issues).
When you post an issue, please provide the details to reproduce the issue.

## Contributing code or documentation

An excellent place to start is the
[AstroPy developer docs](https://docs.astropy.org/en/stable/development/workflow/development_workflow.html).
