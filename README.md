
sigpyproc
=========

Installation
------------

### Requirements

	
        * numpy 
        * ctypes 
        * FFTW3
        * OpenMP

### Step-by-step guide

As both setuptools and distutils do not have any clear method of support for
distributing C libraries for ctypes, the onus is on the user to ditribute the c libraries 
once built

1. Clone or download the git repositry from https://github.com/ewanbarr/sigpyproc

2. Unzip and untar if needed and move to source directory

3. Make sure FFTW3 has been compiled with the ``--enable-float`` and ``--enable-shared`` options

4. run ``sudo python setup.py install``

5. a lib/c and bin/ directories will be created in the source directory

6. Distribute the contents of these directories if required

7. If you are developing, then append the lib/c directory to the LD_LIBRARY_PATH environment variable





