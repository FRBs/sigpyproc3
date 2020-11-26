import codecs
import os
import sys
import sysconfig
import platform

from setuptools import setup

# With setup_requires, this runs twice - once without setup_requires, and once
# with. The build only happens the second time.
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    from setuptools import Extension as Pybind11Extension
    from setuptools.command.build_ext import build_ext


MACOS   = sys.platform.startswith("darwin")
TRIPLET = sysconfig.get_config_var('MULTIARCH') or ''  # x86_64-linux-gnu
BITS    = platform.architecture()[0][:-3]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def append_path(dirs_list, *args):
    entry = os.path.normpath(os.path.join(*args))
    if os.path.isdir(entry):
        if entry not in dirs_list:
            dirs_list.append(entry)


def get_include_dirs():
    prefix_dirs = ['/usr', '/usr/local']
    dirs = []
    sys_include = sysconfig.get_config_var('INCLUDEDIR')
    if 'FFTW_PATH' in os.environ:
        append_path(dirs, os.environ.get('FFTW_PATH', ''), 'include')
    if sys_include is not None:
        append_path(dirs, sys_include, TRIPLET)
        append_path(dirs, sys_include)
    for prefix in prefix_dirs:
        append_path(dirs, prefix, 'include', TRIPLET)
        append_path(dirs, prefix, 'include')
    return dirs


def get_library_dirs():
    prefix_dirs = ['/usr', '/usr/local']
    dirs = []
    sys_lib = sysconfig.get_config_var('LIBDIR')
    if 'FFTW_PATH' in os.environ:
        append_path(dirs, os.environ.get('FFTW_PATH', ''), 'lib')
    if sys_lib is not None:
        append_path(dirs, sys_lib, TRIPLET)
        append_path(dirs, sys_lib)
    for prefix in prefix_dirs:
        append_path(dirs, prefix, f'lib{BITS}')
        append_path(dirs, prefix, 'lib', TRIPLET)
        append_path(dirs, prefix, 'lib')
    return dirs


def get_compile_flags():
    cflags = ['-Wall', '-Wextra']
    if MACOS:
        cflags += ['-Xpreprocessor', '-fopenmp']
    else:
        cflags += ['-fopenmp']
    return cflags


def get_link_flags():
    lflags = ['-lm', '-lfftw3', '-lfftw3f']
    if MACOS:
        lflags += ['-lomp']
    else:
        lflags += ['-lgomp']
    return lflags


# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

package_version = get_version("sigpyproc/__init__.py")

ext_modules = [
    Pybind11Extension(
        'sigpyproc.libSigPyProc',
        sources=['c_src/bindings.cpp'],
        include_dirs=get_include_dirs() + ['c_src/'],
        library_dirs=get_library_dirs(),
        define_macros=[('VERSION_INFO', package_version)],
        extra_link_args=get_link_flags(),
        extra_compile_args=get_compile_flags(),
    ),
]

setup(name='sigpyproc',
      version=package_version,
      ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext},
      )
