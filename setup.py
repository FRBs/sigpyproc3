from distutils.core import setup
import os,sys,glob

def describe(filename):
    f = open(filename,"r")
    lines = f.readlines()
    return "".join(lines)

LIBDEST = os.path.join(os.getcwd(),"lib/c/")
try:
    os.makedirs(LIBDEST)
except OSError:
    pass

setup(name='sigpyproc',
      version='0.0.1',
      description='Python pulsar data toolbox',
      author='Ewan Barr',
      author_email='ewan.d.barr@googlemail.com',
      packages=['sigpyproc'],
      long_description=describe('README.md')
      #scripts=glob.glob("scripts/*.py")
      )

pythonversion = "python%s"%(".".join(sys.version.split(".")[:2]))
os.putenv("PYTHONVER",pythonversion)
os.chdir("c_src/")
os.system("make all")
os.system("mv *.so %s"%(LIBDEST))
os.system("make clean")
