from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("myCythonFunc", ["myCythonFunc.pyx"], extra_compile_args=["-O2"], extra_link_args=[])]
setup(name = 'myFunctions', cmdclass = {'build_ext': build_ext}, include_dirs = [numpy.get_include()], ext_modules = ext_modules)

#To install in place: 
#python setup.py build_ext --inplace -c mingw32
