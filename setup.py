# To run this in Pycharm:
# 1. Go to the terminal
# 2. Run: python setup.py build_ext --inplace


from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

module = Extension ('graph_alg_c_lib', sources=['graph_alg_c_lib.pyx'], language="c++", extra_compile_args=["-std=c++11"])

setup(
    name='Separators_Enumeration_1',
    version='1',
    author='Einan',
    ext_modules=[module]
)
