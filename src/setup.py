from setuptools import setup, find_packages, Extension
import os
import numpy as np

try:
    from Cython.Build import cythonize
    extension = [
        Extension('simnibs.cython_code.cython_msh',
                  ["simnibs/cython_code/cython_msh.pyx"],
                  include_path=[np.get_include()],
                  include_dirs=[np.get_include()]),
        Extension('simnibs.cython_code._marching_cubes_lewiner_cy',
                  ["simnibs/cython_code/_marching_cubes_lewiner_cy.pyx"],
                  include_path=[np.get_include()],
                  include_dirs=[np.get_include()])]
    extension = cythonize(extension, language_level=2)
except ImportError:
    extension = [Extension('simnibs.cython_code.cython_msh',
                           ['simnibs/cython_code/cython_msh.c'],
                           include_dirs=[np.get_include()]),
                 Extension('simnibs.cython_code._marching_cubes_lewiner_cy',
                           ['simnibs/cython_code/_marching_cubes_lewiner_cy.c'],
                           include_dirs=[np.get_include()])]

setup(name='simnibs',
      version='2.1.2',
      description='simnibs python stuff',
      author='Guilherme B Saturnino, Jesper D Nielsen, Andre Antunes, Kristoffer H '+ \
      'Madsen, Axel Thielscher',
      author_email='support@simnibs.org',
      packages=find_packages(),
      license='GPL3',
      #packages=['simnibs'],
      ext_modules=extension,
      install_requires=[
        'numpy>=1.13',
        'scipy>=1.0.0',
        'nibabel'],
      zip_safe=False)
