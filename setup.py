from setuptools import setup, find_packages
import os, sys, distutils, numpy


setup(name='comPy',

	  version='0.2',
	  description='communication system and signal processing library',
	  url='https://github.com/yang0110/comPy',
	  author='Kaige Yang',
	  license='MIT',
	  packages=['comPy', 'comPy.comm','comPy.signal'],
	  install_requires=['numpy','scipy','matplotlib'],
	  zip_safe=False,
	  classifiers = [

        'Development Status :: 1st stage',

        'Intended Audience :: Science/Research',

        'Intended Audience :: communication and signal processing academic',

        'Programming Language :: Python',

        'Topic :: Scientific/Engineering',]
        )

        


DESCRIPTION = 'Communication signal system and signal processing  with Python'

LONG_DESCRIPTION = open('README.py').read()

MAINTAINER= 'Kaige Yang'

OWNER_EMAIL = 'Kaige.yang0110@gmail.com'

URL = 'https://github.com/yang0110/comPy'

LICENSE = 'MIT'

VERSION = '0.2'


