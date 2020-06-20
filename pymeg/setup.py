#!/usr/bin/env python

# Requires:
#   - pyfftw >= 0.10.3
#   - joblib
#   - pymne

from setuptools import setup

setup(name='pyMEG',
      version='0.0.1',
      description='Python MEG tools for preprocessing etc.',
      author='Niklas Wilming',
      author_email='nwilming@uke.de',
      url='https://github.com/DonnerLab/pymeg',
      packages=['pymeg'],
      scripts=['scripts/to_cluster', 'scripts/reconall.sh'])
