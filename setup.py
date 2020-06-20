#!/usr/bin/env python

from setuptools import setup
import os

def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from MNE
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)

setup(name='conf_analysis',
      version='0.0.1',
      description='Python MEG tools for preprocessing etc.',
      author='Niklas Wilming',
      author_email='nwilming@uke.de',
      url='',
      packages=package_tree('conf_analysis'))
