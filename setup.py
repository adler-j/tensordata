"""Setup script for tensordata.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='tensordata',

    version='0.1.0',

    description='Datasets for Tensorflow',

    url='https://github.com/adler-j/tensordata',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    license='MPL',

    keywords='research development tensorflow data',

    packages=find_packages(exclude=['*test*']),
    package_dir={'tensordata': 'tensordata'},

    install_requires=['numpy', 'tensorflow']
)
