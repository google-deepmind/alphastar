# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module setuptools script."""

from setuptools import find_packages
from setuptools import setup

LONG_DESCRIPTION = (
    'This package provides libraries for ML-friendly Starcraft II trajectory '
    'data generation, architectures and agents along with the entire training '
    'and evaluation setup for training an offline RL agent.'
)

setup(
    name='AlphaStar',
    version='1.0.0',
    description='Package for offline RL agent training and evaluation on StarCraftII',
    long_description=LONG_DESCRIPTION,
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='StarCraft AI',
    url='https://github.com/deepmind/alphastar',
    # This is important if you have some non-standard files as part of package.
    include_package_data=True,
    packages=find_packages(),
    # dm-acme 0.2.4 until the clash of pybind11 absl status bindings with
    # PySC2 is resolved.
    install_requires=[
        'apache-beam[gcp]',
        'dm-acme[jax,tf]',
        's2clientprotocol',
        'pysc2',
        'ml-collections',
        'tensorflow-datasets',
        'pandas',
        'jax>=0.2.26',
        'jaxlib',
        'dm-pix'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
