from setuptools import setup, find_packages
import koogu


with open('README.md', 'r') as fd:
    long_description = fd.read()

setup(
    name='koogu',
    version=koogu.__version__,
    description='Machine Learning for Bioacoustics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shyamblast/Koogu',
    author='Shyam Madhusudhana',
    author_email='shyamm@cornell.edu',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=[
        'numpy',
        'h5py',
        'scipy',
        'librosa',
        #'tensorflow-gpu>=2.4'
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Multimedia :: Sound/Audio :: Analysis"
    ],
)

# Below piece of code copied (and adapted) from
#     https://github.com/openai/baselines/blob/master/setup.py
# Ensure there is some tensorflow build with version above 2.4
import pkg_resources
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version at least 2.4'
import re
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('2.4'), 'TensorFlow needed, of version at least 2.4'
