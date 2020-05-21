from setuptools import setup, find_packages


with open('README.md', 'r') as fd:
    long_description = fd.read()

setup(
    name='koogu',
    version='0.2.2',
    description='...',
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
        #'tensorflow-gpu>=2.0'
    ],
    #zip_safe=False,
)

# Below piece of code copied (and adapted) from
#     https://github.com/openai/baselines/blob/master/setup.py
# Ensure there is some tensorflow build with version above 2.0
import pkg_resources
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version at least 2.0'
import re
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('2.0'), 'TensorFlow needed, of version at least 2.0'
