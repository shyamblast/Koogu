from setuptools import setup, find_packages
import koogu


def build_long_description():

    # Read in first two sections from README.md
    # (without the page header & any links)
    stop_point_found = False
    readme = []
    with open('README.md', 'r') as fd:
        for line in fd:     # Skip until first blank line is encountered
            if line in ['\n', '\r\n']:
                break
        for line in fd:     # Read everything up until the "How to" section
            if line.startswith('How to use Koogu'):
                stop_point_found = True
                break
            readme.append(line)

    # Failsafe checks
    assert (stop_point_found and
            readme[0].startswith('A python package for')), \
        'Contents of README.md appears to have changed. ' + \
        'Might need to update setup.py'

    # Add everything from HOWTO.md
    with open('HOWTO.md', 'r') as fd:
        how_to = fd.read()

    return (    # Add header, and then join all the pieces
        'Koogu\n======\n\n' + ''.join(readme) + '\n\n' + how_to)


setup(
    name='koogu',
    version=koogu.__version__,
    description='Machine Learning for Bioacoustics',
    long_description=build_long_description(),
    long_description_content_type='text/markdown',
    url='https://shyamblast.github.io/Koogu/',
    author='Shyam Madhusudhana',
    author_email='shyamm@cornell.edu',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=[
        'numpy',
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
