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
    author_email='shyam.m@curtin.edu.au',
    license='GNU General Public License v3.0',
    packages=find_packages(include=['koogu', 'koogu.*']),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'audioread',
        'resampy',
        'tensorflow>=2.7'
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

