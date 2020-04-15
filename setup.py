from setuptools import setup, find_packages


with open('README.md', 'r') as fd:
    long_description = fd.read()

setup(
    name='koogu',
    version='0.1.1',
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
        'os',
        'numpy',
        'h5py',
        'scipy',
        'librosa',
        'tensorflow>=2.0'
    ],
    #zip_safe=False,
)
