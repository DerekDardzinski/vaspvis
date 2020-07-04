from setuptools import setup

setup(
    name='vaspvis',
    version='0.0.1',
    description='A highly flexible and customizable library for visualizing electronic structure data from VASP calculations.',
    py_modules=['band', 'dos'],
    package_dir = {'':'core'}
)
