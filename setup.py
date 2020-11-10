from setuptools import setup, find_packages

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='vaspvis',
    version='0.2.6',
    description='A highly flexible and customizable library for visualizing electronic structure data from VASP calculations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires = ['pymatgen', 'matplotlib', 'numpy', 'pandas', 'scipy', 'ase', 'pychemia', 'pyprocar'],
    url='https://github.com/DerekDardzinski/vaspvis',
    authour='Derek Dardzinski',
    authour_email='dardzinski.derek@gmail.com',
    license='MIT',
)
