from setuptools import setup

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='vaspvis',
    version='0.0.7',
    description='A highly flexible and customizable library for visualizing electronic structure data from VASP calculations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['vaspvis'],
    install_requires = ['pymatgen', 'matplotlib', 'numpy', 'pandas', 'scipy'],
    url='https://github.com/DerekDardzinski/vaspvis',
    authour='Derek Dardzinski',
    authour_email='dardzinski.derek@gmail.com',
    license='MIT',
)
