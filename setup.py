from setuptools import setup, find_packages

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vaspvis",
    version="1.2.17",
    description="A highly flexible and customizable library for visualizing electronic structure data from VASP calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pyprocar==5.6.6",
        "scipy==1.10.1",
        "pymatgen",
        "matplotlib",
        "numpy",
        "pandas",
        "ase",
        "pychemia",
        "fastdtw",
        "scikit-learn",
    ],
    url="https://github.com/DerekDardzinski/vaspvis",
    author="Derek Dardzinski",
    author_email="dardzinski.derek@gmail.com",
    license="MIT",
)
