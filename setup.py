from setuptools import find_packages, setup

setup(
    name='hp_classify',
    packages=find_packages(),
    version='0.1.0',
    description="""Tools for classifying housing quality as an ordinal value using string
    description of housing materials.""",
    author='Joseph Frostad',
    license='BSD-3',
)
