from setuptools import setup, find_packages
from setuptools.command.install import install



with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='amadeus',
    version='0.3.0',
    description='Content based recommendation system',
    long_description=readme,
    author='Gerardo Vitagliano, Solange Nunes',
    author_email='vitaglianog@gmail.com',
    url='https://github.com/Mozzers/amadeus',
    license=license,
    packages=find_packages(exclude=('docs')),
    install_requires=[
    "networkx",
   'pyBN'
   ]

)

