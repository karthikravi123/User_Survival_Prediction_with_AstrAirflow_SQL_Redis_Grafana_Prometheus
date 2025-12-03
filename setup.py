from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
   
setup(
    name="mlops-project3",
    version="0.01",
    author="karthik",
    packages=find_packages(),
    install_requires = requirements,
)
