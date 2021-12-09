from distutils.core import setup
import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()

setup(
    name='tokenlearner_pytorch',
    version='0.1.1',    
    description='Unofficial PyTorch implementation of TokenLearner by Google AI',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    url='https://github.com/rish-16/tokenlearner-pytorch',
    author='Rishabh Anand',
    author_email='mail.rishabh.anand@gmail.com',
    license='MIT',
    packages=['tokenlearner_pytorch'],
    install_requires=['torch'],
    
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
    ],
)