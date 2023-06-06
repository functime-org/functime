import os
import sys

# Add current directory to the path
sys.path.append(os.getcwd())

from setuptools import setup  # noqa

setup(packages=["functime"])
