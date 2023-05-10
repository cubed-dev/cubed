#!/usr/bin/env python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='cubed',
        packages=['cubed'],
        package_dir={'cubed': 'cubed'},  # avoids pip install problems if one creates a tmp directory alongside cubed directory
    )
