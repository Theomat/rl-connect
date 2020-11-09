#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='rfl',
    version="1.0.0",
    packages=find_packages("rfl"),
    author="Theomat",
    author_email="theomatricon@gmail.com",

    description="RL library",
    long_description=open('README.md').read(),
    install_requires=[
        "tensorflow",
        "numpy",
        "tqdm"
    ],

    include_package_data=True,
    url='https://github.com/Theomat/rl_connect',
    license="Creative Commons 3",
)
