# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from setuptools import setup

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
setup(
    name="rehydra-example-registered-plugin",
    version="1.0.0",
    author="Jasha Sommer-Simpson",
    author_email="jasha10@fb.com",
    description="Example of Rehydra Plugin Registration",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/rehydra/",
    packages=["example_registered_plugin"],
    classifiers=[
        # Feel free to use another license.
        "License :: OSI Approved :: MIT License",
        # Rehydra uses Python version and Operating system to determine
        # In which environments to test this plugin
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # consider pinning to a specific major version of Rehydra to avoid unexpected problems
        # if a new major version of Rehydra introduces breaking changes for plugins.
        # e.g: "rehydra-core==1.1.*",
        "rehydra-core",
    ],
)
