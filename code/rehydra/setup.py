# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
import pathlib

import pkg_resources
from setuptools import find_namespace_packages, setup

from build_helpers.build_helpers import (
    ANTLRCommand,
    BuildPyCommand,
    CleanCommand,
    Develop,
    SDistCommand,
    find_version,
)

with pathlib.Path("requirements/requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setup(
        cmdclass={
            "antlr": ANTLRCommand,
            "clean": CleanCommand,
            "sdist": SDistCommand,
            "build_py": BuildPyCommand,
            "develop": Develop,
        },
        name="rehydra",
        version=find_version("rehydra", "__init__.py"),
        packages=find_namespace_packages(include=["rehydra", "rehydra.*"]),
        include_package_data=True,
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
        ],
        install_requires=install_requires,
        entry_points={"pytest11": ["rehydra_pytest = rehydra.extra.pytest_plugin"]},
        # Install development dependencies with
        # pip install -r requirements/dev.txt -e .
    )
