#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

from distutils.command.install import install
import json
import os
import platform
from setuptools import setup, find_namespace_packages

README = r"""
Fleet Compiler CLI Tools
"""

exe_suffix = ".exe" if platform.system() == "Windows" else ""

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
IREESRC_DIR = os.path.join(THIS_DIR, "..", "..", "..", "..")
VERSION_INFO_FILE = os.path.join(IREESRC_DIR, "version_info.json")


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info()
except FileNotFoundError:
    print("version_info.json not found. Using defaults")
    version_info = {}

PACKAGE_SUFFIX = version_info.get("package-suffix") or ""
PACKAGE_VERSION = version_info.get("package-version") or "0.1dev1"

setup(
    name=f"fleet-compiler-cli{PACKAGE_SUFFIX}",
    version="0.1",
    description="Fleet Compiler CLI",
    license='Apache License v2.0',
    author='Shuang Huang',
    author_email='nikshuang@sina.com',
    url='https://github.com/alexshuang/fleet-compiler',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'],
    packages=find_namespace_packages(
        include=[
            "fleet_compiler.tools.cli",
            "fleet_compiler.tools.cli.*",
        ]
    ),
    package_data={
        "fleet_compiler.tools.cli": [
            f"fleet-compiler-cli{exe_suffix}",
        ],
    },
    entry_points={
        "console_scripts": [
            "fleet-compiler-cli = fleet_compiler_cli.fleet_compiler.tools.cli.scripts.fleet_compiler_cli.__main__:main",
        ],
    },
    zip_safe=False,  # This package is fine but not zipping is more versatile.
)
