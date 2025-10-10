import os
import setuptools

long_description = """A project to explore best practices in QAOA parameter setting."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="qaoa_parameter_setting",
    version=VERSION,
    description="QAOA Parameter setting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting",
    author="Opt. working group",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qaoa",
    packages=setuptools.find_packages(
        include=["qaoa_parameter_setting", "qaoa_parameter_setting.*"]
    ),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
