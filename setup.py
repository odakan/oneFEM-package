from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="oneFEM",
    version="0.0.1",
    description="A Python package for finite element modeling and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://feimplementlib.github.io/",
    author="FE Implementation Library",
    author_email="fe.implement.library@gmail.com",
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'oneFEM = oneFEM._systools.cli:_cli_main',
        ],
    },
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "ipython"
    ],
    python_requires=">=3.7, <4",
)
