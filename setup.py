from setuptools import setup, find_packages

import pathlib
# get version number
_ROOT = pathlib.Path(__file__).parent

with open(str(_ROOT / "src" / "cofitestsuite" / "_version.py")) as f:
    for line in f:
        if line.startswith("__version__ ="):
            _, _, version = line.partition("=")
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError("unable to read the version from ./_version.py")



setup(
    name="cofitestsuite",
    version=VERSION,
    description="A gravity forward calculation",
    author="Hannes",
    author_email="hannes.hollmann@anu.edu.au",
    license='BSD 2-clause',
    install_requires=['numpy',],
    include_package_data=True,

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
