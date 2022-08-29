# Espresso Documentation

This file contains some developer notes for Espresso documentation.

## Pre-requisites

This file [environment.yml](environment.yml) specifies packages required for developing 
this documentation.

## Build and see locally

1. Clone `espresso`

```console
$ git clone https://github.com/inlab-geo/espresso
$ cd espresso/docs
```

2. Create a virtual environment with mamba/conda (or other tools)

```console
$ mamba env create -f environment.yml
```

3. Install `cofi_espresso` with all contributions by running one of the below

```console
$ python tools/build_package/build.py               # if you want to be quick
$ python tools/build_package/build_with_checks.py   # if you want to feel safe
```

4. To build your changes

```console
$ make html
```

5. Serve the built files

```console
$ python -m http.server 8000 -d build/html
```

6. Open `localhost:8000` in your browser.

Redo step 3 (potentially in a different terminal session) when you've made new changes.

## Notes

Folder `source` contains all the text files for this documentation:

- `source/conf.py` has all the configurations for this documentation, including the
  theme, extensions, title, where templates are, what to exclude / include when building 
  the documentation, etc.

- `source/index.rst` corresponds to the home page, in which you can see the source 
  of the introductory paragraph, 4 panels in the home page and the table of contents.

- `source/user_guide/`, `source/contributor_guide/` and `source/developer_notes` contain
  the other documentation pages.
