# How-to Guides

This is a minimal guide on using the scripts in this folder.

For more details about contributing to Espresso, check 
[this documentation page](https://cofi-espresso.readthedocs.io/en/latest/contributor_guide/ways.html).

## Generate new Espresso problem

```console
python new_contribution/create_new_contrb.py <example_name>
```

## Validate a new Espresso contribution (pre building)

```console
python build_package/validate.py
```

## Build Espresso with all contributions

```console
python build_package/build.py
```

## Validate a new Espresso contribution (post building)

```console
python build_package/validate.py post
```

## Build Espresso with pre/post validation

(combination of the three operations above)

```console
python build_package/build_with_checks.py
```
