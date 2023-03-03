# Espresso Machine

Espresso machine contains all the code used for infrastructure.

When Espresso is packaged with the contributed inference problems, we move most of the 
files under this module into it as `espresso._machine`.

This page contains a minimal guide on using them. For more details about contributing to Espresso, check 
our [Contributor Guide](https://cofi-espresso.readthedocs.io/en/latest/contributor_guide/index.html).

## Generate new Espresso problem

```console
python new_contribution/create_new_contrb.py <example_name>
```

## Validate a new Espresso contribution (pre building)

```console
python build_package/validate.py [--pre] [--contrib <example_name>] [--all]
```

## Build Espresso with all contributions

```console
python build_package/build.py
```

## Validate a new Espresso contribution (post building)

```console
python build_package/validate.py --post [--contrib <example_name>] [--all]
```

## Build Espresso with pre/post validation

(combination of the three operations above)

```console
python build_package/build_with_checks.py
```
