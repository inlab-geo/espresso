"""Helper functions to analyse the base class `EspressoProblem`

This script assumes you have cofi-espresso installed via:
    $ python tools/build_package/build.py
Or the core package via:
    $ pip install .

"""

try:
    import cofi_espresso
except ModuleNotFoundError as e:
    e.msg += "\n\nNote: To run pre-build validation, please firstly install " \
             "`cofi_espresso` core module by running the following from the root" \
             "level of the project\n  $ pip install ."
    raise e


def inspect_espresso_problem():
    all_props = dir(cofi_espresso.EspressoProblem)
    abs_meta = cofi_espresso.EspressoProblem.__abstract_metadata_keys__
    abs_props = cofi_espresso.EspressoProblem.__abstractmethods__
    opt_props = [p for p in all_props if p not in abs_meta and p not in abs_meth]
    return {
        "required meta keys": abs_meta, 
        "required attributes": abs_props, 
        "optional attributes": opt_props, 
    }

def criteria_for_problems():
    raise NotImplementedError

def criteria_for_example():
    raise NotImplementedError
