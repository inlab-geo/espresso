_all_problems = []
_capability_matrix = dict()

def list_problem_names(capabilities: list = None):
    """Returns a list of all Espresso problem names"""
    _problems = list_problems(capabilities)
    _all_names = [p.__name__ for p in _problems]
    return _all_names

def list_problems(capabilities: list = None):
    """Returns a list of all Espresso problem classes"""
    if capabilities is None:
        return _all_problems
    elif not isinstance(capabilities, (list, set, tuple)):
        raise ValueError(
            "pass in a list of capabilities, e.g. "
            "`espresso.list_problems(['plot_model'])"
        )
    else:
        _problem_names = []
        for p, c in _capability_matrix.items():
            ok = True
            for to_check in capabilities:
                if not (to_check in c and c[to_check] == 1):
                    ok = False
                    break
            if ok:
                _problem_names.append(p)
        _problems = []
        for p in _all_problems:
            if p.__name__ in _problem_names:
                _problems.append(p)
        return _problems

def list_capabilities(problem_names: Union[list, str] = None) -> dict:
    problems = list(problem_names)
    return {k:v for k,v in _capability_matrix.items() if k in problems}


# from .example_name import ExampleName

# _all_problems = [ ExampleName, ]

# _capability_matrix = {
#     "ExampleName": {
#         "model_size": 1,
#         "data_size": 1,
#         "starting_model": 1,
#         ...
#     },
#     ...
# }
