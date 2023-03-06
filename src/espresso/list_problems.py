_all_problems = []

def list_problem_names():
    """Returns a list of all Espresso problem names"""
    _all_names = [p.__name__ for p in _all_problems]
    return _all_names

def list_problems():
    """Returns a list of all Espresso problem classes"""
    return _all_problems


# from .example_name import ExampleName

# __all_problems__ = [ ExampleName, ]
