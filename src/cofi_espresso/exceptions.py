class EspressoError(Exception):
    """ Base class for all Espresso errors """
    pass

# Multiple inheritance means this can be caught by all of the following:
# ... except InvalidExampleError
# ... except EspressoError
# ... except ValueError
class InvalidExampleError(EspressoError, ValueError):
    """ Raised if user attempts to instantiate an example number that does not exist """
    pass
