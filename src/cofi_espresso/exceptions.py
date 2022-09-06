class EspressoError(Exception):
    """Base class for all Espresso errors
    """
    pass

# Multiple inheritance means this can be caught by all of the following:
# ... except InvalidExampleError
# ... except EspressoError
# ... except ValueError
class InvalidExampleError(EspressoError, ValueError):
    r"""Raised if user attempts to instantiate an example number that does not exist

    This is a subclass of both :exc:`EspressoError` and :exc:`ValueError`.

    Examples
    --------

    >>> from cofi_espresso import SimpleRegression, InvalidExampleError
    >>> try:
    ...     reg = SimpleRegression(6)
    ... except InvalidExampleError:
    ...     print("InvalidExampleError triggered")
    ... 
    InvalidExampleError triggered
    
    """
    def __init__(self, *args):
        super().__init__(*args)
    def __str__(self):
        super_msg = super().__str__()
        msg = "Unrecognised example number.\n\nPlease refer to the Espresso documentation " \
              "(https://cofi-espresso.readthedocs.io/)\nfor full details of the examples " \
              "provided within this test problem."
        if len(super_msg)>0:
            return msg+"\n\n"+super_msg
        else:
            return msg


