class EspressoError(Exception):
    """ Base class for all Espresso errors """
    pass

# Multiple inheritance means this can be caught by all of the following:
# ... except InvalidExampleError
# ... except EspressoError
# ... except ValueError
class InvalidExampleError(EspressoError, ValueError):
    """ Raised if user attempts to instantiate an example number that does not exist """
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


