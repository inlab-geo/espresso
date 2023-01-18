from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError

import numpy as np


class MyNewExample(EspressoProblem):
    """Forward simulation class
    """

    # TODO fill in the following metadata.
    metadata = {
        "problem_title": "My New Example",                # To be used in docs
        "problem_short_description": "This is an example on xxx",    # 1-3 sentences

        "author_names": ["Dummy Name"],    # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

        "contact_name": "Dummy Name",         # Contact for contributor/maintainer of espresso example
        "contact_email": "dummy.name@anu.edu.au",

        "citations": [], # Reference to publication(s) that describe this example. In most 
                                # cases there will only be a single entry in this list.
                                # List of (citation, doi) pairs e.g. 
                                # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
                                # If there are no citations, use empty list `[]`

        "linked_sites": [("Dummy's personal blog","https://dummy-name.github.io")],  # List of (title, address) pairs for any websites that 
                                    # should be linked in the documentation, e.g.
                                    # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
                                    #                 ("Data source"),"https://www.data.com") ]
                                    # If there are no links, use empty list `[]`
    }


    def __init__(self, example_number=1):
        super().__init__(example_number)

        """you might want to set some useful example-specific parameters here
        """
        if example_number == 1:
            self.some_attribute = 1
            self.another_attribute = 1
        elif example_number == 2:
            self.some_attribute = 2
            self.another_attribute = 2
        else:
            raise InvalidExampleError

    @property
    def description(self):
        raise NotImplementedError               # optional

    @property
    def model_size(self):
        return 1

    @property
    def data_size(self):
        return 2

    @property
    def good_model(self):
        return np.array([1])

    @property
    def starting_model(self):
        return np.array([0])
    
    @property
    def data(self):
        return np.array([1,2])

    @property
    def covariance_matrix(self):                # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional
        
    def forward(self, model, with_jacobian=False):
        if with_jacobian:
            raise NotImplementedError           # optional
        else:
            return np.array([1,2])
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self, data, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional
