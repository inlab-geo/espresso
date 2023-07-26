from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError


class ExampleName(EspressoProblem):
    """Forward simulation class"""

    # TODO fill in the following metadata.
    metadata = {
        "problem_title": "",  # To be used in docs
        "problem_short_description": "",  # 1-3 sentences
        # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]
        "author_names": [
            "",
            "",
        ],
        # Contact for contributor/maintainer of espresso example
        "contact_name": "",
        "contact_email": "",
        # Reference to publication(s) that describe this example. In most
        # cases there will only be a single entry in this list.
        # List of (citation, doi) pairs e.g.
        # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
        # If there are no citations, use empty list `[]`
        "citations": [("", "")],
        # List of (title, address) pairs for any websites that
        # should be linked in the documentation, e.g.
        # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
        #                 ("Data source","https://www.data.com") ]
        # If there are no links, use empty list `[]`
        "linked_sites": [("", "")],
    }

    def __init__(self, example_number=1):
        super().__init__(example_number)

        """you might want to set some useful example-specific parameters here
        """
        # if example_number == 1:
        #     self.some_attribute = some_value_0
        #     self.another_attribute = another_value_0
        # elif example_number == 2:
        #     self.some_attribute = some_value_1
        #     self.another_attribute = another_value_1
        # else:
        #     raise InvalidExampleError

    @property
    def description(self):
        raise NotImplementedError  # optional

    @property
    def model_size(self):
        raise NotImplementedError  # TODO implement me

    @property
    def data_size(self):
        raise NotImplementedError  # TODO implement me

    @property
    def good_model(self):
        raise NotImplementedError  # TODO implement me

    @property
    def starting_model(self):
        raise NotImplementedError  # TODO implement me

    @property
    def data(self):
        raise NotImplementedError  # TODO implement me

    @property
    def covariance_matrix(self):  # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError  # optional

    def forward(self, model, return_jacobian=False):
        if return_jacobian:
            raise NotImplementedError  # optional
        else:
            raise NotImplementedError  # TODO implement me

    def jacobian(self, model):
        raise NotImplementedError  # optional

    def plot_model(self, model, **kwargs):
        raise NotImplementedError  # optional

    def plot_data(self, data1, data2=None, **kwargs):
        raise NotImplementedError  # optional

    def misfit(self, data1, data2):
        raise NotImplementedError  # optional

    def log_likelihood(self, data1, data2):
        raise NotImplementedError  # optional

    def log_prior(self, model):
        raise NotImplementedError  # optional
