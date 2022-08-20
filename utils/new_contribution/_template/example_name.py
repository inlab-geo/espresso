from cofi_espresso import EspressoProblem

# TODO fill in the following metadata.
problem_title = "" # To be used in docs
problem_short_description = "" # 1-3 sentences

author_names = ["",""]  # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

contact_name = "" # Contact for contributor/maintainer of espresso example.
contact_email = ""

citations = [("","")] # Reference to publication(s) that describe this example. In most 
                      # cases there will only be a single entry in this list.
                      # List of (citation, doi) pairs e.g. 
                      # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", None)]
                      # If there are no citations, use empty list `[]`

linked_sites = [("","")] # List of (title, address) pairs for any websites that 
                         # should be linked in the documentation, e.g.
                         # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
                         #                 ("Data source"),"https://www.data.com") ]
                         # If there are no links, use empty list `[]`

class ExampleName(EspressoProblem):
    """Forward simulation class
    """

    def __init__(self, example_number=1):
        super().__init__(example_number)

        """you might want to set other useful example specific parameters here
        so that you can access them in the other functions see the following as an 
        example (suggested) usage of `self.params`
        """
        # if example_number == 1:
        #     self.params["some_attribute"] = some_value_0
        #     self.params["another_attribte"] = another_value_0
        # elif example_number == 2:
        #     self.params["some_attribute"] = some_value_1
        #     self.params["another_attribte"] = another_value_1
        # else:
        #     raise ValueError(
        #         "The example number supplied is not supported, please consult "
        #         "Espresso documentation at "
        #         "https://cofi-espresso.readthedocs.io/en/latest/contrib/index.html "
        #         "for problem-specific metadata, e.g. number of examples provided"
        #     )

    @property
    def description(self):
        raise NotImplementedError               # optional

    @property
    def model_size(self):
        raise NotImplementedError               # TODO implement me

    @property
    def data_size(self):
        raise NotImplementedError               # TODO implement me

    @property
    def good_model(self):
        raise NotImplementedError               # TODO implement me

    @property
    def starting_model(self):
        raise NotImplementedError               # TODO implement me
    
    @property
    def data(self):
        raise NotImplementedError               # TODO implement me

    @property
    def covariance_matrix(self):                # TODO implement me
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional
        
    def forward(self, model, with_jacobian=False):
        if with_jacobian:
            raise NotImplementedError           # optional
        else:
            raise NotImplementedError           # TODO implement me
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self, data, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data, data2):              # optional
        raise NotImplementedError

    def log_likelihood(self,data,data2):        # optional
        raise NotImplementedError
    
    def log_prior(self, model):                 # optional
        raise NotImplementedError