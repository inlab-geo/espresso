from cofi_espresso import EspressoProblem, InvalidExampleError
from .waveTracker import *


class FmmTomography(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Fast Marching Wave Front Tracking",                # To be used in docs
        "problem_short_description": "The wave front tracker routines solves boundary"\
            " value ray tracing problems into 2D heterogeneous wavespeed media, "\
            "defined by continuously varying velocity model calculated by 2D cubic "\
            "B-splines.",    # 1-3 sentences

        "author_names": ["Nick Rawlinson","Malcolm Sambridge"],    # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

        "contact_name": "Malcolm Sambridge",         # Contact for contributor/maintainer of espresso example
        "contact_email": "Malcolm.Sambridge@anu.edu.au",

        "citations": [
            (
                "Rawlinson, N., de Kool, M. and Sambridge, M., 2006. Seismic wavefront tracking in 3-D heterogeneous media: applications with multiple data classes, Explor. Geophys., 37, 322-330.",
                None
            ),
            (
                "Rawlinson, N. and Urvoy, M., 2006. Simultaneous inversion of active and passive source datasets for 3-D seismic structure with application to Tasmania, Geophys. Res. Lett., 33 L24313",
                "10.1029/2006GL028105"
            ),
            (
                "de Kool, M., Rawlinson, N. and Sambridge, M. 2006. A practical grid based method for tracking multiple refraction and reflection phases in 3D heterogeneous media, Geophys. J. Int., 167, 253-270",
                None
            )
        ], # Reference to publication(s) that describe this example. In most 
                                # cases there will only be a single entry in this list.
                                # List of (citation, doi) pairs e.g. 
                                # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", None)]
                                # If there are no citations, use empty list `[]`

        "linked_sites": [("Software published on iEarth","http://iearth.edu.au/codes/FMTOMO/")],  # List of (title, address) pairs for any websites that 
                                    # should be linked in the documentation, e.g.
                                    # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
                                    #                 ("Data source"),"https://www.data.com") ]
                                    # If there are no links, use empty list `[]`
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
    def covariance_matrix(self):                # optional
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

    def misfit(self, data, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional
