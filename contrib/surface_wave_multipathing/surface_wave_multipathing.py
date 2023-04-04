from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError

import sys
import os
path = os.path.dirname(__file__)
sys.path.append(path+"/build/_deps/swmp-build/python")

import swmp

class SurfaceWaveMultipathing(EspressoProblem):
    """Forward simulation class
    """

    # TODO fill in the following metadata.
    metadata = {
        "problem_title": "SurfaceWaveMultipathing",                # To be u
        "problem_short_description": "Surface wave multipathing tomography problems as described in Hauser et. al. 2008",    # 1-3 sentences

        "author_names": ["Juerg Hauser"],    # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

        "contact_name": "Juerg Hauser",         # Contact for contributor/maintainer of espresso example
        "contact_email": "juerg.hauser@csiro.au",

        "citations": [("Hauser, J., Sambridge, M. and Rawlinson, N. (2008). Multiarrival wavefront tracking and its applications. Geochem. Geophys. Geosyst., 9(11), Q11001. https://doi.org/10.1029/2008GC002069","")], # Reference to publication(s) that describe this example. In most
                                # cases there will only be a single entry in this list.
                                # List of (citation, doi) pairs e.g.
                                # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
                                # If there are no citations, use empty list `[]`

        "linked_sites": [("Parent project on github","https://github.com/JuergHauser/swmp")],  # List of (title, address) pairs for any websites that
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
        if example_number == 1:
            self.swmp_demo='checkerboard'
        elif example_number == 2:
            self.swmp_demo='random'
        elif example_number == 3:
            self.swmp_demo='blobs'
        else:
            raise InvalidExampleError

        path = os.path.dirname(__file__)
        self.wdir = path+'/build/_deps/swmp-build/demos/'+self.swmp_demo
        os.chdir(self.wdir)

        # True/Good model
        wt=swmp.WaveFrontTracker()
        wt.read_configuration(self.wdir+'/input/true_rat.in')
        self._good_model=wt.get_model_vector()
        wt.read_observations(self.wdir+'/data/observed.dat')
        self._data=wt.tt.obs[:,3]

        # Initial model / Starting model
        wt=swmp.WaveFrontTracker()
        wt.read_configuration(self.wdir+'/input/start_rat.in')
        self._starting_model=wt.get_model_vector()

    @property
    def description(self):
        raise NotImplementedError               # optional

    @property
    def model_size(self):
        return self.good_model.shape[0]

    @property
    def data_size(self):
        return self.data.shape[0]

    @property
    def good_model(self):
        return self._good_model

    @property
    def starting_model(self):
        return self._starting_model

    @property
    def data(self):
        return self._data

    @property
    def covariance_matrix(self):                # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional

    def forward(self, model, with_jacobian=False):
        if with_jacobian:
            wt=swmp.WaveFrontTracker()
            wt.read_configuration(self.wdir+'/input/current_rat.in')
            wt.set_model_vector(model)
            wt.forward()
            wt.read_predictions(self.wdir+'/output/current/arrivals.dat')
            wt.read_jacobian()
            return wt.tt.pred[:,3],wt.get_jacobian()
        else:
            wt=swmp.WaveFrontTracker()
            wt.read_configuration(self.wdir+'/input/current_rat.in')
            wt.set_model_vector(model)
            wt.forward()
            wt.read_predictions(self.wdir+'/output/current/arrivals.dat')
            return wt.tt.pred[:,3]

    def jacobian(self, model):
        return self.forward(model, True)

    def plot_model(self, model):
        import swmp
        self.vis=swmp.Visualisation()
        self.vis.read_configuration(self.wdir+'/input/current_rat.in')
        self.vis.set_model_vector(model)
        return self.vis.get_model_figure(5,5)

    def plot_raypaths(self, model):
        import swmp
        self.vis=swmp.Visualisation()
        self.vis.read_configuration(self.wdir+'/input/current_rat.in')
        self.vis.read_raypaths()
        self.vis.set_model_vector(model)
        return self.vis.get_raypath_figure(5,5)

    def plot_data(self, data, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional

    def log_prior(self, model):
        raise NotImplementedError               # optional
        
    def joint_data_and_jacobian(self):
        wt=swmp.WaveFrontTracker()
        wt.read_configuration(self.wdir+'/input/current_rat.in')
        wt.read_predictions(self.wdir+'/output/current/arrivals.dat')
        wt.read_observations(self.wdir+'/data/observed.dat')
        wt.read_jacobian()
        wt.join_observations_and_predictions()
        obs=wt.get_joint_observations()
        pred=wt.get_joint_predictions()
        jac=wt.get_joint_jacobian()
        return obs[:,3],pred[:,3],jac
        





