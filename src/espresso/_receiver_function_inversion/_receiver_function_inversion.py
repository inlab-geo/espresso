import pathlib
import numpy as np
import matplotlib.pyplot as plt

from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError


LIB_DIR = pathlib.Path(__file__).resolve().parent / "lib"

class ReceiverFunctionInversion(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Receiver function",                # To be used in docs
        "problem_short_description": (
            "'Receiver functions' are a class of seismic data used to study "
            "discontinuities (layering) in the Earth's crust"
        ),    # 1-3 sentences

        "author_names": ["Malcolm Sambridge","Takuo Shibutani"],    # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

        "contact_name": "Malcolm Sambridge",         # Contact for contributor/maintainer of espresso example
        "contact_email": "Malcolm.Sambridge@anu.edu.au",

        "citations": [("Langston, C. A., Structure under Mount Rainer, Washington, inferred from teleseismic body waves, J. Geophys. Res., vol 84, 4749-4762, 1979.", 
                       "Shibutani, T., Kennett, B. and Sambridge, M., Genetic algorithm inversion for receiver functions with application to crust and uppermost mantle structure beneath Eastern Australia, Geophys. Res. Lett., 23 , No. 4, 1829-1832, 1996")], # Reference to publication(s) that describe this example. In most 
                                # cases there will only be a single entry in this list.
                                # List of (citation, doi) pairs e.g. 
                                # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
                                # If there are no citations, use empty list `[]`

        "linked_sites": [],  # List of (title, address) pairs for any websites that 
                                    # should be linked in the documentation, e.g.
                                    # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
                                    #                 ("Data source"),"https://www.data.com") ]
                                    # If there are no links, use empty list `[]`
    }


    def __init__(self, example_number=1):
        super().__init__(example_number)

        from . import rf
        self.rf = rf

        # example initialisation
        if example_number < 4:
            self._ref_model_setup = np.array([[1,4.0,1.7],          # used in example_number=1,2,3
                                            [3.5,4.3,1.7],
                                            [8.0,4.2,2.0],
                                            [20, 6,1.7],
                                            [45,6.2,1.7]])
            # good_model = np.array([1, 4, 3.5, 4.3, 8, 4.2, 20, 6, 45, 6.2])
            # null_model = np.array([1, 4.3, 4, 4.3, 7.5, 4.5, 21, 5, 40, 6.5])
        else:
            self._ref_model_setup = np.array([[8.0,4.2,1.7],          # used in example_number=4
                                            [20, 6,1.7],
                                            [45,6.2,1.7]])
            # good_model = np.array([8., 4.2, 20, 6, 45, 6.2])
            # null_model = np.array([9, 4, 20, 5, 50, 6])

        self._t, self._data = self.rf.rfcalc(self._ref_model_setup, sn=0.2)

        # compute covariance matrix
        self._Cdinv = self.rf.InvDataCov(2.5,0.01,len(self._data))
        # self._Cdinv /= 100        # (potentially we can) temper the likelihood by rescaling the data covariance
        self._Cd = np.linalg.inv(self._Cdinv)

        # example-specific model setup
        if example_number == 1:
            self._description = "Inverting depths of the 2nd and 3rd interfaces"
            self._good_model = np.array([8., 20.])
            self._null_model = np.array([7.5, 21])
            self._interfaces = [2, 3]      # 1st and 2nd interfaces for inversion
            self._nmodel = len(self._interfaces)
        elif example_number == 2:
            self._description = "Inverting velocities of 5 layers"
            self._good_model = np.array([4., 4.3, 4.2, 6., 6.2])
            self._null_model = np.array([4.3, 4.3, 4.5, 5, 6.5])
            self._nmodel = self._ref_model_setup.shape[0]
        elif example_number == 3:
            self._description = "Inverting depths and velocities of 5 layers"
            self._good_model = self._ref_model_setup[:,:2].flatten()
            self._null_model = np.array([1, 4.3, 4, 4.3, 7.5, 4.5, 21, 5, 40, 6.5])
            self._nmodel = len(self._null_model)
        elif example_number == 4:
            self._description = "Inverting dpeths and velocities of 3 layers"
            self._good_model = self._ref_model_setup[:,:2].flatten()
            self._null_model = np.array([9, 4, 20, 5, 50, 6])
            self._nmodel = len(self._null_model)
        else:
            raise InvalidExampleError

    @property
    def description(self):
        return self._description

    @property
    def model_size(self):
        return self._nmodel

    @property
    def data_size(self):
        return len(self._data)

    @property
    def good_model(self):
        return self._good_model

    @property
    def starting_model(self):
        return self._null_model

    @property
    def data(self):
        return self._data

    @property
    def covariance_matrix(self): 
        return self._Cd

    @property
    def inverse_covariance_matrix(self):
        return self._Cdinv

    def _model_setup(self, model):
        model_setup = np.copy(self._ref_model_setup)
        if self.example_number == 1:
            model_setup[self._interfaces,0] = model
        elif self.example_number == 2:
            model_setup[:,1] = model
        elif self.example_number == 3 or self.example_number == 4:
            model_setup[:,:2] = model.reshape(model_setup[:,:2].shape)
        return model_setup

    def forward(self, model, return_jacobian=False, *args, **kwargs):
        if return_jacobian:
            raise NotImplementedError           # optional
        else:
            model_setup = self._model_setup(model)
            _, dpred = self.rf.rfcalc(model_setup, *args, **kwargs)
            return dpred

    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model, model2=None, label=None, label2=None):
        fig, ax = plt.subplots(1, 1, figsize=(4,6))
        ax.set_xlabel('Vs (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.invert_yaxis()
        # process and plot model
        model_setup = self._model_setup(model)
        px = np.zeros([2*len(model_setup),2])
        px[0::2,0] = model_setup[:,1]
        px[1::2,0] = model_setup[:,1]
        px[1::2,1] = model_setup[:,0]
        px[2::2,1] = model_setup[:-1,0]
        ax.plot(px[:,0], px[:,1], 'b-', label=label)
        if model2 is not None:
            model_setup2 = self._model_setup(model2)
            px2 = np.zeros([2*len(model_setup2),2])
            px2[0::2,0] = model_setup2[:,1]
            px2[1::2,0] = model_setup2[:,1]
            px2[1::2,1] = model_setup2[:,0]
            px2[2::2,1] = model_setup2[:-1,0]
            ax.plot(px2[:,0], px2[:,1], 'r-', label=label2)
        return ax
    
    def plot_data(self, data1, data2=None, label=None, label2=None):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self._t,data1,label=label)
        if data2 is not None:
            ax.plot(self._t,data2,'r-',label=label2)
        ax.set_xlabel('Time/s')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        return ax

    def misfit(self, data1, data2):
        return -self.log_likelihood(data1, data2)

    def log_likelihood(self,data1,data2):
        Cdinv = self.inverse_covariance_matrix
        res = data2 - data1
        logLike = -np.dot(res,np.transpose(np.dot(Cdinv, res)))
        return logLike.item()

    def log_prior(self, model):
        if self.example_number == 1:
            depths_increasing = model[0] < model[1]
            depths_in_range = model[0] >= 3.5 and model[1] <= 40
            if depths_increasing and depths_in_range: 
                return np.log(1/(40-3.5)).item()
        elif self.example_number == 2:
            veloc_in_3_7 = all([m_p < 7. and m_p > 3. for m_p in model])
            if veloc_in_3_7: return np.log(1/2).item()
        elif self.example_number == 3:
            depths_in_0_60 = all([m_p < 60 and m_p > 0 for m_p in model[[0,2,4,6,8]]])
            veloc_in_3_7 = all([m_p < 7. and m_p > 3. for m_p in model[[1,3,5,7,9]]])
            params_increasing = all([model[i] < model[i+2] for i in range(0,len(model)-2,2)])
            if depths_in_0_60 and veloc_in_3_7 and params_increasing:
                return np.log(1/60).item()
        elif self.example_number == 4:
            depths_in_0_60 = all([m_p < 60 and m_p > 0 for m_p in model[[0,2,4]]])
            veloc_in_3_7 = all([m_p < 7. and m_p > 3. for m_p in model[[1,3,5]]])
            params_increasing = all([model[i] < model[i+2] for i in range(0,len(model)-2,2)])
            if depths_in_0_60 and veloc_in_3_7 and params_increasing:
                return np.log(1/60).item()
        return float("-inf")


# 37 EARTH SCIENCES -> 3706	Geophysics -> 370609 Seismology and seismic exploration -> Receiver function -> ReceiverFunctionInversion
# description: 'Receiver functions' are a class of seismic data used to study discontinuities (layering) in the Earth's crust.
