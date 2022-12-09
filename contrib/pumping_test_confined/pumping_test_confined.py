import numpy as np
from cofi_espresso import EspressoProblem, InvalidExampleError


class PumpingTestConfined(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Confined aquifer constant rate discharge pumping test",
        "problem_short_description": "This example involves the matching of an appropriate" \
                                "forward model to time-drawdown data obtained from a" \
                                "constant rate discharge test undertaken in a confined aquifer.",

        "author_names": ["Chris Turnadge"],

        "contact_name": "Chris Turnadge",
        "contact_email": "chris.turnadge@csiro.au",

        "citations": [],

        "linked_sites": [],
    }
    

    def __init__(self):
        self._xp = np.array([ 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 
                             12.0, 14.0, 18.0, 24.0, 30.0, 40.0, 50.0, 60.0, 
                             80.0, 100.0, 120.0, 150.0, 180.0, 210.0, 240.0])
        self._yp = np.array([0.20, 0.27, 0.30, 0.34, 0.37, 0.41, 0.45, 0.48, 
                             0.53, 0.57, 0.60, 0.63, 0.67, 0.72, 0.76, 0.81, 
                             0.85, 0.88, 0.93, 0.96, 1.00, 1.04, 1.07, 1.10, 1.12])
        self._m = np.array([1.0, 1e-4])
        self._sigma = 0.1

    @property
    def description(self):
        return self._desc

    @property
    def model_size(self):
        return self._m.size

    @property
    def data_size(self):
        return self._yp.size

    @property
    def good_model(self):
        return self._m.copy()

    @property
    def starting_model(self):
        return np.zeros_like(self._m)
    
    @property
    def data(self):
        return self._yp.copy()

    @property
    def covariance_matrix(self):
        return self._sigma**2 * np.eye(self.data_size)

    @property
    def inverse_covariance_matrix(self):
        return 1./self._sigma**2 * np.eye(self.data_size)
        
    def forward(self, model):
        from scipy.special import expi
        Q,r = 31.0/1000.*86400., 61.0
        result = np.array([Q/4./np.pi/10.**model[0]*-expi(r**2.*10.**model[1]/4./10.**model[0]/t) for t in data_x/1440.])
        result[result==-np.inf] = 0.
        result[result== np.inf] = 0.
        return result
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self):
        plt.errorbar(self._xp, self._yp, yerr=self._sigma, fmt='.', color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")

    def misfit(self, data, data2):              # optional
        raise NotImplementedError

    def log_likelihood(self, model):
        y_synthetics = forward(model)
        residual = self._yp - y_synthetics
        return -0.5 * residual @ inverse_covariance_matrix(self) @ residual.T
    
    def log_prior(self, model):
        for i in range(len(m_lower_bound)):
            if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
        return 0.0 # model lies within bounds -> return log(1)

