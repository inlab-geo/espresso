import numpy as np
import matplotlib.pyplot as plt

# from os import chdir
# chdir('../../src')
from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
# chdir('../contrib/pumping_test')


class SinusoidalTest(EspressoProblem):
    """Forward simulation class
    """ 

    metadata = {
        "problem_title": "Aquifer sinusoidal hydraulic test",
        "problem_short_description": "This example involves the matching of "\
                                     "an appropriate forward model to "\
                                     "time-drawdown data obtained from a " \
                                     "two-well sinusoidal hydraulic test.",

        "author_names": ["Chris Turnadge"],

        "contact_name": "Chris Turnadge",
        "contact_email": "chris.turnadge@csiro.au",

        "citations": [],

        "linked_sites": [],
    }
    

    def __init__(self, example_number=1):
        super().__init__(example_number)
        self._xp = xp.copy()
        if self.example_number == 1:
            self._yp = yp1.copy()
            self._m = np.ones(4)
            #self._basis = ''
            self._desc = "Rasmussen et al. (2003) confined aquifer "\
                         "sinusoidal pumping test solution"
            self._sigma = 0.1
        # elif self.example_number == 2:
        #     self._yp = yp2.copy()
        #     self._m = np.ones(7)
        #     #self._basis = ''
        #     self._desc = "Rasmussen et al. (2003) leaky aquifer sinusoidal "\
        #                  "pumping test solution excluding aquitard storage"
        #     self._sigma = 0.1
        #     self._sigma = 0.1
        # elif self.example_number == 3:
        #     self._yp = yp3.copy()
        #     self._m = np.ones(6)
        #     #self._basis = ''
        #     self._desc = "Barker (1988) fractured rock aquifer sinusoidal "\
        #                  "pumping test solution"
        #     self._sigma = 0.1
        else:
            raise InvalidExampleError

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
        return self._sigma**2.*np.eye(self.data_size)

    @property
    def inverse_covariance_matrix(self):
        return 1./self._sigma**2.*np.eye(self.data_size)
        
    def forward(self, model, with_jacobian=False):
        if with_jacobian:
            raise NotImplementedError
        if self.example_number == 1:
            from mpmath import pi, sqrt, exp, besselk
            times = self.xp1
            r, Q0, T, S, w = self.m
            return np.array([float(np.real(Q0*exp(1j*w*t)/(2.*pi*T)*
                             besselk(0, r*sqrt(1j*w*S/T)))) if t!=0. else 0. 
                             for t in times])
        elif self.example_number == 2:
            from mpmath import pi, sqrt, besselk
            times = self.xp2
            r, Q0, K, b, Ss, Kp, bp, w = self.m
            return np.array([float(np.real(Q0*exp(1j*w*t)/(2.*pi*K*b)*
                             besselk(0, r*sqrt(1j*w*Ss/K+Kp*b/(K*bp))))) 
                             if t!=0. else 0. for t in times])
        elif self.example_number == 3:
            from mpmath import pi, sqrt, gamma, besselk
            times = self.xp3
            r, rw, Q0, K, b, Ss, Sw, n, w = self.m
            return np.array([float(np.real(Q0*(r/rw)**(1.-n/2.)/
                            (besselk(1.-n/2., rw*sqrt(Ss/K))**(1j*w)*Sw*
                            (1.+Ss*((rw*sqrt(Ss/K))*besselk(1.-n/2.-1., 
                            rw*sqrt(Ss/K))/besselk(1.-n/2., rw*sqrt(Ss/K))+
                            K*b**(3.-n)*(2.*pi**(n/2.)/gamma(n/2.))))))) 
                            if t!=0. else 0. for t in times])
        else:
            raise InvalidExampleError
           
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self, data):
        plt.errorbar(self._xp, data, yerr=self._sigma, fmt='.', 
                     color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.show()

    def misfit(self, data, data2):              # optional
        raise NotImplementedError

    def log_likelihood(self, data, data2): 
        residual = data - data2
        return (-0.5*residual@self.inverse_covariance_matrix@residual.T).item()
    
    def log_prior(self, model):
        raise NotImplementedError

# The following arrays define the 'sampling points' used in various examples
# In principle we could generate these on-the-fly but using hard-coded values
# guarantees repeatibility across different machines, architectures, prngs.

xp = np.array([])

# The following arrays define the 'data' at these sampling points.

yp1 = np.array([])

yp2 = np.array([])

yp3 = np.array([])
