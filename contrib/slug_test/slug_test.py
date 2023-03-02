import numpy as np
import matplotlib.pyplot as plt

# from os import chdir
# chdir('../../src')
from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
# chdir('../contrib/slug_test')


class SlugTest(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Aquifer slug test",
        "problem_short_description": "This example involves the matching of "\
                                     "an appropriate forward model to "\
                                     "time-displacement data obtained from "\
                                     "a traditional single well slug test.",

        "author_names": ["Chris Turnadge"],

        "contact_name": "Chris Turnadge",
        "contact_email": "chris.turnadge@csiro.au",

        "citations": [],

        "linked_sites": [],
    }
    

    def __init__(self, example_number=1):
        super().__init__(example_number)
        if self.example_number == 1:
            self._xp = xp1.copy()
            self._yp = yp1.copy()
            self._m_fixed = np.array([0.1, 1.])
            self._m = np.array([0.1, 1e-3])
            self._desc = "Cooper-Bredehoeft-Papadopulos (1967) confined "\
                         "aquifer slug test solution"
            self._sigma = 0.1
        elif self.example_number == 2:
            self._xp = xp2.copy()
            self._yp = yp2.copy()
            self._m_fixed = np.array([0.1, 10., 1.])
            self._m = np.array([0.01, 0.003])
            self._desc = "Hvorslev (1951) confined aquifer slug test "\
                         "solution"
            self._sigma = 0.1
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
        return np.ones_like(self._m)
    
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
            from mpmath import sqrt, besselk, invertlaplace
            times = self._xp
            rw, H0 = self._m_fixed
            T, S = model
            fp = lambda p: (H0*rw*S*besselk(0, rw*sqrt(p*S/T))/
                           ((T*sqrt(p*S/T))*(rw*sqrt(p*S/T)*
                            besselk(0, rw*sqrt(p*S/T))+2*(rw**2.)*S/(rw**2.)*
                            besselk(1, rw*sqrt(p*S/T)))))
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else H0 for t in times])
        elif self.example_number == 2:
            from mpmath import sqrt, asinh, besselk, invertlaplace
            times = self._xp
            rw, L, H0 = self._m_fixed
            Kr, Kz = model
            fp = lambda p: H0/(p+(2.*Kr*L)/(rw**2.*asinh(L/(2.*rw*
                           sqrt(Kz/Kr)))))
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else H0 for t in times])
        else:
            raise InvalidExampleError
           
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        plt.errorbar(self._xp, self.forward(model), yerr=self._sigma, fmt='.', 
                     color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.xlabel('Time elapsed')
        plt.ylabel('Displacement')
    
    def plot_data(self, data):
        plt.errorbar(self._xp, data, yerr=self._sigma, fmt='.', 
                     color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.xlabel('Time elapsed')
        plt.ylabel('Displacement')

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

xp1 = np.array([0.0100000, 0.0104713, 0.0109648, 0.0114815, 0.0120226, 
                0.0125893, 0.0131826, 0.0138038, 0.0144544, 0.0151356, 
                0.0158489, 0.0165959, 0.0173780, 0.0181970, 0.0190546, 
                0.0199526, 0.0208930, 0.0218776, 0.0229087, 0.0239883, 
                0.0251189, 0.0263027, 0.0275423, 0.0288403, 0.0301995, 
                0.0316228, 0.0331131, 0.0346737, 0.0363078, 0.0380189, 
                0.0398107, 0.0416869, 0.0436516, 0.0457088, 0.0478630, 
                0.0501187, 0.0524807, 0.0549541, 0.0575440, 0.0602560, 
                0.0630957, 0.0660693, 0.0691831, 0.0724436, 0.0758578, 
                0.0794328, 0.0831764, 0.0870964, 0.0912011, 0.0954993, 
                0.1000000, 0.1047130, 0.1096480, 0.1148150, 0.1202260, 
                0.1258930, 0.1318260, 0.1380380, 0.1445440, 0.1513560, 
                0.1584890, 0.1659590, 0.1737800, 0.1819700, 0.1905460, 
                0.1995260, 0.2089300, 0.2187760, 0.2290870, 0.2398830, 
                0.2511890, 0.2630270, 0.2754230, 0.2884030, 0.3019950, 
                0.3162280, 0.3311310, 0.3467370, 0.3630780, 0.3801890, 
                0.3981070, 0.4168690, 0.4365160, 0.4570880, 0.4786300, 
                0.5011870, 0.5248070, 0.5495410, 0.5754400, 0.6025600, 
                0.6309570, 0.6606930, 0.6918310, 0.7244360, 0.7585780, 
                0.7943280, 0.8317640, 0.8709640, 0.9120110, 0.9549930, 
                1.0000000])

xp2 = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 
                0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 
                0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 
                0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 
                0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 
                0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 
                0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 
                0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 
                0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 
                1.00])
    
# The following arrays define the 'data' at these sampling points.

yp1 = np.array([0.917543, 0.914595, 0.911542, 0.908382, 0.905109, 
                0.901722, 0.898216, 0.894588, 0.890834, 0.886949, 
                0.882931, 0.878775, 0.874478, 0.870034, 0.865440, 
                0.860693, 0.855787, 0.850718, 0.845483, 0.840077, 
                0.834496, 0.828735, 0.822791, 0.816659, 0.810335, 
                0.803816, 0.797097, 0.790174, 0.783045, 0.775704, 
                0.768150, 0.760379, 0.752387, 0.744172, 0.735733, 
                0.727066, 0.718170, 0.709043, 0.699685, 0.690096, 
                0.680274, 0.670220, 0.659936, 0.649423, 0.638683, 
                0.627719, 0.616535, 0.605136, 0.593525, 0.581710, 
                0.569697, 0.557493, 0.545108, 0.532551, 0.519832, 
                0.506964, 0.493958, 0.480827, 0.467587, 0.454253, 
                0.440841, 0.427367, 0.413852, 0.400312, 0.386768, 
                0.373240, 0.359749, 0.346317, 0.332964, 0.319714, 
                0.306587, 0.293607, 0.280794, 0.268172, 0.255760, 
                0.243579, 0.231650, 0.219990, 0.208618, 0.197551, 
                0.186803, 0.176388, 0.166319, 0.156606, 0.147257, 
                0.138280, 0.129681, 0.121461, 0.113623, 0.106167, 
                0.099090, 0.092389, 0.086058, 0.080091, 0.074478, 
                0.069211, 0.064279, 0.059670, 0.055372, 0.051372, 
                0.047655])

yp2 = np.array([1.0000000, 0.9625890, 0.9265770, 0.8919120, 0.8585450, 
                0.8264250, 0.7955080, 0.7657470, 0.7370990, 0.7095230, 
                0.6829790, 0.6574280, 0.6328330, 0.6091570, 0.5863680,
                0.5644310, 0.5433150, 0.5229890, 0.5034230, 0.4845890,
                0.4664600, 0.4490090, 0.4322110, 0.4160420, 0.4004770,
                0.3854950, 0.3710730, 0.3571900, 0.3438270, 0.3309640,
                0.3185830, 0.3066640, 0.2951910, 0.2841480, 0.2735170,
                0.2632850, 0.2534350, 0.2439540, 0.2348270, 0.2260420,
                0.2175850, 0.2094450, 0.2016090, 0.1940670, 0.1868070,
                0.1798180, 0.1730910, 0.1666150, 0.1603820, 0.1543820,
                0.1486060, 0.1430470, 0.1376950, 0.1325440, 0.1275850,
                0.1228120, 0.1182170, 0.1137950, 0.1095370, 0.1054390,
                0.1014950, 0.0976978, 0.0940428, 0.0905245, 0.0871378,
                0.0838779, 0.0807399, 0.0777193, 0.0748117, 0.0720129,
                0.0693188, 0.0667255, 0.0642292, 0.0618263, 0.0595133,
                0.0572868, 0.0551437, 0.0530807, 0.0510948, 0.0491833,
                0.0473433, 0.0455721, 0.0438672, 0.0422261, 0.0406463,
                0.0391257, 0.0376620, 0.0362530, 0.0348967, 0.0335912,
                0.0323345, 0.0311248, 0.0299604, 0.0288395, 0.0277606,
                0.0267220, 0.0257223, 0.0247600, 0.0238337, 0.0229421,
                0.0220838])
