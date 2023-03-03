import numpy as np
import matplotlib.pyplot as plt

# from os import chdir
# chdir('../../src')
from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
# chdir('../contrib/pumping_test')


class PumpingTest(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Aquifer constant rate discharge pumping test",
        "problem_short_description": "This example involves the matching of "\
                                     "an appropriate forward model to "\
                                     "time-drawdown data obtained from a " \
                                     "two-well constant rate discharge test.",

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
            self._m_fixed = np.array([10., 432.])
            self._m = np.array([1., 1e-4])
            self._desc = "Theis (1935) confined aquifer pumping test solution"
            self._sigma = 0.1
        elif self.example_number == 2:
            self._yp = yp2.copy()
            self._m_fixed = np.array([10., 432.])
            self._m = np.array([0.1, 10., 1e-5, 1e-3, 1.6])
            self._desc = "Hantush and Jacob (1955) leaky aquifer pumping "\
                         "test solution excluding aquitard storage"
            self._sigma = 0.1
        elif self.example_number == 3:
            self._yp = yp3.copy()
            self._m_fixed = np.array([10., 432.])
            self._m = np.array([0.1, 10., 1e-5, 1e-3, 1.6, 1e-10])
            self._desc = "Hantush (1960) leaky aquifer pumping test solution "\
                         "including aquitard storage"
            self._sigma = 0.1
        elif self.example_number == 4:
            self._yp = yp4.copy()
            self._m_fixed = np.array([10., 432.])
            self._m = np.array([0.1, 10., 1e-5, 2.])
            self._desc = "Barker (1988) fractured rock aquifer pumping test "\
                         "solution"
            self._sigma = 0.1
        #elif self.example_number == 5:
        #    self._yp = yp5.copy()
        #    self._m_fixed = np.array([10., 432.])
        #    self._m = np.ones(2)
        #    self._desc = "Neuman (1974) unconfined aquifer pumping test "\
        #                 "solution"
        #    self._sigma = 0.1
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
            from mpmath import pi, sqrt, besselk, invertlaplace
            times = self._xp
            r, Q = self._m_fixed
            T, S = model
            fp = lambda p: Q/(2.*pi*T*p)*besselk(0, r*sqrt(p*S/T))
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else 0. for t in times])
        elif self.example_number == 2:
            from mpmath import pi, sqrt, besselk, invertlaplace
            times = self._xp
            r, Q = self._m_fixed
            K, b, Ss, Kp, bp = model
            fp = lambda p: Q/(2.*pi*K*b*p)*besselk(0, r*sqrt(p*Ss/K+1./
                                                            (K*b*bp/Kp))) 
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else 0. for t in times])
        elif self.example_number == 3:
            from mpmath import pi, sqrt, coth, besselk, invertlaplace
            times = self._xp
            r, Q = self._m_fixed
            K, b, Ss, Kp, bp, Ssp = model
            fp = lambda p: Q/(2.*pi*K*b*p)*besselk(0, r*sqrt(p*Ss/K+1./
                           (K*b*bp/Kp)*sqrt(p*Ssp/(Ss*b)*(K*b*bp/Kp))*
                           coth(sqrt(p*Ssp/(Ss*b)*(K*b*bp/Kp)))))
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else 0. for t in times])
        elif self.example_number == 4:
            from mpmath import pi, sqrt, gamma, besselk, invertlaplace
            times = self._xp
            r, Q = self._m_fixed
            K, b, Ss, n = model
            fp = lambda p: (Q*r**(1.-n/2.)*besselk(1.-n/2., r*sqrt(p*Ss/K))/
                           (p*K*b**(3.-n)*(2.*pi**(n/2.)/gamma(n/2.))*
                           (r*sqrt(p*Ss/K))**(1.-n/2.)*2.**(-(1.-n/2.))*
                           gamma(1.-(1.-n/2.))))
            return np.array([float(invertlaplace(fp, t, method='dehoog')) 
                             if t!=0. else 0. for t in times])
        #elif self.example_number == 5:
        #    pass
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
        plt.ylabel('Drawdown')
    
    def plot_data(self, data):
        plt.errorbar(self._xp, data, yerr=self._sigma, fmt='.', 
                     color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.xlabel('Time elapsed')
        plt.ylabel('Drawdown')

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
xp = np.array([0.00100000, 0.00100803, 0.00102428, 0.00105756, 0.00112741,
               0.00120805, 0.00129444, 0.00138702, 0.00148622, 0.00159251,
               0.00170641, 0.00182845, 0.00195922, 0.00209934, 0.00224949,
               0.00241037, 0.00258276, 0.00276747, 0.00296540, 0.00317748,
               0.00340474, 0.00364824, 0.00390916, 0.00418874, 0.00448832,
               0.00480932, 0.00515328, 0.00552183, 0.00591675, 0.00633991,
               0.00679334, 0.00727919, 0.00779980, 0.00835763, 0.00895537,
               0.00959585, 0.01028210, 0.01101750, 0.01180550, 0.01264980,
               0.01355450, 0.01452390, 0.01556260, 0.01667570, 0.01786830,
               0.01914620, 0.02051560, 0.02198280, 0.02355500, 0.02523970,
               0.02704480, 0.02897900, 0.03105160, 0.03327230, 0.03565200,
               0.03820180, 0.04093390, 0.04386150, 0.04699840, 0.05035970,
               0.05396140, 0.05782070, 0.06195600, 0.06638700, 0.07113500,
               0.07622250, 0.08167390, 0.08751520, 0.09377420, 0.10048100,
               0.10766700, 0.11536700, 0.12361800, 0.13246000, 0.14193300,
               0.15208400, 0.16296100, 0.17461600, 0.18710400, 0.20048600,
               0.21482400, 0.23018800, 0.24665100, 0.26429200, 0.28319400,
               0.30344700, 0.32515000, 0.34840400, 0.37332200, 0.40002100,
               0.42863100, 0.45928600, 0.49213400, 0.52733100, 0.56504500,
               0.60545700, 0.64875900, 0.69515800, 0.74487500, 0.79814800,
               0.85523100, 0.91639600, 0.98193600, 1.00000000])

# The following arrays define the 'data' at these sampling points.
yp1 = np.array([  0.856510,   0.879297,   0.926247,   1.025790,   1.248680, 
                  1.527600,   1.849360,   2.217030,   2.633440,   3.101130,
                  3.622390,   4.199140,   4.833020,   5.525300,   6.276920,
                  7.088530,   7.960410,   8.892580,   9.884780,  10.936500,
                 12.046900,  13.215200,  14.440000,  15.720200,  17.054300,
                 18.440700,  19.877900,  21.364000,  22.897300,  24.476100,
                 26.098500,  27.762600,  29.466700,  31.209000,  32.987700,
                 34.801000,  36.647200,  38.524700,  40.431900,  42.367200,
                 44.329100,  46.316200,  48.327100,  50.360400,  52.414800,
                 54.489300,  56.582500,  58.693400,  60.821000,  62.964300,
                 65.122200,  67.294000,  69.478800,  71.675800,  73.884200,
                 76.103300,  78.332500,  80.571200,  82.818600,  85.074400,
                 87.337900,  89.608600,  91.886100,  94.170000,  96.459900,
                 98.755300, 101.056000, 103.361000, 105.672000, 107.986000,
                110.304000, 112.626000, 114.952000, 117.280000, 119.612000,
                121.947000, 124.284000, 126.624000, 128.966000, 131.310000,
                133.656000, 136.005000, 138.355000, 140.706000, 143.059000,
                145.414000, 147.770000, 150.127000, 152.485000, 154.844000,
                157.205000, 159.566000, 161.928000, 164.292000, 166.655000,
                169.020000, 171.385000, 173.751000, 176.118000, 178.485000,
                180.852000, 183.220000, 185.589000, 186.214000])

yp2 = np.array([  0.852267,   0.874912,   0.921564,   1.020470,   1.241830,
                  1.518720,   1.837980,   2.202580,   2.615260,   3.078480,
                  3.594380,   4.164800,   4.791220,   5.474770,   6.216260,
                  7.016120,   7.874510,   8.791230,   9.765810,  10.797500,
                 11.885400,  13.028100,  14.224400,  15.472600,  16.771000,
                 18.117800,  19.510800,  20.948200,  22.427700,  23.947200,
                 25.504300,  27.096900,  28.722600,  30.379100,  32.064100,
                 33.775300,  35.510400,  37.267000,  39.043000,  40.835900,
                 42.643600,  44.463900,  46.294300,  48.132800,  49.977100,
                 51.825100,  53.674400,  55.522900,  57.368300,  59.208600,
                 61.041400,  62.864500,  64.675700,  66.472800,  68.253400,
                 70.015400,  71.756400,  73.474100,  75.166200,  76.830400,
                 78.464400,  80.066000,  81.632600,  83.162200,  84.652500,
                 86.101100,  87.506100,  88.865200,  90.176500,  91.438100,
                 92.648200,  93.805200,  94.907600,  95.954200,  96.943900,
                 97.875900,  98.749700,  99.564900, 100.322000, 101.020000,
                101.662000, 102.247000, 102.777000, 103.254000, 103.680000,
                104.057000, 104.387000, 104.675000, 104.923000, 105.134000,
                105.311000, 105.459000, 105.580000, 105.678000, 105.756000,
                105.818000, 105.865000, 105.900000, 105.927000, 105.946000,
                105.960000, 105.969000, 105.975000, 105.977000])

yp3 = np.array([  0.839321,   0.861653,   0.907660,   1.005190,   1.223450,
                  1.496460,   1.811240,   2.170790,   2.577850,   3.034910,
                  3.544170,   4.107500,   4.726480,   5.402290,   6.135820,
                  6.927590,   7.777810,   8.686370,   9.652860,  10.676600,
                 11.756700,  12.891900,  14.080800,  15.322000,  16.613700,
                 17.954000,  19.340900,  20.772500,  22.246600,  23.760900,
                 25.313200,  26.901300,  28.522800,  30.175500,  31.856900,
                 33.564800,  35.297000,  37.050900,  38.824500,  40.615400,
                 42.421300,  44.240000,  46.069200,  47.906700,  49.750400,
                 51.597900,  53.447100,  55.295700,  57.141600,  58.982600,
                 60.816400,  62.640800,  64.453500,  66.252400,  68.035200,
                 69.799500,  71.543200,  73.263900,  74.959200,  76.627000,
                 78.264900,  79.870500,  81.441600,  82.975900,  84.471100,
                 85.925000,  87.335400,  88.700300,  90.017600,  91.285400,
                 92.501900,  93.665500,  94.774700,  95.828200,  96.825000,
                 97.764100,  98.645000,  99.467500, 100.231000, 100.937000,
                101.586000, 102.178000, 102.714000, 103.198000, 103.630000,
                104.013000, 104.349000, 104.642000, 104.895000, 105.110000,
                105.292000, 105.444000, 105.568000, 105.669000, 105.749000,
                105.813000, 105.861000, 105.898000, 105.925000, 105.945000,
                105.959000, 105.968000, 105.975000, 105.976000])

yp4 = np.array([  0.012805,   0.882463,   0.930532,   1.032340,   1.259800, 
                  1.543600,   1.870010,   2.241950,   2.662140,   3.133020, 
                  3.656820,   4.235430,   4.870470,   5.563240,   6.314720, 
                  7.125590,   7.996230,   8.926730,   9.916890,  10.966300, 
                 12.074200,  13.239800,  14.462000,  15.739500,  17.071000, 
                 18.454900,  19.889600,  21.373500,  22.904800,  24.481700, 
                 26.102400,  27.765000,  29.467800,  31.209000,  32.986600, 
                 34.799100,  36.644700,  38.521600,  40.428400,  42.363300, 
                 44.325000,  46.311900,  48.322600,  50.355800,  52.410200, 
                 54.484600,  56.577800,  58.688700,  60.816300,  62.959600, 
                 65.117600,  67.289400,  69.474300,  71.671200,  73.879700, 
                 76.098800,  78.328000,  80.566600,  82.814100,  85.069800, 
                 87.333300,  89.604000,  91.881500,  94.165400,  96.455200, 
                 98.750600, 101.051000, 103.357000, 105.667000, 107.981000, 
                110.299000, 112.621000, 114.947000, 117.275000, 119.607000, 
                121.942000, 124.279000, 126.619000, 128.961000, 131.305000,
                133.651000, 135.999000, 138.349000, 140.700000, 143.054000, 
                145.408000, 147.764000, 150.121000, 152.479000, 154.839000, 
                157.199000, 159.560000, 161.923000, 164.286000, 166.649000, 
                169.014000, 171.379000, 173.745000, 176.112000, 178.479000, 
                180.846000, 183.214000, 185.583000, 186.208000])

#yp5 = np.array([])
