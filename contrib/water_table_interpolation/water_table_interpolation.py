import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from os import chdir
chdir('../../src')
from cofi_espresso import EspressoProblem, InvalidExampleError
chdir('../contrib/water_table_interpolation')

class WaterTableInterpolation(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Water table interpolation",
        "problem_short_description": "This example involves the matching of an interpolated" \
                                "potentiometric surface to hydraulic head data obtained over" \
                                "a single continuous aquifer.",

        "author_names": ["Chris Turnadge"],

        "contact_name": "Chris Turnadge",
        "contact_email": "chris.turnadge@csiro.au",

        "citations": [],

        "linked_sites": [],
    }
    

    def __init__(self):
        self._xp = np.atleast_2d([268692.830, 268980.100, 268211.780, 269532.130, 268422.800, 
                                  268941.810, 270906.820, 268342.750, 271897.130, 272417.800, 
                                  268262.820, 270438.320, 268703.740, 273639.790, 274713.720, 
                                  272506.290, 275128.320, 275801.930, 273473.760, 277367.760, 
                                  277537.720, 275943.740, 277413.040, 279007.250, 276962.960, 
                                  277428.350, 277896.780, 277896.720, 276053.770, 277727.790, 
                                  277730.770, 280116.720, 280044.600, 279613.750, 279504.120, 
                                  280074.770, 282211.770, 279696.340, 280938.750, 274760.800, 
                                  282058.690, 278422.620, 280857.310, 282055.700, 281209.660, 
                                  278350.780, 281253.760, 281239.490, 281855.730, 284219.310, 
                                  283739.030, 279772.730, 280090.750, 284454.060, 280071.750, 
                                  283720.720, 277790.800, 284885.700, 285247.330, 284757.740, 
                                  286977.760, 285955.350, 284228.990, 286225.300, 285787.700, 
                                  285126.700, 284627.730]).T
        self._yp = np.atleast_2d([6090381.570, 6091011.040, 6090420.490, 6091667.150, 6088616.530, 
                                  6093083.520, 6096136.560, 6089793.530, 6092138.110, 6092324.500, 
                                  6090831.550, 6089954.640, 6091952.550, 6092481.480, 6099737.570, 
                                  6097255.780, 6101288.740, 6093895.200, 6090244.520, 6096220.520, 
                                  6098076.460, 6099024.540, 6097596.190, 6097347.300, 6099957.870, 
                                  6101578.420, 6099396.560, 6099395.550, 6102164.520, 6100638.500, 
                                  6100639.570, 6098418.560, 6098728.490, 6100153.550, 6103716.500, 
                                  6100037.460, 6100862.530, 6101840.750, 6101721.530, 6104373.480, 
                                  6102406.470, 6104055.050, 6103228.910, 6104194.560, 6105755.300, 
                                  6094351.480, 6105752.370, 6105754.580, 6106253.500, 6104842.780, 
                                  6105457.000, 6096025.560, 6096736.500, 6105562.620, 6096593.290, 
                                  6108522.450, 6093805.560, 6107988.550, 6107931.760, 6108702.510, 
                                  6107901.460, 6109203.720, 6102553.080, 6110008.680, 6105829.250, 
                                  6104832.290, 6103416.130]).T
        self._zp = np.atleast_2d([0.450, 0.500, 0.550, 0.980, 1.120, 4.090, 4.330, 4.550, 5.540, 
                                  9.010, 10.150, 15.900, 16.110, 17.540, 18.120, 18.220, 34.150, 
                                  34.940, 35.770, 40.120, 48.430, 48.780, 50.210, 51.760, 52.740, 
                                  53.830, 54.340, 55.440, 55.520, 55.580, 55.680, 61.130, 61.810, 
                                  63.480, 63.870, 65.000, 70.070, 70.970, 72.420, 72.620, 77.200, 
                                  82.420, 86.420, 106.910, 128.990, 129.420, 131.490, 131.850, 
                                  136.000, 139.640, 142.080, 142.260, 143.980, 145.320, 147.470, 
                                  150.520, 152.490, 161.250, 166.650, 167.700, 175.140, 176.090, 
                                  184.260, 194.240, 200.480, 204.770, 208.710]).T
        self._m = np.array([2.])
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
        from pyinterpolate import inverse_distance_weighting
        result = inverse_distance_weighting(np.hstack([self._xp, self._yp, self._zp]),
                                            np.hstack([self._xp, self._yp, self._zp]),
                                            number_of_neighbours=np.int(model[0]),
                                            power=model[1])
        return result
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self):
        plt.errorbar(self._xp, self._yp, yerr=self._sigma, fmt='.', color="lightcoral", ecolor='lightgrey', ms=10)
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.show()

    def misfit(self, data, data2):              # optional
        raise NotImplementedError

    def log_likelihood(self, model):
        y_synthetics = self.forward(model)
        residual = self.data - y_synthetics
        return -0.5 * residual @ self.inverse_covariance_matrix @ residual.T
    
    def log_prior(self, model):
        m_lower_bound = np.log(np.array([2, len(self._xp)]))  # lower bound for uniform prior
        m_upper_bound = np.log(np.array([1, 10]))             # upper bound for uniform prior
        for i in range(len(m_lower_bound)):
            if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
        return 0.0 # model lies within bounds -> return log(1)

