import numpy as np
from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError


class SimpleRegression(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "Simple 1D regression",
        "problem_short_description": "This example includes various small 1D regression (curve-fitting) " \
                                "problems. Several different forms of basis function are supported: " \
                                "polynomial, Fourier and discrete.",

        "author_names": ["Andrew Valentine"],

        "contact_name": "Andrew Valentine",
        "contact_email": "andrew.valentine@durham.ac.uk",

        "citations": [],

        "linked_sites": [],
    }
    

    def __init__(self, example_number=1):
        super().__init__(example_number)
        if self.example_number == 1:
            self._xp = xp1.copy()
            self._yp = yp1.copy()
            self._m = np.array([0.5,2])
            self._basis = 'polynomial'
            self._desc = "Fitting a straight line to data"
            self._sigma = 0.1
        elif self.example_number == 2:
            self._xp = xp2.copy()
            self._yp = yp2.copy()
            self._m = np.array([1.,-0.2,0.5,0.3])
            self._basis = 'polynomial'
            self._desc = "Fitting a cubic polynomial to data"
            self._sigma = 0.1
        elif self.example_number == 3:
            self._xp = xp3.copy()
            self._yp = yp3.copy()
            self._m = np.array([1,0,0.3,-0.2,0,0,0.7,0.,0.,0.3,0.,0.,0.,0.,-.2])
            self._basis = 'fourier'
            self._desc = "Fitting a Fourier series to data"
            self._sigma = 0.1
        elif self.example_number == 4:
            self._xp = xp4.copy()
            self._yp = yp4.copy()
            self._m = np.array([0.,2,-.3,-.6])
            self._basis = 'polynomial'
            self._desc = "Fitting a polynomial to a very small dataset"
            self._sigma = 0.05
        elif self.example_number == 5:
            self._xp = xp5.copy()
            self._yp = yp5.copy()
            self._m = np.array([0.3,0.3,0.,-0.2,0.5,-.8,0.1,0.125])
            self._basis = 'polynomial'
            self._desc = "Fitting a polynomial to a dataset with a gap"
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
        return np.zeros_like(self._m)
    
    @property
    def data(self):
        return self._yp.copy()

    @property
    def covariance_matrix(self):
        return self._sigma**2 * np.eye(self.data_size)

    @property
    def inverse_covariance_matrix(self):
        return (1/self._sigma**2) * np.eye(self.data_size)
        
    def forward(self, model, with_jacobian=False):
        d = curveFittingFwd(self._m,self._xp,self._basis)
        if with_jacobian:
            return d, self.jacobian(model)
        else:
            return d
    
    def jacobian(self, model):
        nModel = model.shape[0]
        return curveFittingJac(self._xp,nModel,self._basis)

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

# The following arrays define the 'sampling points' used in various examples
# In principle we could generate these on-the-fly but using hard-coded values
# guarantees repeatibility across different machines, architectures, prngs.

# The first four are uniformly-distributed points between 0 and 1
xp1 = np.array([0.02814768, 0.0885654 , 0.11748064, 0.12294414, 0.16288759,
                0.16877703, 0.1843951 , 0.39127594, 0.46038946, 0.4914394 ,
                0.54084596, 0.5641636 , 0.58900649, 0.58976937, 0.6547135 ,
                0.85518813, 0.86779304, 0.91481368, 0.951955  , 0.98545064])
xp2 = np.array([0.02061079, 0.02454563, 0.19277473, 0.2784715 , 0.29414602,
                0.32944733, 0.42601344, 0.42966824, 0.45568706, 0.4727606 ,
                0.47873683, 0.50393437, 0.59354021, 0.60901867, 0.68636357,
                0.70977542, 0.73149133, 0.78673413, 0.82707107, 0.84599549,
                0.88076936, 0.9171245 , 0.95712395, 0.97514086, 0.99942137])
xp3 = np.array([0.02129797, 0.0680587 , 0.07325879, 0.07540233, 0.0815562 ,
                0.11653215, 0.13302591, 0.17036322, 0.17948113, 0.18668185,
                0.22055489, 0.22751703, 0.26470851, 0.31516965, 0.34038959,
                0.34051367, 0.34721832, 0.35517515, 0.36644436, 0.42221368,
                0.4780765 , 0.50384201, 0.54011153, 0.56945004, 0.65217677,
                0.65762461, 0.66908502, 0.68851014, 0.71684459, 0.717921  ,
                0.72093096, 0.73367274, 0.73389493, 0.75033087, 0.75890497,
                0.76225345, 0.76552936, 0.83833901, 0.86436217, 0.88042872,
                0.90603222, 0.91094849, 0.9336106 , 0.9400528 , 0.96037445,
                0.965113  , 0.96609565, 0.97968897, 0.98044997, 0.99266562])
xp4 = np.array([0.17833636, 0.19050344, 0.26430464, 0.51520092, 0.93174146])
# This set is made up of 25 samples uniformly-distributed in [0,0.3] and a further
# 5 samples uniformly-distributed in [0.9,1]
xp5 = np.array([0.01394879, 0.02194952, 0.02925089, 0.03395566, 0.05389559,
                0.0754086 , 0.08002495, 0.08598053, 0.09475561, 0.12191561,
                0.12315494, 0.14364847, 0.18455025, 0.19247459, 0.19436732,
                0.20205877, 0.20425212, 0.21203449, 0.21238893, 0.2316263 ,
                0.2381924 , 0.24158175, 0.260489  , 0.26776801, 0.27662425,
                0.90844754, 0.92687324, 0.95818959, 0.98446575, 0.99313721])

# The following arrays define the 'data' at these sampling points.
# yp1 = curveFittingFwd(np.array([0.5,2]),xp1,'polynomial') + np.random.normal(0,0.1,size=20)
yp1 = np.array([0.62075831, 0.66404134, 0.6541734 , 0.73098024, 0.98690644,
                0.96543423, 0.72329687, 1.39495485, 1.25161344, 1.32057621,
                1.49919959, 1.64859318, 1.67027886, 1.64236188, 1.75145179,
                2.15888014, 2.24377283, 2.29298466, 2.55081749, 2.32692856])
# yp2 = curveFittingFwd(np.array([1.,-0.2,0.5,0.3]),xp2,'polynomial') + np.random.normal(0,0.1,size=25)
yp2 = np.array([1.08325397, 1.03071536, 0.94623718, 0.91563318, 0.9506136 ,
                0.9367868 , 0.89703371, 1.02743885, 1.09065488, 1.08749366,
                1.06110559, 1.12980868, 1.14805172, 1.1982756 , 1.09985941,
                1.2375668 , 1.16813041, 1.28086391, 1.29770187, 1.51483868,
                1.40831814, 1.54380668, 1.49585496, 1.60347859, 1.58731048])
# yp3 = curveFittingFwd(np.array([1,0,0.3,-0.2,0,0,0.7,0.,0.,0.3,0.,0.,0.,0.,-.2]),xp3,'fourier') + np.random.normal(0,0.1,size=50)
yp3 = np.array([ 1.59943492,  1.29476968,  1.12470341,  1.21775137,  0.97557667,
                -0.30627497, -0.60216963, -0.44427947, -0.40124672, -0.35675986,
                0.47297917,  0.78227069,  0.97731201,  1.01290847,  1.06372535,
                0.99466542,  1.02487743,  0.9231454 ,  1.14490614,  0.42386911,
                -0.16176363, -0.33208629, -0.70905047, -0.46227597,  1.37939264,
                1.2141395 ,  1.21686748,  0.92069665,  0.54537001,  0.3790561 ,
                0.39905301,  0.20934235,  0.15370332,  0.29606956,  0.19790313,
                0.21644467,  0.21123332,  0.27204311,  0.39561473,  0.55357158,
                0.7340072 ,  0.8733571 ,  1.11949072,  1.26242578,  1.11077514,
                0.97156732,  1.05720015,  1.14813579,  1.3159233 ,  1.29343741])
# yp4 = curveFittingFwd(np.array([0.,2,-.3,-.6]),xp4,'polynomial') + np.random.normal(0,0.05,5)
yp4 = np.array([0.36882181, 0.43255407, 0.51268123, 0.94149677, 1.07863492])
#yp5 = curveFittingFwd(np.array([0.3,0.3,0.,-0.2,0.5,-.8,0.1,0.125]),xp5,'polynomial') + np.random.normal(0,0.1,30)
yp5 = np.array([0.32980738, 0.39458956, 0.24584324, 0.44555868, 0.31664395,
                0.35942105, 0.28608304, 0.16866817, 0.30498625, 0.24166827,
                0.25188134, 0.31265473, 0.33615641, 0.31067832, 0.29573255,
                0.37717869, 0.25847226, 0.54623771, 0.40138535, 0.22123748,
                0.32116732, 0.38347333, 0.43804293, 0.31016492, 0.34991239,
                0.38015441, 0.25919217, 0.25775872, 0.23213485, 0.41639483])

def curveFittingFwd(model, xpts, basis='polynomial',domainLength = 1.):
    """
    A fairly generic forward model for a range of 1D curve-fitting problems.
    model        - array of model parameters
    xpts         - array of sampling points
    basis        - one of "polynomial", "fourier" or "discrete"
    domainLength - float; all valid values for xpts should be in the range [0,domainLength]
    """
    singlePoint = False
    if domainLength<=0: raise ValueError("Argument 'domainLength' must be positive")
    # Convert list inputs to arrays
    if type(model) is type([]): model = np.array(model)
    if type(xpts) is type([]): xpts = np.array(xpts)
    # Check model parameters
    try:
        nModelParameters = model.shape[0]
    except AttributeError:
        raise ValueError("Argument 'model' should be a 1-D array")
    if len(model.shape)>1: 
        raise ValueError("Argument 'model' should be a 1-D array")
    # Check x-values
    try:
        npts = xpts.shape[0]
        if len(xpts.shape)!=1: raise ValueError("Argument 'xpts' should be a 1-D array")
    except AttributeError:
        singlePoint = True
        npts = 1
    if basis == 'polynomial':
        # y = m[0] + m[1]*x + m[2]*x**2 +...
        y = model[0]*np.ones([npts])
        for iord in range(1,nModelParameters):
            y += model[iord]*xpts**iord
    elif basis == 'fourier':
        if nModelParameters%2==0: 
            raise ValueError("Fourier basis requires odd number of model parameters")
        if not np.all(0<= xpts) and np.all(xpts<= domainLength): 
            raise ValueError("For Fourier basis, all sample points must be in interval (0,domainLength)")
        # y = m[0]/2 + m[1] sin (pi x/L) + m[2] cos (pi x/L) + m[3] sin(pi x/L) + ...
        y = np.ones([npts])*0.5*model[0]
        n = 1
        for i in range(1,nModelParameters,2):
            y += model[i]*np.sin(2*n*np.pi*xpts/domainLength) + model[i+1]*np.cos(2*n*np.pi*xpts/domainLength)
            n+=1
    elif basis == 'discrete':
        if not np.all(0<= xpts) and np.all(xpts<= domainLength): 
            raise ValueError("For discrete basis, all sample points must be in interval (0,domainLength)")
        bounds = np.linspace(0,domainLength,nModelParameters+1)
        y = np.zeros([npts])
        for ipt in range(0,npts):
            y[ipt] = model[max(0,np.searchsorted(bounds,xpts[ipt])-1)]
    else:
        raise ValueError("Unrecognised  value for 'basis'; please specify one of: 'polynomial', 'fourier' or 'discrete'")
    if singlePoint:
        return y[0]
    else:
        return y

def curveFittingJac(xpts,nModelParameters, basis='polynomial',domainLength=1.):
    singlePoint=False
    if domainLength<0:raise ValueError("Argument 'domainLength' must be positive")
    if type(xpts) is type([]):xpts=np.array(xpts)
    try:
        npts = xpts.shape[0]
        if len(xpts.shape)!=1: raise ValueError("Argument 'xpts' should be a 1-D array")
    except AttributeError:
        singlePoint = True
        npts = 1

    if basis == 'polynomial':
        G = np.zeros([npts,nModelParameters])
        for iord in range(0,nModelParameters):
            G[:,iord] = xpts**iord
    elif basis == 'fourier':
        if nModelParameters%2==0: raise ValueError("Fourier basis requires an odd number of model parameters")
        G = np.zeros([npts,nModelParameters])
        G[:,0] = 0.5
        n = 1
        for i in range(1,nModelParameters,2):
            G[:,i] = np.sin(2*n*np.pi*xpts/domainLength)
            G[:,i+1] = np.cos(2*n*np.pi*xpts/domainLength)
            n+=1
    elif basis == 'discrete':
        G = np.zeros([npts, nModelParameters])
        bounds = np.linspace(0,domainLength,nModelParameters+1)
        y = np.zeros([npts])
        for ipt in range(0,npts):
            G[ipt,max(0,np.searchsorted(bounds,xpts[ipt])-1)] = 1.
    else:
        raise ValueError("Unsupported basis")
    return G
