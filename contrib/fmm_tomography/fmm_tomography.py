import subprocess
import numpy as np
from scipy.stats import multivariate_normal

from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError
from cofi_espresso.utils import absolute_path as path, silent_remove
from . import waveTracker as wt


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
                ""
            ),
            (
                "Rawlinson, N. and Urvoy, M., 2006. Simultaneous inversion of active and passive source datasets for 3-D seismic structure with application to Tasmania, Geophys. Res. Lett., 33 L24313",
                "10.1029/2006GL028105"
            ),
            (
                "de Kool, M., Rawlinson, N. and Sambridge, M. 2006. A practical grid based method for tracking multiple refraction and reflection phases in 3D heterogeneous media, Geophys. J. Int., 167, 253-270",
                ""
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

        current_dir = path(".")
        self.tmp_files = ["fm2dss.in", "frechet.out", "gridc.vtx", "otimes.dat",
                    "raypath.out", "receivers.dat", "rtravel.out", "sources.dat",
                    "globalp.mod", "traveltime.mod"]
        self.tmp_paths = []
        for name in self.tmp_files:
            self.tmp_paths.append(current_dir / name)

        if example_number == 1:
            # read in data set
            sourcedat=np.loadtxt(path('datasets/ttimes/sources_crossb_nwt_s10.dat'))
            recdat = np.loadtxt(path('datasets/ttimes/receivers_crossb_nwt_r10.dat'))
            ttdat = np.loadtxt(path('datasets/ttimes/ttimes_crossb_nwt_s10_r10.dat'))
            recs = recdat.T[1:].T # set up receivers
            srcs = sourcedat.T[1:].T # set up sources
            nr,ns = np.shape(recs)[0],np.shape(srcs)[0] # number of receivers and sources
            print(' New data set has:\n',np.shape(recs)[0],
                ' receivers\n',np.shape(sourcedat)[0],
                ' sources\n',np.shape(ttdat)[0],' travel times')
            # rays = (ttdat[:,1] + ttdat[:,0]*nr).astype(int) # find rays from travel time file

            # Add Gaussian noise to data
            print(' Range of travel times: ',np.min(ttdat.T[2]),np.max(ttdat.T[2]),'\n Mean travel time:',np.mean(ttdat.T[2]))
            sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals
            np.random.seed(61254557)              # set random seed
            ttdat[:,2]+=np.random.normal(0.0, sigma, len(ttdat.T[2]))

            # true model
            extent = [0.,20.,0.,30.]
            mtrue = get_gauss_model(extent,32,48) # we get the true velocity model domain for comparison 
            slowness_true = 1 / mtrue

            # starting model
            nx,ny = mtrue.shape                   # set up grid
            mb = 2000.*np.ones([nx,ny])           # reference velocity model in m/s
            slowness_starting = 1 / mb

            # assign properties
            self.exe_fm2dss = str(current_dir)
            self._mtrue = mtrue
            self._mstart = mb
            self._strue = slowness_true
            self._sstart = slowness_starting
            self._data = ttdat
            self.params["extent"] = extent
            self.params["receivers"] = recs
            self.params["sources"] = srcs
            self.params["model_shape"] = slowness_true.shape
        else:
            raise InvalidExampleError

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
        return self._strue.flatten() 

    @property
    def starting_model(self):
        return self._sstart.flatten()
    
    @property
    def data(self):
        return self._data[:,2].flatten()

    @property
    def covariance_matrix(self):                # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional

    def forward(self, model, with_jacobian=False, **kwargs): # accepting "slowness" though keyword is "model"
        slowness_reshaped = model.reshape(self._mstart.shape)
        velocity = 1 / slowness_reshaped
        g = wt.gridModel(velocity, extent=self.extent)
        if "wdir" in kwargs: kwargs.pop("wdir")
        if "frechet" in kwargs: kwargs.pop("frechet")
        fmm = g.wavefront_tracker(
            self.receivers, 
            self.sources, 
            # verbose=True, 
            # paths=True, 
            frechet=True, 
            wdir=self.exe_fm2dss,
            **kwargs,
        )
        # paths = fmm.paths
        ttimes = fmm.ttimes
        A = fmm.frechet.toarray()
        self.clean_tmp_files()
        if with_jacobian:
            return np.array(ttimes).flatten(), A
        else:
            return np.array(ttimes).flatten()
    
    def jacobian(self, model, **kwargs):      # accepting "slowness" though keyword is "model"
        return self.forward(model, True, **kwargs)[1]

    def plot_model(self, model, with_paths=False, return_paths=False, **kwargs): # accepting "slowness" though keyword is "model"
        slowness_reshaped = model.reshape(self._mstart.shape)
        velocity = 1 / slowness_reshaped
        cline = kwargs.pop("cline", "g")
        alpha = kwargs.pop("alpha", 0.5)
        if with_paths or return_paths:
            g = wt.gridModel(velocity, extent=self.extent)
            fmm = g.wavefront_tracker(
                self.receivers, 
                self.sources, 
                paths=True, 
                wdir=self.exe_fm2dss,
            )
            paths = fmm.paths
            if with_paths:
                fig = wt.displayModel(
                    velocity, 
                    paths=paths, 
                    extent=self.extent, 
                    cline=cline, 
                    alpha=alpha,
                    **kwargs
                )
            else:
                fig = wt.displayModel(
                    velocity, 
                    paths=None, 
                    extent=self.extent, 
                    cline=cline, 
                    alpha=alpha,
                    **kwargs
                )
            return (fig, paths) if return_paths else fig
        else:
            return wt.displayModel(
                velocity, 
                paths=None, 
                extent=self.extent, 
                cline=cline, 
                alpha=alpha,
                **kwargs
            ) 
    
    def plot_data(self, data, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional

    def clean_tmp_files(self):
        for file_path in self.tmp_paths:
            silent_remove(file_path)
    

def get_gauss_model(extent,nx,ny): # build two gaussian anomaly velocity model
    vc1 = 1700.                           # velocity of circle 1
    vc2 = 2300.                           # velocity of circle 2
    vb = 2000.                            # background velocity
    dx = (extent[1]-extent[0])/nx                           # cell width
    dy = (extent[3]-extent[2])/ny                           # cell height
    xc = np.linspace(extent[0],extent[1],nx)    # cell centre
    yc = np.linspace(extent[2],extent[3],ny)    # cell centre
    X,Y = np.meshgrid(xc, yc,indexing='ij')   # cell centre mesh

    # Multivariate Normal
    c1,sig1 = np.array([7.0,22.]),6.0     # location and radius of centre of first circle
    c2,sig2 = np.array([12.0,10.]),10.0    # location and radius of centre of first circle
    rv1 = multivariate_normal(c1, [[sig1, 0], [0, sig1]])
    rv2 = multivariate_normal(c2, [[sig2, 0], [0, sig2]])

    # Probability Density
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    gauss1,gauss2 = rv1.pdf(pos),rv2.pdf(pos)
    return   2000.*np.ones([nx,ny])  + (vc1-vb)*gauss1/np.max(gauss1) + (vc2-vb)*gauss2/np.max(gauss2)
