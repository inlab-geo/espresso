import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.stats import multivariate_normal
from scipy import sparse

from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError
from espresso.utils import absolute_path as path
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
            ),
            (
                "Saygin, E. 2007. Seismic receiver and noise correlation based studies in Australia, PhD thesis, Australian National University.",
                "10.25911/5d7a2d1296f96"
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

        # random seed for data noise          
        np.random.seed(61254557)              # set random seed

        if example_number == 1:
            # read in data set
            sourcedat=np.loadtxt(path('datasets/example1/sources_crossb_nwt_s10.dat'))
            recdat = np.loadtxt(path('datasets/example1/receivers_crossb_nwt_r10.dat'))
            ttdat = np.loadtxt(path('datasets/example1/ttimes_crossb_nwt_s10_r10.dat'))
            recs = recdat.T[1:].T # set up receivers
            srcs = sourcedat.T[1:].T # set up sources
            print(' New data set has:\n',np.shape(recs)[0],
                ' receivers\n',np.shape(sourcedat)[0],
                ' sources\n',np.shape(ttdat)[0],' travel times')

            # Add Gaussian noise to data
            self.params["noise_sigma"] =  0.00001                   # Noise is 1.0E-5, ~5% of standard deviation of initial travel time residuals
            print(' Range of travel times: ',np.min(ttdat.T[2]),np.max(ttdat.T[2]),'\n Mean travel time:',np.mean(ttdat.T[2]))
            ttdat[:,2]+=np.random.normal(0.0, self.noise_sigma, len(ttdat.T[2]))

            # true model
            extent = [0.,20.,0.,30.]
            mtrue = get_gauss_model(extent,32,48) # we get the true velocity model domain for comparison 
            slowness_true = 1 / mtrue

            # starting model
            nx,ny = mtrue.shape                   # set up grid
            mb = 2000.*np.ones([nx,ny])           # reference velocity model in m/s
            slowness_starting = 1 / mb

            # assign properties
            self._mtrue = mtrue
            self._mstart = mb
            self._strue = slowness_true
            self._sstart = slowness_starting
            self._data = ttdat[:,2]
            self.params["extent"] = extent
            self.params["receivers"] = recs
            self.params["sources"] = srcs
            self.params["model_shape"] = slowness_true.shape
        elif example_number == 2:
            filenamev = path('datasets/example2/gridt_ex1.vtx')     # filename to read in example velocity model 2
            filenames = path('datasets/example2/sources_ex1.dat')   # filename to read in example sources for model 2
            filenamer = path('datasets/example2/receivers_ex1.dat') # filename to read in example receivers for model 2
            filenamett = path('datasets/example2/ttimes.dat')
            # set up velocity model and source/receivers
            m,extent = read_vtxmodel(filenamev) # set up velocity model
            srcs     = read_sources(filenames)  # set up sources
            recs     = read_sources(filenamer)  # set up receivers
            self._mtrue = m
            self._mstart = 5 * np.ones(m.shape)
            self._strue = 1 / m
            self._sstart = 1 / self._mstart
            self.params["extent"] = extent
            self.params["receivers"] = recs
            self.params["sources"] = srcs
            self.params["model_shape"] = m.shape
            # generate data
            ttdat = np.loadtxt(filenamett)
            print(' New data set has:\n',np.shape(recs)[0],
                ' receivers\n',np.shape(srcs)[0],
                ' sources\n',np.shape(ttdat)[0],' travel times')
            print(' Range of travel times: ',np.min(ttdat),np.max(ttdat),'\n Mean travel time:',np.mean(ttdat))
            # add noise to data
            self.params["noise_sigma"] =  2                   # Noise is 2, ~5% of standard deviation of initial travel time residuals
            ttdat+=np.random.normal(0.0, self.noise_sigma, len(ttdat))
            self._data = ttdat
        elif example_number == 3:
            filenamev = path('datasets/example3/gridt_ex2.vtx')     # filename to read in example velocity model 3
            filenames = path('datasets/example3/sources_ex2.dat')   # filename to read in example sources for model 3
            filenamer = path('datasets/example3/receivers_ex2.dat') # filename to read in example receivers for model 3
            filenamett = path('datasets/example3/ttimes.dat')
            # set up velocity model and source/receivers
            m,extent = read_vtxmodel(filenamev, with_line_breaks=False) # set up velocity model
            srcs     = read_sources(filenames)  # set up sources
            recs     = read_sources(filenamer)  # set up receivers
            self._mtrue = m
            self._mstart = 5 * np.ones(m.shape)
            self._strue = 1 / m
            self._sstart = 1 / self._mstart
            self.params["extent"] = extent
            self.params["receivers"] = recs
            self.params["sources"] = srcs
            self.params["model_shape"] = m.shape
            # generate data
            ttdat = np.loadtxt(filenamett) 
            print(' New data set has:\n',np.shape(recs)[0],
                ' receivers\n',np.shape(srcs)[0],
                ' sources\n',np.shape(ttdat)[0],' travel times')
            print(' Range of travel times: ',np.min(ttdat),np.max(ttdat),'\n Mean travel time:',np.mean(ttdat))
            # add noise to data
            self.params["noise_sigma"] =  1                   # Noise is 1, ~5% of standard deviation of initial travel time residuals
            ttdat+=np.random.normal(0.0, self.noise_sigma, len(ttdat))
            self._data = ttdat
        else:
            raise InvalidExampleError

    @property
    def description(self):
        if self.example_number == 1:
            return "Cross borehole velocity model"
        elif self.example_number == 2:
            return (
                "Simple alternating checkerboard model, commonly used for testing "
                "inversion schemes as a synthetic true model. It was taken from the "
                "fmst fast Marching package of N. Rawlinson."
            )
        elif self.example_number == 3:
            return (
                "Australian Surface wave Shear velocity model of Australia, derived "
                "from ambient noise tomography.  From the Ph.D. thesis of E. Saygin "
                "(2007)"
            )

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
        return self._data.flatten()

    @property
    def covariance_matrix(self):                # optional
        sigma_sq = self.noise_sigma ** 2
        return sparse.diags([sigma_sq] * self.data_size)

    @property
    def inverse_covariance_matrix(self):
        sigma_sq_inv = 1 / self.noise_sigma ** 2
        return sparse.diags([sigma_sq_inv] * self.data_size)

    def forward(self, model, return_jacobian=False, **kwargs): # accepting "slowness" though keyword is "model"
        slowness_reshaped = model.reshape(self._mstart.shape)
        velocity = 1 / slowness_reshaped
        g = wt.gridModel(velocity, extent=self.extent)
        if "wdir" in kwargs: kwargs.pop("wdir")
        if "frechet" in kwargs: kwargs.pop("frechet")
        fmm = self.call_wavefront_tracker(
            velocity, 
            frechet=True, 
            **kwargs, 
        )
        # paths = fmm.paths
        ttimes = fmm.ttimes
        A = fmm.frechet.toarray()
        if return_jacobian:
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
            fmm = self.call_wavefront_tracker(
                velocity,
                paths=True,
            )
            paths = fmm.paths
            if with_paths:
                fig = wt.displayModel(
                    velocity, 
                    paths=paths, 
                    extent=self.extent, 
                    cline=cline, 
                    alpha=alpha,
                    use_geographic=self.example_number>=3, 
                    **kwargs
                )
            else:
                fig = wt.displayModel(
                    velocity, 
                    paths=None, 
                    extent=self.extent, 
                    cline=cline, 
                    alpha=alpha,
                    use_geographic=self.example_number>=3, 
                    **kwargs
                )
            ax = fig.get_axes()[0]
            self._plot_labelling(ax)
            return (ax, paths) if return_paths else ax
        else:
            fig = wt.displayModel(
                velocity, 
                paths=None, 
                extent=self.extent, 
                cline=cline, 
                alpha=alpha,
                use_geographic=self.example_number>=3, 
                **kwargs
            ) 
            ax = fig.get_axes()[0]
            self._plot_labelling(ax)
            return ax
    
    def plot_data(self, data1, data2=None):
        raise NotImplementedError               # optional

    def misfit(self, data1, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional
    
    def call_wavefront_tracker(self, velocity_reshaped, **kwargs):
        original_dir = Path.cwd()
        g = wt.gridModel(velocity_reshaped, extent=self.extent)
        with tempfile.TemporaryDirectory(dir=path(".")) as tmpdir:
            tmpdir = path(tmpdir)
            # print(f'Temporary directory created at {tmpdir}')
            # os.chdir(tmpdir)
            fmm = g.wavefront_tracker(
                self.receivers,
                self.sources, 
                wdir=str(tmpdir), 
                **kwargs,
            )
        os.chdir(original_dir)
        return fmm
    
    def _plot_labelling(self, ax):
        if self.example_number < 3:
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
        else:
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        

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


# build test velocity models
# read vtx format velocity model and source receivers files
def read_vtxmodel(filename, with_line_breaks=True):
    with open(filename, 'r') as f:
        lines = f.readlines()
        columns = lines[0].split()
        ny,nx = int(columns[0]),int(columns[1])
        columns = lines[1].split()
        extent = 4*[0.]
        extent[3],extent[0] = float(columns[0]),float(columns[1])
        columns = lines[2].split()
        dlat,dlon = float(columns[0]),float(columns[1])
        extent[1] = extent[0] + nx*dlon
        extent[2] = extent[3] - ny*dlat
        vc = np.zeros((nx+2,ny+2))
        k = 3
        for i in range(nx+2):
            if with_line_breaks:
                k += 1
            for j in range(ny+2):
                columns = lines[k].split()
                vc[i,j] = float(columns[0])
                k+=1
        if with_line_breaks:
            v = vc[1:nx+1,1:ny+1]
        else:
            v = vc[1:nx+1,ny+1:1:-1]
        f.close()
    return v,extent
    
def read_sources(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        columns = lines[0].split()
        ns = int(columns[0])
        srcs = np.zeros((ns,2))
        for i in range(ns):
            columns = lines[i+1].split()
            srcs[i,1],srcs[i,0] = float(columns[0]),float(columns[1])
        f.close()
    return srcs
def read_receivers(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        columns = lines[0].split()
        nr = int(columns[0])
        recs = np.zeros((nr,2))
        for i in range(nr):
            columns = lines[i+1].split()
            recs[i,1],recs[i,0] = float(columns[0]),float(columns[1])
        f.close()
    return recs


# 37 EARTH SCIENCES -> 3706	Geophysics -> 370609 Seismology and seismic exploration -> Fast Marching Method -> FmmTomography
# description: The wave front tracker routines solves boundary value ray tracing problems into 2D heterogeneous wavespeed media, defined by continuously varying velocity model calculated by 2D cubic B-splines.
