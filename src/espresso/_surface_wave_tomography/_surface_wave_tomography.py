from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError
from espresso.utils import absolute_path
from seislib.plotting import plot_map
from seislib.utils import load_pickle
import seislib.colormaps as scm
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class SurfaceWaveTomography(EspressoProblem):
    """Forward simulation class"""

    metadata = {
        "problem_title": "Surface-wave Tomography",  # To be used in docs
        "problem_short_description": (
            "Mapping lateral variations in surface-wave "
            "velocity at continental (USA -- example 1) and global (example 2) "
            "scale. Here, the problem is linearized, meaning that we assume that "
            "surface waves travel along the great-circle path connecting two "
            "points on the Earth surface."
        ), # 1-3 sentences
        # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]
        "author_names": [
            "Fabrizio Magrini",
        ],
        # Contact for contributor/maintainer of espresso example
        "contact_name": "Fabrizio Magrini",
        "contact_email": "fabrizio.magrini@anu.edu.au",
        # Reference to publication(s) that describe this example. In most
        # cases there will only be a single entry in this list.
        # List of (citation, doi) pairs e.g.
        # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
        # If there are no citations, use empty list `[]`
        "citations": [
            ('Surface-wave tomography using SeisLib: a Python package for '
             'multiscale seismic imaging, GJI, vol 84, 1011-1030, 2022. '
             'F. Magrini, L. Sebastian, E. K{\"a}stle, L. Boschi',
            'https://doi.org/10.1093/gji/ggac236'
            ),
            ('A global model of Love and Rayleigh surface wave dispersion and '
             'anisotropy, GJI, vol 187, 1668-1686, 2011. G. Ekstr{\"o}m',
             'https://doi.org/10.1111/j.1365-246X.2011.05225.x'
            )
        ],
        # List of (title, address) pairs for any websites that
        # should be linked in the documentation, e.g.
        # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
        #                 ("Data source","https://www.data.com") ]
        # If there are no links, use empty list `[]`
        "linked_sites": [("PyPI Installation", "https://pypi.org/project/seislib/"), 
                          ("SeisLib docs", "https://seislib.readthedocs.io/en/latest/")],
    }
    
        
    def __init__(self, example_number=1):
        super().__init__(example_number)

        if example_number == 1:
            self._description = (
                'Rayleigh-wave phase velocity across USA at 10 s period as '
                'calculated by Magrini et al. (2022)'
                )
            
        elif example_number == 2:
            self._description = (
                'Global measurements of Rayleigh-wave phase velocity at 50 s '
                'period as calculated by Ekström et al. (2011)'
                )
        else:
            raise InvalidExampleError
        
        self._example_number = example_number
        self.example_dict = load_pickle(
            absolute_path('./data/example%d.pickle'%example_number)
            )
        self._data = self.example_dict['slowness']
        self._jacobian = self.example_dict['jacobian']
        self._good_model = self.example_dict['model']
        ref_slowness = np.mean(self._data)
        self._null_model = np.full(self._good_model.size, ref_slowness)
        self.parameterization = self.example_dict['grid']


    @property
    def description(self):
        return self._description

    @property
    def model_size(self):
        return self._null_model.size

    @property
    def data_size(self):
        return self._data.size

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
    def covariance_matrix(self):  # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError  # optional

    def forward(self, model, return_jacobian=False):
        jacobian = self.jacobian(model)
        dpred = jacobian @ model
        if return_jacobian:
            return dpred, jacobian
        return dpred 

    def jacobian(self, model):
        return self._jacobian

    def plot_model(self, model, **kwargs):
        velocity = 1 / model
        vmean = np.mean(velocity)
        proj = ccrs.Robinson() if self._example_number==2 else ccrs.Mercator()
        fig = plt.figure()
        ax = plt.subplot(111, projection=proj)
        ax.coastlines()
        img, cb = plot_map(self.parameterization.mesh,
                           velocity,
                           ax=ax,
                           cmap=scm.roma,
                           vmin=vmean - vmean*0.1,
                           vmax=vmean + vmean*0.1,
                           show=False,
                           **kwargs)       
        cb.set_label('Phase velocity [m/s]')      
        plt.tight_layout()
        return ax

    def plot_data(self, data, data2=None):
        raise NotImplementedError  # optional

    def misfit(self, data, data2):
        return np.sum(np.square(data - data2))

    def log_likelihood(self, data, data2):
        raise NotImplementedError  # optional

    def log_prior(self, model):
        raise NotImplementedError  # optional


# 37 EARTH SCIENCES -> 3706	Geophysics -> 370609 Seismology and seismic exploration -> Ambient noise -> SurfaceWaveTomography
# description: Mapping lateral variations in surface-wave velocity at continental (USA -- example 1) and global (example 2) scale. Here, the problem is linearized, meaning that we assume that surface waves travel along the great-circle path connecting two points on the Earth surface.
