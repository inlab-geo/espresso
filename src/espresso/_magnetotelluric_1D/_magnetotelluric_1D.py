from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0


class Magnetotelluric1D(EspressoProblem):
    """Forward simulation class
    """

    metadata = {
        "problem_title": "1D magnetotelluric",               
        "problem_short_description": "Compute the MT response of a 1D resistivty model",    

        "author_names": ["Hoël Seillé"],    

        "contact_name": "Hoël Seillé",         
        "contact_email": "hoel.seille@csiro.au",

        "citations": [],

        "linked_sites": [], 
    }


    def __init__(self, example_number=1):
        super().__init__(example_number)


        if example_number == 1:
            
            # true model
            # layer electrical resitivity in log10 ohm.m to ensures positive resistivities
            nLayers, min_thickness, vertical_growth= 50, 5, 1.15
            thicknesses = [min_thickness * vertical_growth**i for i in range(nLayers-1)]
            true_depths = np.cumsum(thicknesses)
            true_model = np.ones((nLayers))*3
            true_model[:10] = 2
            true_model[10:17] = 1
            
            # define frequencies (in Hz) where responses are computed
            fmin, fmax, f_per_decade = 0.1, 1e4, 5
            freqs = get_frequencies(fmin,fmax,f_per_decade)
            
            # generate synthetic data 
            # calculate impedance Z
            Z = forward_1D_MT(true_model, true_depths, freqs, return_Z = True)
            # add noise
            Z, Zerr = add_noise(Z, percentage = 5, seed = 1234)
            # transform Z to log10 apparent resistivity and phase (dobs)
            dobs, derr = z2rhophy(freqs, Z, dZ=Zerr)

            # define a starting 1D mesh and model 
            # the mesh contains many cells to produce a smooth model
            nLayers, min_thickness, vertical_growth= 50, 5, 1.15
            thicknesses = [min_thickness * vertical_growth**i for i in range(nLayers-1)]
            starting_depths = true_depths
            starting_model = np.ones((nLayers)) * 2 # 100 ohm.m starting model (log10 scale) 

            self._desc = "1 MT sounding over a 3 layers Earth model, smooth inversion"
            self._mtrue = true_model
            self._dptrue = true_depths
            self._mstart = starting_model
            self._dpstart = starting_depths
            self._dobs = dobs
            self._derr = derr
            self._freqs = freqs


        else:
            raise InvalidExampleError

    @property
    def description(self):
        return self._desc

    @property
    def model_size(self):
        return len(self.good_model)

    @property
    def data_size(self):
        return len(self.data)

    @property
    def good_model(self):
        return self._mtrue

    @property
    def starting_model(self):
        return self._mstart.flatten()
    
    @property
    def data(self):
        return self._dobs

    @property
    def covariance_matrix(self):
        return np.identity(len(self.data)) * self._derr**2

    @property
    def inverse_covariance_matrix(self):
        return np.identity(len(self.data)) * (1/self._derr**2)
        
    def set_start_model(self, new_start_model):
        self._mstart = new_start_model
        self._mtrue = new_start_model

    def set_start_mesh(self, new_start_mesh):
        self._dpstart = new_start_mesh
        self._dptrue = new_start_mesh

    def set_obs_data(self, dobs, derr, freqs):
        self._dobs = dobs
        self._derr = derr
        self._freqs = freqs

    def forward(self, model, return_jacobian = False):
        if return_jacobian:
            dpred, G = forward_1D_MT(model, self._dptrue, self._freqs, return_G=True)
            return dpred, G
        else:
            dpred= forward_1D_MT(model, self._dptrue, self._freqs, return_G=False)
            return dpred
    
    def jacobian(self, model):
        dpred, G = forward_1D_MT(model, self._dptrue, self._freqs, return_G = True)
        return G

    def plot_model(self, model, depths = None, max_depth = -1000, res_bounds = [0,4], title = None):
        nLayers = len(model) 
        if depths is None:
            max_z = min(max_depth,-self._dptrue[-1])
            depths = np.r_[-self._dptrue,max_z]
        else:
            max_z = min(max_depth,-depths[-1])
            depths = np.r_[-depths,max_z]

        fig = plt.figure(1,figsize=(4,4))
        ax = fig.add_subplot(1, 1, 1)
        px = np.zeros([2*nLayers,2])
        px[0::2,0],px[1::2,0],px[1::2,1],px[2::2,1] = model,model,depths[:],depths[:-1]
        ax.plot((px[:,0]),px[:,1],'k-',lw=3)
        ax.set_xlim(res_bounds);
        ax.set_ylim(max_depth,0)
        ax.set_xlabel(r'Log$_{10}$ resistivity ($\Omega$m)')
        ax.set_ylabel('Depth (m)')
        ax.grid(lw=0.2)
        ax.set_title(title)
        plt.tight_layout()
        #plt.show()
        return ax
    
    def plot_data(self, data1, data2 = None, Cm = None):
        nf = len(self._freqs)
        log10_rho = data1[:nf]
        phase = data1[nf:]
        fig, axs = plt.subplots(2,1,sharex = True,figsize = (4,4), num = 2)
        if Cm is None:
            axs[0].plot(np.log10(1/self._freqs), log10_rho,'k.')
            axs[1].plot(np.log10(1/self._freqs), phase,'k.')
        else:
            d_log10_rho = (np.diag(Cm)**0.5)[:nf]
            d_phase = (np.diag(Cm)**0.5)[nf:]
            axs[0].errorbar(np.log10(1/self._freqs), log10_rho, yerr = d_log10_rho, 
                            fmt='wo',elinewidth=0.6,markersize=4 ,ecolor='k',
                            capsize=2,capthick=0.6,mec='k',mew=0.5, alpha=0.9, zorder = 1, label = 'Obs')
            axs[1].errorbar(np.log10(1/self._freqs), phase, yerr = d_phase, 
                            fmt='wo',elinewidth=0.6,markersize=4 ,ecolor='k',
                            capsize=2,capthick=0.6,mec='k',mew=0.5, alpha=0.9, zorder = 1)
        if data2 is not None:
            log10_rho2 = data2[:nf]
            phase2 = data2[nf:]
            axs[0].plot(np.log10(1/self._freqs), log10_rho2, 'k-', lw = 1, zorder = 2, label = 'Resp')
            axs[1].plot(np.log10(1/self._freqs), phase2, 'k-', lw = 1, zorder = 2)
        axs[0].set_ylabel(r'Log$_{10}$ $\rho_{app}$ ($\Omega$m)')
        axs[1].set_yticks([0,45,90])
        axs[1].set_ylabel('Phase (deg.)')
        axs[1].set_xlabel('Log$_{10}$ Period (s)')
        axs[0].grid(lw=0.2)
        axs[1].grid(lw=0.2)
        # axs[0].legend()
        plt.tight_layout()
        #plt.show()
        return axs

    def misfit(self, data1, data2, Cm_inv = None):
        res = data1 - data2
        if Cm_inv is None:
            return float(res.T @ res)
        else:
            return float(res.T @ Cm_inv @ res)          

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional



def get_frequencies(fmin, fmax, f_per_decade):
    nf = ((np.log10(fmax) - np.log10(fmin)) * f_per_decade) + 1
    freqs = np.logspace(np.log10(fmax), np.log10(fmin), num = int(nf))
    return freqs


def forward_1D_MT(model, depths, freqs, return_G = False, return_Z = False):
    # Impedance recursive approach to compute analytically MT transfer
    # functions over a layered 1D Earth, as in:
    # Pedersen, J., Hermance, J.F. Least squares inversion of one-dimensional 
    #       magnetotelluric data: An assessment of procedures employed 
    #       by Brown University. Surv Geophys 8, 187–231 (1986).

    # adapted from https://empymod.emsig.xyz/en/stable/gallery/fdomain/magnetotelluric.html
    
    w = 2*np.pi*freqs # angular frequencies
    model_lin = 10**model # electrical resistivities in linear scale

    # Calculate impedance Z at the top of the bottom half space 
    Z = np.sqrt(1j * w * mu_0 * model_lin[-1])
    
    # The surface impedance Z is found recursively from the bottom propagating upwards
    for j in range(len(depths)-1,-1,-1):
        # calculate thickness th of layer j
        if j == 0: th = depths[j]
        else: th = depths[j] - depths[j-1]
        # calculate intrinsic impedance zo of layer j
        zo = np.sqrt(1j * w * mu_0 * model_lin[j])
        # calculate reflection coefficient R of layer j
        R = (zo - Z) / (zo + Z)
        # calculate induction parameter gamma of layer j
        gamma = np.sqrt(1j * w * mu_0 / model_lin[j])
        # update impedance Z at the top of layer j
        Z = zo * (1 - R * np.exp(-2*gamma*th)) / (1 + R * np.exp(-2*gamma*th))
        
    # instead of using the complex impedance, we use the log amplitude and phase
    # of Z (log10 apparent resistivity and phase)(Wheelock, Constable, Key; GJI 2015)
    data, _ = z2rhophy(freqs, Z, dZ=None)
    
    if return_G:
        # Jacobian calculation using finite differences 
        M = len(model)
        N = len(data)
        G = np.zeros((N, M))
        
        # ppert = 0.01 # Perturbation (percentage)
        # for i in range(M):
        #     model_pert = model + model * ppert * np.identity(M)[:,i]
        #     dpert = forward_1D_MT(model_pert,depths, freqs)
        #     G[:,i] = (dpert - data) / (model * ppert)[i]

        apert = 0.01 # Perturbation (absolute)
        for i in range(M):
            model_pert = model + apert * np.identity(M)[:,i]
            dpert = forward_1D_MT(model_pert,depths, freqs)
            G[:,i] = (dpert - data) / apert

    if return_Z:
        return Z

    elif return_G:
        return data, G

    else:
        return data



def z2rhophy(freqs,Z,dZ=None):
    
    # calcul of apparent resistivity and phases from the impedance Z
    rho_app = abs(Z)**2 / (mu_0*2*np.pi*freqs)
    phase = np.degrees(np.arctan2(Z.imag,Z.real))
    
    # calculate errors of apparent resistivity and phases
    if dZ is None:
        return np.r_[np.log10(rho_app), phase], np.r_[np.zeros(Z.shape[0]*2)]
    else:
        drho_app = 2*rho_app*dZ / abs(Z)
        dphase = np.degrees(0.5 * (drho_app/rho_app))
        log10_drho_app = (1/np.log(10)) * (drho_app/rho_app)
        return np.r_[np.log10(rho_app), phase], np.r_[log10_drho_app, dphase]




def add_noise(Z, percentage = 5, seed = 1234):
    # The standard deviations are taken as a percentage of the amplitude |Z|.
    # We assume a circularly symmetric Gaussian distribution in the complex 
    # plane for Z, with a common standard deviation Zerr for the real and
    # imaginary parts.
    
    np.random.seed(seed)
    Zerr = np.zeros_like(Z, dtype=float)
    for f in range(Z.shape[0]):
        Zerr[f] = 0.01*percentage*(Z[f].real**2+Z[f].imag**2)**0.5
        Zr = Z[f].real + np.random.normal(0.0, 0.01*percentage*abs(Z[f]))
        Zi = Z[f].imag + np.random.normal(0.0, 0.01*percentage*abs(Z[f]))
        Z[f] = Zr + 1j*Zi
    return Z, Zerr


def load_data(filename, error_floor = 0.05, subsampling = 3):

    # load the data, transform the apparent resistivity and its error to log10
    MTdata = np.loadtxt('../data/%s'%(filename),skiprows=1)
    MTdata = MTdata[::subsampling,:]
    
    freqs = MTdata[:,0]
    rho = MTdata[:,1]
    rho_err = MTdata[:,2]
    phase = MTdata[:,3]
    phase_err = MTdata[:,4]
    
    dobs = np.r_[np.log10(rho), phase]
    
    log10_drho = (1/np.log(10)) * (rho_err/rho)
    
    # add an error floor to the data: if the error if lower than the floor, then it is raised to the floor
    # it prevents from including unrealistic small data error into the inversion
    error_floor_rho = error_floor # the error floor is set as a percentage of the apparent resistivity 
    ef_rho_log = np.log10(1 + error_floor_rho) 
    ef_phase = (100 * error_floor_rho) * 0.286    
    
    err_rho_log = np.maximum(log10_drho, ef_rho_log)
    err_phase = np.maximum(phase_err, ef_phase)
    
    derr = np.r_[err_rho_log, err_phase]
    
    return freqs, dobs, derr


# 37 EARTH SCIENCES -> 3706 Geophysics -> 370602 Electrical and electromagnetic methods in geophysics -> Magnetotelluric -> Magnetotelluric1D
# description: Compute the MT response of a 1D resistivty model.
