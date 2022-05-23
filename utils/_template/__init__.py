import numpy as np

class new_problem(auxiliary_functions):
    """
    !!! The main code goes here !!!
    Description of new_problem.
    
    Parameters:
    --------------------
    v:         Seismic velocity
    density1:  Starting guess for the density of the upper layer, in [kg/m^3]
    density2:  Starting guess for the density of the lower layer, in [kg/m^3]
    dx:        Spatial resolution [m]
    dt:        Temporal resolution [ms]
    m:         The model in a 1-D array containing densities

    --------------------
    Functions:
    --------------------
    forward:       
    
    gradient:      
    
    plot_model:    
    
    --------------------
    """
    
    def __init__(self, example_number=0):
        
        # In case of having multiple configurations, separate them using example_number
        self._ieg=example_number


        if self._ieg==0:
            """
            # First configuration goes here.
            # Load basic parameters of the new problem
            # 
            
            Parameters
            --------------------
            *args

            v: Seismic velocity
            density1: Starting guess for the density of the upper layer, in [kg/m^3]
            density2: Starting guess for the density of the lower layer, in [kg/m^3]
            dx: Spatial resolution [m]
            dt: Temporal resolution [ms]
            m: The model in a 1-D array containing densities
            
            """
            name =__name__
            
            # Load ASCII data
            tmp = pkgutil.get_data(__name__, "data/testdata.txt")
            tmp2=tmp.decode("utf-8") 
            self.m=np.loadtxt(StringIO(tmp2))
            del  tmp, tmp2
            
            # Set some values
            self.v = 3800
            self.density1 = 3000
            self.density2 = 4000
            self.dt = 5
            self.dx = 0.5
            self.recording_locations=np.linspace(-100,100,200)


        elif self._ieg == 1:
            
            # Second configuration goes here.
            
        else:
            
            print("Error - example number not defined")
    
    def get_model(self):
        """
        Description of get_model().
        
        Returns the model.
        
        Parameters
        --------------------
        *args

        --------------------
        """
        
        return self.m

    
    def forward(self, m):
        """
        Description of forward().
        
        Calculates synthetic data using the model.
        
        Parameters
        --------------------
        *args
        
        m: The model in a 1-D array containing densities
        --------------------
        """

        return synth_data
    
    def gradient(self, m):
        """
        Description of gradient().
        
        Returns the jacobian given the model and recording locations.
        
        Parameters
        --------------------
        *args
        
        m: The model in a 1-D array containing densities
        --------------------
        """

        return jacobian
    
    def plot_model(self, m, data):
        
        if self._ieg == 0:
            # Plot command for the first configuration.
            
        elif self._ieg == 1:
            # Plot command for the first configuration.
            
        else:
            print("Error - example number not defined")
    
        
    
