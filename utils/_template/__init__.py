import numpy as np

# The name of the class has to be the same as the name of the containing folder:
#   '/contrib/new_problem/__init__.py'
class new_problem(auxiliary_functions):
    """
    !!! The main code goes here !!!
    Description of new_problem.

    Parameters:
    --------------------
    v:         Seismic velocity
    density1:  Starting guess for the density of an imaginary upper layer, in [kg/m^3]
    density2:  Starting guess for the density of an imaginary lower layer, in [kg/m^3]
    dx:        Spatial resolution of the model [m]
    dt:        Temporal resolution used by the forward code [ms]
    m:         The model in a 1-D array containing densities, in [kg/m^3]

    --------------------
    Functions:
    --------------------
    forward: Calculates synthetic data using the model and recording locations.

    gradient: Calculates the jacobian given the model and recording locations.

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
            m: The model in a 1-D array containing densities, in [kg/m^3]

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

        Returns the starting model.

        Parameters
        --------------------
        *args

        m: The starting model

        --------------------
        """

        return self.m


    def forward(self, m):
        """
        Description of forward().

        Calculates synthetic data using the model and recording locations.

        Parameters
        --------------------
        *args

        m: The model
        synth_data: Synthetic data based, generated using my amazing forward code

        --------------------
        """

        return synth_data

    def gradient(self, m):
        """
        Description of gradient().

        Calculates the jacobian given the model and recording locations.

        Parameters
        --------------------
        *args

        J: The jacobian, given the model and recording locations.

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
