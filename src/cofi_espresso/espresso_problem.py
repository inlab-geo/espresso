from abc import abstractmethod, ABCMeta


class EspressoProblem(metaclass=ABCMeta):
    """Base class for all Espresso problems

    All Espresso problems shoud be a subclass of this class.
    """

    def __init__(self, example_number=1):
        self.example_number = example_number
        self.params = dict()
    
    @property
    @abstractmethod
    def model_size(self):
        """
        Returns (M,) with

        M - integer -- The number of model parameters (i.e. the
            dimension of a model vector).
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def data_size(self):
        """
        Returns (N,) with

        N - integer -- The number of data points (i.e. the
            dimension of a data vector).
        """

    @property
    @abstractmethod
    def suggested_model(self):
        """
        Returns (m,) with
        
        m - np.array, shape(model_size) -- A model vector that the 
            contributor regards as being a 'correct' or 'sensible' 
            explanation of the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def starting_model(self):
        """
        Returns (m,) with

        m - np.array, shape(model_size) -- A model vector, possibly
            just np.zeros(model_size), representing a typical 
            starting point or 'null model' for an inversion.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self):
        """
        Returns (d,) with
        
        d - np.array, shape(data_size) -- A data vector in the same
            format as output by forward().
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, model, with_jacobian=False):
        """
        model - np.array, shape(model_size) -- A model vector
        with_jacobian - boolean, optional -- A switch governing the
            output required.

        If with_jacobian == True, returns (d, G)
                         == False,        (d,) with

        d - np.array, shape(data_size,) -- A simulated data vector 
            corresponding to `model`.
        G - np.array, shape(data_size, model_size) -- The jacobian
            such that G[i,j] = \partial d[i]/\partial model[j]

        If an example does not permit calculation of the Jacobian
        then calling with with_jacobian=True should result in a 
        NotImplementedError being raised.
        """
        raise NotImplementedError

    def jacobian(self, model):
        """
        model - np.array, shape(model_size) -- A model vector

        Returns (G,) with

        G - np.array, shape(data_size, model_size) -- The jacobian
            such that G[i,j] = \partial d[i]/\partial model[j]
        """
        raise NotImplementedError

    def plot_model(self, model):
        """
        model - np.array, shape(model_size) -- A model vector for visualisation

        Returns (fig,) with

        fig - matplotlib.pyplot.Figure -- A figure handle containing a
        basic visualisation of the model.
        """
        raise NotImplementedError
    
    def plot_data(self, data, data2 = None):
        """
        data - np.array, shape(data_size) -- A data vector for visualisation
        data2 - np.array, shape(data_size) -- A second data vector, for 
            comparison with the first

        Returns (fig,) with
        
        fig - matplotlib.pyplot.Figure -- A figure handle containing a 
        basic visualisation of a dataset and (optionally) comparing it
        to a second dataset.
        """
        raise NotImplementedError
    
    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )
