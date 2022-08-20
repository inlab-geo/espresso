from abc import abstractmethod, ABCMeta
from ast import Not


class EspressoProblem(metaclass=ABCMeta):
    """Base class for all Espresso problems

    All Espresso problems shoud be a subclass of this class.
    """

    def __init__(self, example_number=1):
        self.example_number = example_number
        self.params = dict()
    
    @property
    def description(self):
        """
        Returns (desc,) with
        
        desc - str -- A string containing a brief (1-3 sentence) 
            description of the example.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    @abstractmethod
    def good_model(self):
        """
        Returns (m,) with
        
        m - np.array, shape(model_size) -- A model vector that the 
            contributor regards as being a 'correct' or 'sensible' 
            explanation of the dataset. (In some problems it may 
            be the case that there are many 'equally good' models.
            The contributor should select just one of these.)
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

    @property
    @abstractmethod
    def covariance_matrix(self):
        """
        Returns (C,) with

        C - np.array, shape(data_size, data_size) -- The covariance
            matrix describing any uncertainty and correlations in the
            data vector.
        """
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        """
        Returns (iC,) with
        
        iC - np.array, shape(data_size, data_size) -- The inverse
             data covariance matrix.
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

    def misfit(self, data, pred):
        """
        data - np.array, shape(data_size)
        pred - np.array, shape(data_size)

        Returns (phi,) with

        phi - float -- A measure of the extent to which a predicted data
            vector, `pred`, agrees with observed data, `data`. Smaller 
            numbers imply better agreement; 0 -> perfect match.
        """
        raise NotImplementedError
    
    def log_likelihood(self, data, pred):
        """
        data - np.array, shape(data_size)
        pred - np.array, shape(data_size)
        
        Returns (p,) with

        p - float -- The log likelihood that `data` is an imperfect 
            observation of a system generating data `pred`.

        """
        raise NotImplementedError

    def log_prior(self, model):
        """
        model - np.array, shape(model_size)

        Returns (p,) with

        p - float -- The log probability that a system is described by
            `model` prior to seeing any data.
        """
        raise NotImplementedError
        

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )
