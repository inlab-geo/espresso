from abc import abstractmethod, ABCMeta


class EspressoProblem(metaclass=ABCMeta):
    """Base class for all Espresso problems

    All Espresso problems shoud be a subclass of this class.
    """

    def __init__(self, example_number=0):
        self.example_number = example_number
        self.params = dict()
    
    @abstractmethod
    def suggested_model(self):
        raise NotImplementedError

    @abstractmethod
    def data(self):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, model, with_jacobian=False):
        raise NotImplementedError

    def jacobian(self, model):
        raise NotImplementedError

    def plot_model(self, model):
        raise NotImplementedError
    
    def plot_data(self, data):
        raise NotImplementedError
    
    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        else:
            raise AttributeError
