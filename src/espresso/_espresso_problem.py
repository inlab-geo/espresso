from abc import abstractmethod, ABCMeta
import numbers

import numpy
from matplotlib.axes import Axes


def abstract_metadata_key(*names):
    """Class decorator to add one or more abstract attribute.
    ref: https://stackoverflow.com/questions/45248243/most-pythonic-way-to-declare-an-abstract-class-property
    """

    def _func(cls, *names):
        """Function that extends the __init_subclass__ method of a class."""
        cls.__abstract_metadata_keys__ = names
        for name in names:
            setattr(cls, name, NotImplemented)
        orig_init_subclass = cls.__init_subclass__

        def new_init_subclass(cls, **kwargs):
            try:
                orig_init_subclass(cls, **kwargs)
            except TypeError:
                orig_init_subclass(**kwargs)
            if getattr(cls, "metadata", NotImplemented) is NotImplemented:
                raise NotImplementedError(
                    "please define the metadata as a dictionary field in problem class"
                )
            for name in names:
                # if getattr(cls, name, NotImplemented) is NotImplemented:
                if name not in cls.metadata:
                    raise NotImplementedError(
                        f"{name} is required as a metadata entry but you haven't"
                        " defined it"
                    )

        cls.__init_subclass__ = classmethod(new_init_subclass)
        return cls

    return lambda cls: _func(cls, *names)


@abstract_metadata_key(
    "problem_title",
    "problem_short_description",
    "author_names",
    "contact_name",
    "contact_email",
    "citations",
    "linked_sites",
)
class EspressoProblem(metaclass=ABCMeta):
    r"""Base class for all Espresso problems.
    All Espresso problems shoud be a subclass of this class.

    Parameters
    ----------
    example_number : int, optional
        The index of example you want to access. A typical Espresso problem will have
        several examples that have indices starting from 1. By default 1.

    Raises
    ------
    InvalidExampleError
        when you've passed in an example number that isn't included in current problem


    .. rubric:: Metadata

    Problem-sepecific metadata include the following keys:

    - ``problem_title``
    - ``problem_short_description``
    - ``author_names``
    - ``contact_name``
    - ``contact_email``
    - ``citations``
    - ``linked_sites``

    And they can be accessed through the :code:`metadata` dictionary:

    .. code-block:: pycon

       >>> from espresso import <ProblemClass>
       >>> <ProblemClass>.metadata["problem_title"]
       This is a problem about...
       >>> <ProblemClass>.metadata.keys()
       dict_keys(['problem_title', 'problem_short_description', 'author_names', 'contact_name', 'contact_email', 'citations', 'linked_sites'])

    .. rubric:: Required attributes

    Required methods and properties are guaranteed to be written by problem
    contributors and available for user to access.

    .. autosummary::
        EspressoProblem.model_size
        EspressoProblem.data_size
        EspressoProblem.good_model
        EspressoProblem.starting_model
        EspressoProblem.data
        EspressoProblem.forward

    .. rubric:: Optional attributes

    Optional methods and properties have standards but are not always implemented for
    each Espresso problem. Try using them or check the documentation page for each
    problem to figure out whether they are available.

    .. autosummary::
        EspressoProblem.description
        EspressoProblem.covariance_matrix
        EspressoProblem.inverse_covariance_matrix
        EspressoProblem.jacobian
        EspressoProblem.plot_model
        EspressoProblem.plot_data
        EspressoProblem.misfit
        EspressoProblem.log_likelihood
        EspressoProblem.log_prior

    """

    def __init__(self, example_number: int = 1):
        self.example_number = example_number
        self.params = dict()

    @property
    def description(self) -> str:
        """Returns a brief description of current example

        Returns
        -------
        str
            A string containing a brief (1-3 sentence) description of the example.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_size(self) -> int:
        """Returns the number of model parameters

        Returns
        -------
        int
            The number of model parameters (i.e. the dimension of a model vector).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_size(self) -> int:
        """Returns the number of data points

        Returns
        -------
        int
            The number of data points (i.e. the dimension of a data vector).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def good_model(self) -> numpy.ndarray:
        """Returns a model vector that the contributor regards as a sensible
        explanation of the dataset

        Returns
        -------
        numpy.ndarray
            A model vector that the contributor regards as being a 'correct' or
            'sensible' explanation of the dataset. (In some problems it may be the
            case that there are many 'equally good' models. The contributor should
            select just one of these.) It has the same shape as :attr:`model_size`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def starting_model(self) -> numpy.ndarray:
        """Returns a model vector representing a typical starting point for inversion

        Returns
        -------
        numpy.ndarray
            A model vector, possibly just np.zeros(model_size), representing a typical
            starting point or 'null model' for an inversion. It has the same shape as
            :attr:`model_size`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> numpy.ndarray:
        """Returns a data vector in the same format as output by :meth:`forward`

        Returns
        -------
        numpy.ndarray
            A data vector in the same shape as :attr:`data_size` and the output from
            :meth:`forward`
        """
        raise NotImplementedError

    @property
    def covariance_matrix(self) -> numpy.ndarray:
        """Returns the covariance matrix for the data

        Returns
        -------
        numpy.ndarray
            The covariance matrix describing any uncertainty and correlations in the
            data vector. The output has shape (:attr:`data_size`, :attr:`data_size`)
        """
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self) -> numpy.ndarray:
        """Returns the inverse data covariance matrix for the data

        Returns
        -------
        numpy.ndarray
            The inverse data covariance matrix, in the shape
            (:attr:`data_size`, :attr:`data_size`)
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, model: numpy.ndarray, return_jacobian: bool = False
    ) -> numpy.ndarray:
        """Perform forward simulation with a model to produce synthetic data

        If return_jacobian == True, returns (d, G); else, returns d, where:

        - d : numpy.ndarray, shape(:attr:`data_size`,), a simulated data vector
          corresponding to the given model
        - G : numpy.ndarray, shape(:attr:`data_size`, :attr:`model_size`), the
          Jacobian such that :math:`G[i,j] = \partial d[i]/\partial model[j]`

        If an example does not permit calculation of the Jacobian then calling with
        return_jacobian=True should result in a NotImplementedError being raised.

        Parameters
        ----------
        model : numpy.ndarray
            a model vector, in the same shape of :attr:`model_size`
        return_jacobian: bool
            a switch governing the output required

        Returns
        -------
        (numpy.ndarray, numpy.ndarray) | numpy.ndarray
            (d, G) or d, depending on the value of return_jacobian. Details above.
        """
        raise NotImplementedError

    def jacobian(self, model: numpy.ndarray) -> numpy.ndarray:
        """Returns the Jacobian matrix

        Parameters
        ----------
        model : numpy.ndarray
            a model vector, in the same shape of :attr:`model_size`

        Returns
        -------
        numpy.ndarray
            the Jacobian such that :math:`G[i,j] = \partial d[i]/\partial model[j]`, in
            the shape of (:attr:`data_size`, :attr:`model_size`)
        """
        raise NotImplementedError

    def plot_model(self, model: numpy.ndarray) -> Axes:
        """Returns a figure containing a basic visualisation of the model

        Parameters
        ----------
        model : numpy.ndarray
            a model vector for visualisatioin, in the same shape of :attr:`model_size`

        Returns
        -------
        matplotlib.axes.Axes
            A matplotlib Axes handle containing a basic visualisation of the model.
        """
        raise NotImplementedError

    def plot_data(
        self, data1: numpy.ndarray, data2: numpy.ndarray = None
    ) -> Axes:
        """Returns a figure containing a basic visualisation of a dataset and
        (optionally) comparing it to a second dataset

        Parameters
        ----------
        data : numpy.ndarray
            A data vector for visualisation
        data2 : numpy.ndarray, optional
            A second data vector, for comparison with the first, by default None

        Returns
        -------
        matplotlib.axes.Axes
            A matplotlib Axes handle containing a basic visualisation of a dataset and
            (optionally) comparing it to a second dataset.
        """
        raise NotImplementedError

    def misfit(self, data: numpy.ndarray, pred: numpy.ndarray) -> numbers.Number:
        """Returns a measure of the extent to which a predicted data vector agrees with
        observed data

        Parameters
        ----------
        data : numpy.ndarray
            An observed data vector to base on
        pred : numpy.ndarray
            A predicted data vector to evaluate

        Returns
        -------
        Number
            A measure of the extent to which a predicted data vector, ``pred``, agrees
            with observed data, ``data``. Smaller numbers imply better agreement;
            0 -> perfect match.
        """
        raise NotImplementedError

    def log_likelihood(
        self, data: numpy.ndarray, pred: numpy.ndarray
    ) -> numbers.Number:
        """Returns the log likelihood density value

        Parameters
        ----------
        data : numpy.ndarray
            An observed data vector to base on
        pred : numpy.ndarray
            A predicted data vector to evaluate

        Returns
        -------
        Number
            The log likelihood that ``data`` is an imperfect observation of a system
            generating data ``pred``.
        """
        raise NotImplementedError

    def log_prior(self, model: numpy.ndarray) -> numbers.Number:
        """Returns the log prior density value

        Parameters
        ----------
        model : numpy.ndarray
            A model vector to evaluate, shape(:attr:`model_size`)

        Returns
        -------
        Number
            The log probability that a system is described by ``model`` prior to
            seeing any data.
        """
        raise NotImplementedError

    def list_capabilities(self) -> list:
        """Returns a dictionary describing the capabilities of the current example

        Examples
        --------
        >>> import espresso
        >>> r = espresso.ReceiverFunctionInversion()
        >>> r.list_capabilities()
        ['model_size', 'data_size', 'good_model', 'starting_model', 'data', 'description', 'covariance_matrix', 'plot_model', 'plot_data', 'log_likelihood', 'log_prior', 'rf', 'capability_report']
        """
        from .capabilities import list_capabilities

        return list_capabilities(self.__class__.__name__)[self.__class__.__name__]

    def __getattr__(self, key):
        if hasattr(self, "params") and key in self.params:
            return self.params[key]
        if key in self.metadata:
            return self.metadata[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )
