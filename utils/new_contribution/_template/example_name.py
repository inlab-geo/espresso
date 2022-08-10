import numpy as np


"""values set in _params can be accessed in all functions
feel free to assign more values in the following dictionary
check set_example_number function below for details"""
_params = {"example_number": 0}

def set_example_number(num):
    _params["example_number"] = num

    """you might want to set other useful example specific parameters 
    here so that you can access them in the other functions
    see the following as an example (suggested) usage of `_params`"""
    # if num == 0:
    #     _params["model"] = np.ones(10, 10)
    #     _params["data"] = np.ones(20,) * 2
    # elif num == 1:
    #     _params["model"] = np.ones(10, 10) * 3
    #     _params["data"] = np.ones(100,) * 4
    # else:
    #     raise ValueError("Invalide example_number, please choose between [0, 1]")

def suggested_model():
    raise NotImplementedError               # TODO implement me

def data():
    raise NotImplementedError               # TODO implement me

def forward(model, with_jacobian=False):
    if with_jacobian:
        raise NotImplementedError           # optional
    else:
        raise NotImplementedError           # TODO implement me

def jacobian(model):
    raise NotImplementedError               # optional

def plot_model(model):
    raise NotImplementedError               # optional

def plot_data(data):
    raise NotImplementedError               # optional
