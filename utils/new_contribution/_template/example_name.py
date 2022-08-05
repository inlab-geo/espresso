import numpy as np


example_number = 0

def set_example_number(num):
    example_number = num

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
