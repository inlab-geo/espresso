import numpy as np


np.random.seed(42)

_m_true = np.array([-6,-5,2,1])                                            # m
_sample_size = 20                                                          # N
x = np.random.choice(np.linspace(-3.5,2.5), size=_sample_size)             # x
def basis_func(x):
    return np.array([x**i for i in range(4)]).T                            # x -> G
def forward_func(m): 
    return (np.array([x**i for i in range(4)]).T) @ m                      # m -> y_synthetic
y_observed = forward_func(_m_true) + np.random.normal(0,1,_sample_size)    # d

print(list(x))
print(list(y_observed))
