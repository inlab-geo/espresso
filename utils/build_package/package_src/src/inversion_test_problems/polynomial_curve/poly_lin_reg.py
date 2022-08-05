import numpy as np

from ......contrib.base_example import BaseExample


class PolynomialCurve(BaseExample):
    def __init__(self, example_number=0):
        super().__init__(example_number)


    def get_suggested_model(self):
        return np.array([-6, -5, 2, 1])
    