from ..espresso_problem import EspressoProblem


class ExampleName(EspressoProblem):
    """Forward simulation class
    """

    def __init__(self, example_number=0):
        super().__init__(example_number)

        """you might want to set other useful example specific parameters here
        so that you can access them in the other functions see the following as an 
        example (suggested) usage of `self.params`
        """
        # if example_number == 0:
        #     self.params["some_attribute"] = some_value_0
        #     self.params["another_attribte"] = another_value_0
        # elif example_number == 1:
        #     self.params["some_attribute"] = some_value_1
        #     self.params["another_attribte"] = another_value_1
        # else:
        #     raise ValueError(
        #         "The example number supplied is not supported, please consult "
        #         "Espresso documentation at "
        #         "https://cofi-espresso.readthedocs.io/en/latest/contrib/index.html"
        #         "for problem-specific metadata, e.g. number of examples provided"
        #     )


    def suggested_model(self):
        raise NotImplementedError               # TODO implement me
    
    def data(self):
        raise NotImplementedError               # TODO implement me

    def forward(self, model, with_jacobian=False):
        if with_jacobian:
            raise NotImplementedError           # optional
        else:
            raise NotImplementedError           # TODO implement me
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        raise NotImplementedError               # optional
    
    def plot_data(self, data):
        raise NotImplementedError               # optional
