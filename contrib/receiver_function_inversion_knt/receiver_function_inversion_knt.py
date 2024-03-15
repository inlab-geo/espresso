import numpy as np
import matplotlib.pyplot as plt

from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError
from espresso.utils import absolute_path as path


def form_layercake_model(thicknesses, vs):
    model = np.zeros((len(vs) * 2 - 1))
    model[1::2] = thicknesses
    model[::2] = vs
    return model


def split_layercake_model(model):
    thicknesses = model[1::2]
    vs = model[::2]
    return thicknesses, vs


class ReceiverFunctionInversionKnt(EspressoProblem):
    """Forward simulation class"""

    metadata = {
        "problem_title": "Receiver function (C)",
        "problem_short_description": (
            "Receiver function inference problem based on a forward code by Brian "
            "Kennet and adapted by Lupei Zhu and Sheng Wang"
        ),
        "author_names": [
            "Sheng Wang",
            "Jiawen He",
        ],
        "contact_name": "Jiawen He",
        "contact_email": "hanghur@gmail.com",
        "citations": [
            (
                (
                    "Kennett, B. (2009). Seismic wave propagation in stratified media"
                    " (p. 288). ANU Press."
                ),
                "10.26530/OAPEN_459524",
            )
        ],
        "linked_sites": [
            ("Code source", "https://www.eas.slu.edu/People/LZhu/home.html")
        ],
    }

    def __init__(self, example_number=1):
        super().__init__(example_number)

        from .build import rf

        self.rf = rf

        if example_number == 1:
            self._description = "a three-layer model with a synthetic dataset"
            self._thicknesses = [10, 20]
            self._vs = [3.3, 3.4, 4.5]
            self._vp_vs = [1.732, 1.732, 1.732]
            self._ray_param_s_km = 0.07
            self._t_shift = 5
            self._t_duration = 50
            self._t_sampling_interval = 0.5
            self._gauss = 1.0
            self._data_noise = 0.02
            _dataset = np.loadtxt(path(f"data/dataset1.txt"))
            self._data_times = _dataset[:, 0]
            self._data_rf = _dataset[:, 1]
        elif example_number == 2:
            self._description = "a nine-layer model with a synthetic dataset"
            self._thicknesses = [10, 10, 15, 20, 20, 20, 20, 20]
            self._vs = [3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5]
            self._vp_vs = [1.77] * 9
            self._ray_param_s_km = 0.07
            self._t_shift = 5
            self._t_duration = 25
            self._t_sampling_interval = 0.1
            self._gauss = 1.0
            self._data_noise = 0.02
            _dataset = np.loadtxt(path(f"data/dataset2.txt"))
            self._data_times = _dataset[:, 0]
            self._data_rf = _dataset[:, 1]
        # elif example_number == 3:   # TODO real data example from computer programs in seismology
        #     raise NotImplementedError
        else:
            raise InvalidExampleError

        self._true_model = form_layercake_model(self._thicknesses, self._vs)
        self._starting_model = np.ones(self._true_model.size)
        self._starting_model[::2] = 3.5
        self._starting_model[1::2] = 10

    @property
    def description(self):
        return self._description

    @property
    def model_size(self):
        return len(self._thicknesses) + len(self._vs)

    @property
    def data_size(self):
        return len(self._data_times)

    @property
    def good_model(self):
        return self._true_model

    @property
    def starting_model(self):
        return self._starting_model

    @property
    def data(self):
        return self._data_rf

    @property
    def covariance_matrix(self):  # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError  # optional

    def forward(self, model, return_jacobian=False):
        if return_jacobian:
            raise NotImplementedError  # optional
        else:
            thicknesses, vs = split_layercake_model(model)
            data_rf = self.rf.rf_calc(
                ps=0, 
                thik=thicknesses,
                beta=vs,
                kapa=self._vp_vs, 
                p=self._ray_param_s_km, 
                duration=self._t_duration,
                dt=self._t_sampling_interval,
                shft=self._t_shift,
                gauss=self._gauss,
            )
            print(data_rf.shape, self.data_size)
            return data_rf

    def jacobian(self, model):
        raise NotImplementedError  # optional

    def plot_model(self, model, ax=None, title="model", **kwargs):
        # process data
        thicknesses = np.append(model[1::2], max(model[1::2]))
        velocities = model[::2]
        y = np.insert(np.cumsum(thicknesses), 0, 0)
        x = np.insert(velocities, 0, velocities[0])

        # plot depth profile
        if ax is None:
            _, ax = plt.subplots()
        plotting_style = {
            "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 0.5)),
            "alpha": 0.2,
            "color": kwargs.pop("color", kwargs.pop("c", "blue")),
        }
        plotting_style.update(kwargs)
        ax.step(x, y, where="post", **plotting_style)
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
        ax.set_xlabel("Vs (km/s)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(title)
        return ax

    def plot_data(
        self,
        data1,
        # data2=None,
        ax=None,
        scatter=False,
        title="receiver function data",
        xlabel="Times (s)",
        ylabel="Amplitude",
        **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()
        plotting_style = {
            "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 1)),
            "alpha": 1,
            "color": kwargs.pop("color", kwargs.pop("c", "blue")),
        }
        plotting_style.update(**kwargs)
        if scatter:
            ax.scatter(self._data_times, data1, **plotting_style)
        else:
            ax.plot(self._data_times, data1, **plotting_style)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax

    def misfit(self, data1, data2):
        raise NotImplementedError  # optional

    def log_likelihood(self, data1, data2):
        raise NotImplementedError  # optional

    def log_prior(self, model):
        raise NotImplementedError  # optional
