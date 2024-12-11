import numpy as np
import matplotlib.pyplot as plt
import warnings
import pyhk

from espresso import EspressoProblem
from espresso.exceptions import InvalidExampleError
from espresso.utils import absolute_path as path


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
        elif example_number == 3:
            self._description = "a field dataset from the computer programs in seismology"
            self._thicknesses, self._vs = read_mod_file(path(f"data/cps_rf_data/end.txt"))
            self._vp_vs = [1.77] * len(self._vs)
            self._t_shift = 5
            self._t_duration = 70
            self._t_sampling_interval = 0.5
            self._data_noise = 0.015        # estimated from the data
            self._all_ray_param_s_km = []
            self._all_gauss = []
            self._data_times = None
            self._all_data_rf = []
            # list all the *.txt files under data/cps_rf_data
            _dataset_files = path("data/cps_rf_data").glob("*_interpolated.txt")
            for file in _dataset_files:
                # e.g. file = data/cps_rf_data/rf_00_1.0_0.0658_interpolated.txt
                # extract the ray parameter and gauss from file name
                _gauss, _ray_param_s_km = file.stem.split("_")[-3:-1]
                self._all_ray_param_s_km.append(float(_ray_param_s_km))
                self._all_gauss.append(float(_gauss))
                # load the data
                _dataset = np.loadtxt(file)
                if self._data_times is None:
                    self._data_times = _dataset[:, 0]
                self._all_data_rf.append(_dataset[:, 1])
            self._data_rf = np.array(self._all_data_rf).reshape((-1,))
        else:
            raise InvalidExampleError

        self._good_model = form_layercake_model(self._thicknesses, self._vs)

        if self.example_number < 3:
            self._starting_model = np.ones(self._good_model.size)
            self._starting_model[::2] = 3.5
            self._starting_model[1::2] = 10
        else:
            h_start, vs_start = read_mod_file(path(f"data/cps_rf_data/start.txt"))
            self._starting_model = form_layercake_model(h_start, vs_start)

    @property
    def description(self):
        return self._description

    @property
    def model_size(self):
        return len(self._thicknesses) + len(self._vs)

    @property
    def data_size(self):
        return len(self._data_rf)

    @property
    def good_model(self):
        return self._good_model

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
            if self.example_number < 3:
                data_rf = pyhk.rf_calc(
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
                return data_rf
            else:
                all_data_pred = []
                for ray, gauss, data_rf in zip(
                    self._all_ray_param_s_km, self._all_gauss, self._all_data_rf
                ):
                    data_pred = pyhk.rf_calc(
                        ps=0, 
                        thik=thicknesses,
                        beta=vs,
                        kapa=self._vp_vs, 
                        p=ray, 
                        duration=self._t_duration,
                        dt=self._t_sampling_interval,
                        shft=self._t_shift,
                        gauss=gauss,
                    )
                    all_data_pred.append(data_pred)
                return np.array(all_data_pred).reshape((-1,))

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
        if self.example_number < 3:
            return plot_data(
                self._data_times, 
                self._data_rf,
                ax=ax,
                scatter=scatter,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                **kwargs,
            )
        else:
            if ax is not None:
                warnings.warn("ax is not used in this example, a list of axes will be returned")
            fig, axes = plt.subplots(13, 3, figsize=(10, 12))
            data = np.reshape(data1, (len(self._data_times), -1))
            for i, ax in enumerate(axes.flat):
                plot_data(
                    self._data_times, 
                    data[:, i], 
                    ax=ax, 
                    scatter=scatter,
                    title=f"ray (s/km) = {self._all_ray_param_s_km[i]}, gauss = {self._all_gauss[i]}",
                    xlabel=xlabel,
                    ylabel=ylabel,
                    **kwargs,
                )
            for ax in axes[:-1, :].flat:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
            for ax in axes[:, 1:].flat:
                ax.set_ylabel('')
            fig.tight_layout()
            return axes

    def misfit(self, data1, data2):
        raise NotImplementedError  # optional

    def log_likelihood(self, data1, data2):
        raise NotImplementedError  # optional

    def log_prior(self, model):
        raise NotImplementedError  # optional


def form_layercake_model(thicknesses, vs):
    model = np.zeros((len(vs) * 2 - 1))
    model[1::2] = thicknesses
    model[::2] = vs
    return model


def split_layercake_model(model):
    thicknesses = model[1::2]
    vs = model[::2]
    return thicknesses, vs


def read_mod_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    ref_model = []
    for line in lines[12:]:
        row = line.strip().split()
        ref_model.append([float(row[0]), float(row[2])])
    ref_model = np.array(ref_model)
    return ref_model[:-1, 0], ref_model[:, 1]

def plot_data(
    times: np.ndarray, 
    data: np.ndarray, 
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
        ax.scatter(times, data, **plotting_style)
    else:
        ax.plot(times, data, **plotting_style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


# Espresso -> EARTH SCIENCES -> Geophysics -> Seismology and seismic exploration -> Receiver function -> ReceiverFunctionInversionKnt
# description:  Receiver function inference problem based on a forward code by Brian Kennet and adapted by Lupei Zhu and Sheng Wang
