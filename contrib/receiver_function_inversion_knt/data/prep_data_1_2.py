import numpy as np

import rf


example_1 = {
    "thicknesses": [10,  20,  0.0],
    "vs": [3.3, 3.4, 4.5],
    "vp_vs": [1.732, 1.732, 1.732],
    "ray_param_s_km": 0.07,
    "t_shift": 5,
    "t_duration": 50,
    "t_sampling_interval": 0.5,
    "gauss": 1.0,
    "data_noise": 0.02, 
    "dataset_path": "./dataset1.txt"
}

example_2 = {
    "thicknesses": [10, 10, 15, 20, 20, 20, 20, 20, 0],
    "vs": [3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5], 
    "vp_vs": [1.77] * 9,
    "ray_param_s_km": 0.07,
    "t_shift": 5,
    "t_duration": 25,
    "t_sampling_interval": 0.1,
    "gauss": 1.0,
    "data_noise": 0.02, 
    "dataset_path": "./dataset2.txt"
}

def prep_data(example_dict):
    data = rf.rf_calc(
        ps=0, 
        thik=example_dict["thicknesses"], 
        beta=example_dict["vs"], 
        kapa=example_dict["vp_vs"], 
        p=0.07, 
        duration=example_dict["t_duration"], 
        dt=example_dict["t_sampling_interval"], 
        shft=example_dict["t_shift"], 
        gauss=example_dict["gauss"]
    )
    noisy_data = data + np.random.normal(0, example_dict["data_noise"], data.shape)
    times = np.arange(data.size) * example_dict["t_sampling_interval"] - example_dict["t_shift"]
    dataset = np.column_stack((times, noisy_data))
    dataset_path = example_dict["dataset_path"]
    np.savetxt(dataset_path, dataset)
    return data, noisy_data, times


if __name__ == "__main__":
    prep_data(example_1)
    prep_data(example_2)
