import numpy as np

import rf


example_1 = np.array([[1, 2.5, 1.7],          # used in example_number=1,2,3
                    [3.5, 3.0, 1.7],
                    [8.0, 3.5, 2.0],
                    [20, 3.9, 1.7],
                    [45, 4.4,1.7]])

example_4 = np.array([[8.0, 3.0, 1.7],          # used in example_number=4
                    [20, 3.9, 1.7],
                    [45, 4.4, 1.7]])


def prep_data(example_model):
    times, noisy_data = rf.rfcalc(example_model, sn=0.3)
    times, clean_data = rf.rfcalc(example_model, sn=0.0)
    return times, clean_data, noisy_data

def estimate_noise(noisy_data, clean_data):
    noise = noisy_data - clean_data
    corr_noise_res = np.correlate(noise, noise, mode='full')
    corr_noise_res = corr_noise_res[corr_noise_res.size // 2:]
    correlation_length = np.where(corr_noise_res < 1e-3)[0][0]
    return correlation_length, np.std(noise)

def save_data(times, noisy_data, dataset_path="dataset1.txt"):
    dataset = np.column_stack((times, noisy_data))
    np.savetxt(dataset_path, dataset)


if __name__ == "__main__":
    # examples 1-3
    times, clean_data, noisy_data = prep_data(example_1)
    correlation_length, noise_std = estimate_noise(noisy_data, clean_data)
    save_data(times, noisy_data)

    # example 4
    times, clean_data, noisy_data = prep_data(example_4)
    correlation_length, noise_std = estimate_noise(noisy_data, clean_data)
    save_data(times, noisy_data, dataset_path="dataset4.txt")
