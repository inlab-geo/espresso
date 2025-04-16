"""
This script was used to prepare for the dataset used in the 3rd receiver function
inversion example.

The original dataset is downloaded from:
https://www.eas.slu.edu/eqc/eqc_cps/TUTORIAL/STRUCT/index.html

A copy of the original dataset, including the output from the Computer Programs in
Seismology (CPS) software, is also available here:
https://github.com/inlab-geo/cofi-examples/tree/main/examples/surface_wave_trans_d/data

To run this script, the folder above should be downloaded and placed in the same
directory as this script, and should be named "cps_data". The rf shared library should
also be built and placed in the same directory as this script.

```bash
python prep_data_3.py
```
"""

import shutil
import obspy
import numpy as np

import rf

cps_rf_data_prefix = "cps_data/RFTN"
output_rf_data_prefix = "cps_rf_data"

t_duration = 70
t_sampling_interval = 0.5
t_shift = 5
    

def get_all_file_names():
    lst_files_rftn_data = f"{cps_rf_data_prefix}/rftn.lst"
    with open(lst_files_rftn_data, "r") as file:
        lines = file.readlines()
    files = [line.strip() for line in lines if line]
    return files


def read_data_from_files(files):
    rftn_data_times = []
    rftn_data_amplitudes = []
    rftn_data_gauss = []
    rftn_data_rays = []
    for file_rftn_data in files:
        st = obspy.read(f"{cps_rf_data_prefix}/{file_rftn_data}", debug_headers=True)
        rftn_data_times.append(st[0].times() + st[0].stats.sac.b)
        rftn_data_amplitudes.append(st[0].data)
        rftn_data_gauss.append(st[0].stats.sac.user0)
        rftn_data_rays.append(st[0].stats.sac.user4)
    return rftn_data_times, rftn_data_amplitudes, rftn_data_gauss, rftn_data_rays


def write_original_data_to_files(
    rftn_data_times, rftn_data_amplitudes, rftn_data_gauss, rftn_data_rays
):
    for i, (times, amplitudes, gauss, rays) in enumerate(
        zip(rftn_data_times, rftn_data_amplitudes, rftn_data_gauss, rftn_data_rays)
    ):
        rf_data = np.vstack((times, amplitudes)).T
        f_name = f"{output_rf_data_prefix}/rf_{i:02}_{gauss}_{rays:.4f}.txt"
        np.savetxt(f_name, rf_data)
        print(f"Saved rf data of shape {rf_data.shape} to {f_name}")


def read_mod_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    ref_model = []
    for line in lines[12:]:
        row = line.strip().split()
        ref_model.append([float(row[0]), float(row[2])])
    ref_model = np.array(ref_model)
    return ref_model[:-1, 0], ref_model[:, 1]


def write_interpolated_data_to_files(
    rftn_data_times, rftn_data_amplitudes, rftn_data_gauss, rftn_data_rays
):
    mod_file = f"{output_rf_data_prefix}/end.mod"
    h, vs = read_mod_file(mod_file)
    # try to run the rf_calc function to get times
    
    d_pred = rf.rf_calc(
        ps=0,
        thik=h,
        beta=vs,
        kapa=np.ones(len(vs)) * 1.77,
        p=0.0658,
        duration=t_duration,
        dt=t_sampling_interval,
        shft=t_shift,
        gauss=1.0,
    )
    d_pred_times = np.arange(len(d_pred)) * t_sampling_interval - t_shift
    for i, (times, amplitudes, gauss, rays) in enumerate(
        zip(rftn_data_times, rftn_data_amplitudes, rftn_data_gauss, rftn_data_rays)
    ):
        d_obs_interpolated = np.interp(d_pred_times, times, amplitudes)
        rf_data = np.vstack((d_pred_times, d_obs_interpolated)).T
        f_name = f"{output_rf_data_prefix}/rf_{i:02}_{gauss}_{rays:.4f}_interpolated.txt"
        np.savetxt(f_name, rf_data)
        print(f"Saved interpolated rf data of shape {rf_data.shape} to {f_name}")


def move_model_files():
    shutil.copy(f"{cps_rf_data_prefix}/start.mod", f"{output_rf_data_prefix}/start.txt")
    print(f"Moved start.mod to {output_rf_data_prefix}/start.txt")
    shutil.copy(f"{cps_rf_data_prefix}/end.mod", f"{output_rf_data_prefix}/end.txt")
    print(f"Moved end.mod to {output_rf_data_prefix}/end.txt")


if __name__ == "__main__":
    files = get_all_file_names()
    times, amplitudes, gauss, rays = read_data_from_files(files)
    write_original_data_to_files(times, amplitudes, gauss, rays)
    write_interpolated_data_to_files(times, amplitudes, gauss, rays)
    move_model_files()
