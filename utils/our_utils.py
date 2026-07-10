import os
import csv
from utils.image_utils import psnr
import pandas as pd
import numpy as np
import GPUtil
import subprocess
import re


def recording(image, gt_image, args, iteration, gaussians, init_mem_use, it=100):
    path = os.path.abspath(args.model_path)

    if iteration % it == 0:
        # result = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        # memory_usage_pattern = r"(\d+)MiB /"
        # memory_usage = re.findall(memory_usage_pattern, result)
        # used_memory = int(memory_usage[0]) - init_mem_use
        gpu = GPUtil.getGPUs()[0]
        used_memory = gpu.memoryUsed - init_mem_use
        with open(rf'{path}/memory_used.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([used_memory])

    with open(rf'{path}/psnr.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(psnr(image, gt_image).mean().flatten().tolist())

    point_num = gaussians.get_xyz.shape[0]
    with open(rf'{path}/points_num.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([point_num])

    point_3_num = gaussians.get_sh3.shape[0]
    with open(rf'{path}/points_3_num.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([point_3_num])


def get_folder_size(folder_path):
    """递归计算文件夹的总大小（字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def record(model, outputdir):
    n0gs = rf"{outputdir}/{model}/points_num.csv"
    n0gs_n = np.array(pd.read_csv(n0gs, header=None))
    n0gs_n_Peak = np.max(n0gs_n)
    n0gs_n_Avg = np.mean(n0gs_n)
    n0gs_n_Final = n0gs_n[-1, 0]
    n3gs = rf"{outputdir}/{model}/points_3_num.csv"
    n3gs_n = np.array(pd.read_csv(n3gs, header=None))
    n3gs_n_Peak = np.max(n3gs_n)
    n3gs_n_Avg = np.mean(n3gs_n)
    n3gs_n_Final = n3gs_n[-1, 0]
    Storage = get_folder_size(rf"{outputdir}/{model}/point_cloud/iteration_30000") / 1000_000
    with open(rf'{outputdir}/{model}_metric.txt', 'a') as file:
        file.write(f"{n0gs_n_Peak}\n")
        file.write(f"{n0gs_n_Avg}\n")
        file.write(f"{n0gs_n_Final}\n")
        file.write(f"{n3gs_n_Peak}\n")
        file.write(f"{n3gs_n_Avg}\n")
        file.write(f"{n3gs_n_Final}\n")
        file.write(f"{Storage}")
        file.flush()

    ngs = rf"{outputdir}/{model}/memory_used.csv"
    data_n = pd.read_csv(ngs, header=None)
    data_n = np.array(data_n)
    max_n0 = np.max(data_n[0:150])
    max_n1 = np.max(data_n[150:])
    max_n = np.max(data_n)
    mean_n = np.mean(data_n)
    with open(rf'{outputdir}/{model}_metric.txt', 'a') as file:
        #file.write(f"{max_o}\n")
        file.write(f"{max_n0}\n")
        file.write(f"{max_n1}\n")
        file.write(f"{max_n}\n")
        file.write(f"{mean_n}\n")
        file.flush()


