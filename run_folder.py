import os
import subprocess
import time
from utils.our_utils import record


def func(dataset_name):
    datapath = rf"./data/{dataset_name}"
    for model in os.listdir(datapath):
        colmap_path = rf"{datapath}/{model}"
        outputdir = rf"./output"
        os.makedirs(outputdir, exist_ok=True)
        if os.path.exists(f"{outputdir}/{model}"):
            continue
        command1 = f'python train.py -s {colmap_path} -m {outputdir}/{model} --eval '
        start = time.time()
        subprocess.run(command1, shell=True)
        end = time.time()
        cost = end - start

        with open(rf"./{outputdir}/{model}_metric.txt", "a", encoding="UTF-8") as rec:
            rec.write(model+"\n")
            rec.write(f"{cost}\n")

        command2 = f'python render.py -m {outputdir}/{model} --skip_train --quiet'
        subprocess.run(command2, shell=True)

        command3 = f'python metrics.py -m {outputdir}/{model}'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"{outputdir}/{model}_metric.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)

        record(model, outputdir)


# func("tat")
# func("db")
# func("m360")
# func("bungeenerf")









