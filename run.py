import os
import subprocess
import time
from utils.our_utils import record

model = "amsterdam"

colmap_path = rf"./data/bungeenerf/{model}"
outputdir = rf"./output/test"

command1 = f'python train.py -s {colmap_path} -m {outputdir}/{model} --eval'

start = time.time()
subprocess.run(command1, shell=True)
end = time.time()
cost = end - start

with open(rf"./{outputdir}/{model}_metric.txt", "a", encoding="UTF-8") as rec:
    rec.write(model + "\n")
    rec.write(f"{cost}\n")

command2 = f'python render.py -m {outputdir}/{model} --skip_train --quiet'
subprocess.run(command2, shell=True)

command3 = f'python metrics.py -m {outputdir}/{model}'
result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
with open(rf"./{outputdir}/{model}/metric.txt", "a", encoding="UTF-8") as rec:
    rec.write(result.stdout)

record(model, outputdir)
