import os

fn = os.path.join(os.path.dirname(__file__), "output", "experiments.txt")
start = "python xdo.py"
with open(fn) as f:
    for line in f.read().split("\n"):
        if line.startswith(start):
            args = line[len(start) :].strip().strip(";")
            command = f'sbatch {os.path.join(os.path.dirname(__file__), "output", "run_experiment.sh")} "{args}"'
            os.system(command)
