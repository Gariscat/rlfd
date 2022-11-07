import subprocess
import os
import argparse

TOT_STEPS = 500_000

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=-1)
args = parser.parse_args()
assert args.task_id > 0
# task_id == obs_ord
for seq_len in (64, 128, 256):
    for features_dim in (16, 32):
        # for wptype in [0.25, 0.5, 0.75]:
        for num_layers in (1, 2, 3):
            subprocess.call(f'python train.py \
                --obs_ord={args.task_id} \
                --features_dim={features_dim} \
                --num_layers={num_layers} \
                --tot_steps={TOT_STEPS}', \
                shell=True
            )