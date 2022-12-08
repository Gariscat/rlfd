import subprocess
import os
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--tot_steps', type=int, default=100_000)
parser.add_argument('--obs_ord', type=int, default=-1)
args = parser.parse_args()
assert args.obs_ord > 0
# task_id == obs_ord
    
if __name__ == '__main__':
    print(args)
    for reward_func in ("DefaultL1", "L1PunishFP", ):
        for features_dim in (16, 32,):
            for num_layers in (1, 2,):
                subprocess.call(f'python train.py \
                    --obs_ord {args.obs_ord} \
                    --reward_func {reward_func} \
                    --features_dim {features_dim} \
                    --num_layers {num_layers} \
                    --tot_steps {args.tot_steps}', \
                    shell=True
                )