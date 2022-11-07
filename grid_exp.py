import subprocess
import os

TOT_STEPS = 1_000_000


for seed in (0, ):
    for obs_ord in (1, 2, 3):
        for features_dim in (16, 32):
            # for wptype in [0.25, 0.5, 0.75]:
            for num_layers in (1, 2, 3):
                subprocess.call(f'python train.py \
                    --seed={seed} \
                    --obs_ord={obs_ord} \
                    --features_dim={features_dim} \
                    --num_layers={num_layers}', \
                    shell=True
                )