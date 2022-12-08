#!/bin/bash

conda activate mlfd

nvidia-smi

cd ~/rlfd/

nohup python grid_exp.py --obs_ord 1 >> runs/obs_ord_1.out
nohup python grid_exp.py --obs_ord 2 >> runs/obs_ord_2.out
nohup python grid_exp.py --obs_ord 3 >> runs/obs_ord_3.out

# record time 2022.12.8 9:54