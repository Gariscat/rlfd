import torch
import numpy as np

def default_L1(target, pred):
    dis = np.abs(pred-target).item()
    return np.exp(-dis)


