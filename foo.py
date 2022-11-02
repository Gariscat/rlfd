from cmath import log
import os
import matplotlib.pyplot as plt

with open('../Downloads/trace.log', 'r') as f:
    log_lines = f.readlines()

print(len(log_lines))