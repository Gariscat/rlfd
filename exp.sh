#!/bin/bash
#SBATCH -J rlfd_grid_exp
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --mem=32GB
### SBATCH --mail-type=END
### SBATCH --mail-user=xl3133@nyu.edu
#SBATCH --array=1-3
#SBATCH --output=runs/output_%a.out
#SBATCH --error=runs/output_%a.err

conda init bash
source ~/.bashrc 

module load anaconda3 cuda/11.3.1
source activate mlfd

which python
nvidia-smi

cd ~/rlfd/
python grid_exp.py --tot_steps 20_000 --task_id ${SLURM_ARRAY_TASK_ID}