#!/bin/bash
#SBATCH -J rlfd_grid_exp
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --mem=32GB
### SBATCH --mail-type=END
### SBATCH --mail-user=xl3133@nyu.edu
#SBATCH --array=1-3
#SBATCH --output=runs/output.out
#SBATCH --error=runs/output.err

conda init bash
source ~/.bashrc 

module load anaconda3 cuda/11.3.1
source activate mlfd

which python
nvidia-smi

cd ~/rlfd/
python grid_exp.py --task_id ${SLURM_ARRAY_TASK_ID}
