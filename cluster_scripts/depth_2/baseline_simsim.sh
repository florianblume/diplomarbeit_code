#!/bin/sh
#SBATCH --job-name=depth-2-baseline-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH --array=1-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-01:30:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/depth-2-baseline-simsim.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_depth_2/baseline/simsim/all_${SLURM_ARRAY_TASK_ID}/config.yml
