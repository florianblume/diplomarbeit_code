#!/bin/sh
#SBATCH --job-name=depth-4-baseline-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH --array=0-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-03:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/depth-4-baseline-simsim-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_depth_4/baseline/simsim/all_${SLURM_ARRAY_TASK_ID}/config.yml
