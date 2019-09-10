#!/bin/sh
#SBATCH --job-name=depth-2-baseline-mouse
#SBATCH -A p_biomedicalmodel
#SBATCH --array=1-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/depth-2-baseline-mouse.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

# About 9 configs

python src/train_model.py experiments_depth_2/baseline/mouse/raw_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_depth_2/baseline/mouse/gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_depth_2/baseline/mouse/gauss60_${SLURM_ARRAY_TASK_ID}/config.yml
