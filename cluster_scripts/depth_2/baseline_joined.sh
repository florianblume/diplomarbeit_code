#!/bin/sh
#SBATCH --job-name=depth-2-baseline-joined
#SBATCH -A p_biomedicalmodel
#SBATCH --array=1-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/depth-2-baseline-joined.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

# About 9 configs

python src/train_model.py experiments_depth_2/baseline/joined/fish_mouse/avg16_gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_depth_2/baseline/joined/fish_mouse/gauss60_gauss60_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_depth_2/baseline/joined/fish_simsim/raw_all_even_${SLURM_ARRAY_TASK_ID}/config.yml
