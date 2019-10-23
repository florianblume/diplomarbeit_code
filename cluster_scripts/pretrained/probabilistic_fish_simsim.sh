#!/bin/sh
#SBATCH --job-name=probabilistic-pretrained-fish-simsim
#SBATCH --array=0-4
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-10:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/probabilistic-pretrained-fish-simsim-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_pretrained/probabilistic/fish_simsim/raw_all_even/image_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_pretrained/probabilistic/fish_simsim/raw_all_even/pixel_${SLURM_ARRAY_TASK_ID}/config.yml
