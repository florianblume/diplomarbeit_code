#!/bin/sh
#SBATCH --job-name=n2v-average-fish-simsim
#SBATCH --array=0-4
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-09:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/n2v-average-fish-simsim-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf


python src/train_model.py experiments_n2v/average/fish_simsim/raw_all_even/image_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/average/fish_simsim/raw_all_even/pixel_${SLURM_ARRAY_TASK_ID}/config.yml
