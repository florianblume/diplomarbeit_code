#!/bin/sh
#SBATCH --job-name=missing
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-18:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/missing.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

# About 9 configs

python src/train_model.py experiments_it/probabilistic/fish_mouse/avg16_gauss30/pixel_3/config.yml
python src/train_model.py experiments_it/probabilistic/fish_mouse/avg16_gauss30/pixel_4/config.yml
