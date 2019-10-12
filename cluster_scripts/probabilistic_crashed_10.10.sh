#!/bin/sh
#SBATCH --job-name=probabilistic-crashed-10.10
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-08:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/probabilistic-crashed-10.10.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_it/probabilistic/fish_simsim/raw_all_even/pixel_1/config.yml
python src/train_model.py experiments_it/probabilistic/fish_simsim/raw_all_even/pixel_3/config.yml
