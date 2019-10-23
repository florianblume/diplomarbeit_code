#!/bin/sh
#SBATCH --job-name=probabilistic-fish-mouse-crashed-10.10
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/probabilistic-fish-mouse-crashed-10.10.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_it/probabilistic/fish_mouse/avg16_gauss30/pixel_0/config.yml
