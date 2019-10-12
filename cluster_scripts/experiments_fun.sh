#!/bin/sh
#SBATCH --job-name=experiments_fun
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-07:30:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/experiments_fun.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

#python src/train_model.py experiments_fun/baseline/fish/avg16/config.yml
#python src/train_model.py experiments_fun/baseline/mouse/gauss30/config.yml
python src/train_model.py experiments_fun/baseline/joined/fish_mouse/avg16_gauss30/config.yml
