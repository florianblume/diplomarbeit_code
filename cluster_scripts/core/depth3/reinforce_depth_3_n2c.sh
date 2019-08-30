#!/bin/sh
#SBATCH --job-name=reinforce-depth3-n2c
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o reinforce-core-depth3-n2c.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_core/reinforce/main_3_sub_3/n2c/fish_mouse/avg16_gauss30/config.yml
python src/train_model.py experiments_core/reinforce/main_3_sub_3/n2c/fish_simsim/raw_all_even/config.yml
