#!/bin/sh
#SBATCH --job-name=qlearning-depth3-n2c-decaying
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-06:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/qlearning-depth3-n2c-decaying.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments/q_learning/main_3_sub_3/n2c/fish_simsim/raw_all_even/decaying/config.yml
python src/train_model.py experiments/q_learning/main_3_sub_3/n2c/fish_simsim/raw_all_even/non_decaying/config.yml
