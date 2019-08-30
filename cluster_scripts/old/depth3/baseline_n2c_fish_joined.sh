#!/bin/sh
#SBATCH --job-name=baseline-depth3-n2c-fish-joined
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o baseline-depth3-n2c-fish-joined.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_only/avg16_gauss30/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_only/avg16_raw/config.yml
