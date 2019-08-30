#!/bin/sh
#SBATCH --job-name=baseline-depth3-n2c-single-fish
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-12:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o baseline-depth3-n2c-single-fish.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/fish/avg8/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/fish/avg16/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/fish/gauss15/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/fish/gauss30/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/fish/raw/config.yml

python src/train_model.py experiments/baseline/main_3_sub_3/n2c/higher_depth/config.yml
