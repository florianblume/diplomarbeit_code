#!/bin/sh
#SBATCH --job-name=baseline-depth3-n2c-fish-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-18:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o baseline-depth3-n2c-fish-simsim.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_simsim/raw_all/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_simsim/raw_all_even/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_simsim/raw_part1/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_simsim/raw_part2/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/fish_simsim/raw_part3/config.yml

python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/simsim/all/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/simsim/part_1_2/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/simsim/part_1_3/config.yml
python src/train_model.py experiments/baseline/main_3_sub_3/n2c/joined/simsim/part_2_3/config.yml
