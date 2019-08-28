#!/bin/sh
#SBATCH --job-name=depth2
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-16:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o depth2.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

#python src/train_model.py experiments/baseline/main_2/n2c/joined/fish_simsim/raw_all/config.yml
#python src/train_model.py experiments/baseline/main_2/n2c/joined/simsim/raw_all/config.yml
#python src/train_model.py experiments/baseline/main_2/n2c/fish/avg16/config.yml

#python src/train_model.py experiments/average/main_2_sub_2/n2c/fish_simsim/raw_all/image/config.yml
#python src/train_model.py experiments/average/main_2_sub_2/n2c/fish_simsim/raw_all/pixel/config.yml

python src/train_model.py experiments/probabilistic/main_2_sub_2/n2c/fish_simsim/raw_all/image/config.yml
python src/train_model.py experiments/probabilistic/main_2_sub_2/n2c/fish_simsim/raw_all/pixel/config.yml
