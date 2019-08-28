#!/bin/sh
#SBATCH --job-name=average-depth3-n2c-fish-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-20:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o average-depth3-n2c-fish-simsim.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/image/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/image_entropy/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/image_entropy_less/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/image_even/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/image_even_multiplier/config.yml

python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/pixel/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/pixel_entropy/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/pixel_entropy_less/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/pixel_even/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2c/fish_simsim/raw_all/pixel_even_multiplier/config.yml
