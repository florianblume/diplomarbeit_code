#!/bin/sh
#SBATCH --job-name=average-depth3-n2v-fish-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-10:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o average-depth3-n2v-fish-simsim.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_simsim/raw_all/image/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_simsim/raw_all/image_even/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_simsim/raw_all/pixel/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_simsim/raw_all/pixel_even/config.yml
