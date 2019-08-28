#!/bin/sh
#SBATCH --job-name=probabilistic-depth3-n2v
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-02:30:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o probabilistic-depth3-n2v.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/fish/raw/subnetwork/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_mouse/avg16_gauss30/image/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_mouse/avg16_gauss30/pixel/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/image/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/image_even/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/image_even_weight_multiplier/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/pixel/config.yml
#python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/pixel_even/config.yml
python src/train_model.py experiments/probabilistic/main_3_sub_3/n2v/joined/fish_simsim/raw_all/pixel_even_weight_multiplier/config.yml
