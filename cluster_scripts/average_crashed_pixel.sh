#!/bin/sh
#SBATCH --job-name=average-pixel-crashed
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-15:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/average-pixel-crashed.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_pretrained/average/fish_mouse/avg16_gauss30/pixel_0/configy.yml
python src/train_model.py experiments_pretrained/average/fish_mouse/avg16_gauss30/pixel_1/configy.yml
python src/train_model.py experiments_pretrained/average/fish_mouse/avg16_gauss30/pixel_3/configy.yml
python src/train_model.py experiments_pretrained/average/fish_mouse/avg16_gauss30/pixel_4/configy.yml
python src/train_model.py experiments_pretrained/average/fish_simsim/raw_all_even/pixel_1/config.yml
