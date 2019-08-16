#!/bin/sh
#SBATCH --job-name=avg-image-fish-mouse
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o avg-image-fish-mouse.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/image/config.yml
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/image_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/image_entropy_less/config.yml