#!/bin/sh
#SBATCH --job-name=avg-pixel-fish-mouse
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o avg-pixel-fish-mouse.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/pixel/config.yml
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/pixel_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_mouse/avg16_gauss30/pixel_entropy_less/config.yml