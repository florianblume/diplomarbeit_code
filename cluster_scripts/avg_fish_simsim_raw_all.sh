#!/bin/sh
#SBATCH --job-name=avg-fish-simsim-raw
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-09:30:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o avg-fish-simsim-raw.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/image/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/image_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/image_entropy_less/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/pixel/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/pixel_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_all/pixel_entropy_less/config.yml