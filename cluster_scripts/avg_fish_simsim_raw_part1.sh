#!/bin/sh
#SBATCH --job-name=avg-fish-simsim-part1
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-09:30:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o avg-fish-simsim-part1.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/image/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/image_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/image_entropy_less/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/pixel/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/pixel_entropy/config.yml
python src/train_model.py experiments/average/joined/fish_simsim/raw_part1/pixel_entropy_less/config.yml