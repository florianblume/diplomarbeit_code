#!/bin/sh
#SBATCH --job-name=baseline-fish-mouse
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-10:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o baseline-fish-mouse.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/joined/fish_mouse/avg8_gauss30/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/avg16_gauss30/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/gauss15_gauss15/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/gauss15_gauss30/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/gauss30_gauss30/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/raw/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/raw_gauss15/config.yml
python src/train_model.py experiments/baseline/joined/fish_mouse/raw_gauss30/config.yml