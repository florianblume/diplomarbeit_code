#!/bin/sh
#SBATCH --job-name=baseline-fish-simsim
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:30:00
#SBATCH --mem 8G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o baseline-fish-simsim.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/joined/fish_simsim/raw_all/config.yml