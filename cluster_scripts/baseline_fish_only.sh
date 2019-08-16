#!/bin/sh
#SBATCH --job-name=baseline-fish-only
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-02:30:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o baseline-fish-only.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/joined/fish_only/avg16_gauss30/config.yml
python src/train_model.py experiments/baseline/joined/fish_only/avg16_raw/config.yml