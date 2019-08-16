#!/bin/sh
#SBATCH --job-name=baseline-simsim-only
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o baseline-simsim-only.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/simsim/part1/config.yml
python src/train_model.py experiments/baseline/simsim/part2/config.yml
python src/train_model.py experiments/baseline/simsim/part3/config.yml