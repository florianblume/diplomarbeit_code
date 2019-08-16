#!/bin/sh
#SBATCH --job-name=baseline-simsim-joined
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:30:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o baseline-simsim-joined.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
python src/train_model.py experiments/baseline/joined/simsim/all/config.yml
python src/train_model.py experiments/baseline/joined/simsim/part_1_2/config.yml
python src/train_model.py experiments/baseline/joined/simsim/part_1_3/config.yml
python src/train_model.py experiments/baseline/joined/simsim/part_2_3/config.yml