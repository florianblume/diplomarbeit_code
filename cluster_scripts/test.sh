#!/bin/sh
#SBATCH --job-name=test
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-18:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/test.log
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

# About 9 configs

python src/train_model.py experiments/average/test/config.yml
