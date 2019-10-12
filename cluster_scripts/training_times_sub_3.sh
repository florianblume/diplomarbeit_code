#!/bin/sh
#SBATCH --job-name=training_times_sub_3
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-20:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/training_times_sub_3.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_times/average/sub_3/config.yml
python src/train_model.py experiments_times/probabilistic/sub_3/config.yml
python src/train_model.py experiments_times/q_learning/sub_3/config.yml
python src/train_model.py experiments_times/reinforce/sub_3/config.yml
