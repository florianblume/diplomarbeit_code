#!/bin/sh
#SBATCH --job-name=qlearning-pretrained-fish-simsim
#SBATCH --array=0-4
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/qlearning-pretrained-fish-simsim-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_pretrained/q_learning/fish_simsim/raw_all_even_${SLURM_ARRAY_TASK_ID}/config.yml
