#!/bin/sh
#SBATCH --job-name=baseline-it-fish-fish
#SBATCH --array=0-4
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-06:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/baseline-it-fish-fish-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_it/baseline/joined/fish_only/avg16_avg16/config.yml
python src/train_model.py experiments_it/baseline/joined/fish_only/avg16_avg16_${SLURM_ARRAY_TASK_ID}/config.yml
