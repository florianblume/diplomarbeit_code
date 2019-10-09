#!/bin/sh
#SBATCH --job-name=n2v-baseline
#SBATCH -A p_biomedicalmodel
#SBATCH --array=0-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-18:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/n2v-baseline-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf


python src/train_model.py experiments_n2v/baseline/fish/avg16_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/fish/raw_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/mouse/gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/simsim/all_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/fish_mouse/avg16_gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/fish_simsim/raw_all_even_${SLURM_ARRAY_TASK_ID}/config.yml
