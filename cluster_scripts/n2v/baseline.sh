#!/bin/sh
#SBATCH --job-name=n2v-baseline
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-12:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/n2v-baseline-remainder.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

# About 9 configs

#for file in $(find experiments_n2v/baseline -name config.yml); do
#	echo $file;
#done


python src/train_model.py experiments_n2v/baseline/fish/avg16_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/fish/raw_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/mouse/gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/simsim/all_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/fish_mouse/avg16_gauss30_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_n2v/baseline/joined/fish_simsim/raw_all_even_${SLURM_ARRAY_TASK_ID}/config.yml