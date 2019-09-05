#!/bin/sh
#SBATCH --job-name=multi-2-reinforce
#SBATCH -A p_biomedicalmodel
#SBATCH --array=0-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-16:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/multi-2-reinforce.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_multi/sub_2/reinforce/fish_mouse_simsim/avg16_gauss30_raw_even_${SLURM_ARRAY_TASK_ID}/config.yml
