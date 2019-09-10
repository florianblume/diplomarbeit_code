#!/bin/sh
#SBATCH --job-name=multi-3-reinforce-remainder
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/multi-3-reinforce-remainder-%A-%a.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_multi/sub_3/reinforce/fish_mouse_simsim/avg16_gauss30_raw_even_3/config.yml
