#!/bin/sh
#SBATCH --job-name=crashed
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-12:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/crashed.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_it/probabilistic/fish_mouse/avg16_gauss30/pixel_0/config.yml
python src/train_model.py experiments_multi/sub_3/probabilistic/fish_mouse_simsim/avg16_gauss30_raw_even/pixel_2/config.yml
python src/train_model.py experiments_multi/sub_3/reinforce/fish_mouse_simsim/avg16_gauss30_raw_even_3/config.yml
python src/train_model.py experiments_multi/sub_3/reinforce/fish_mouse_simsim/avg16_gauss30_raw_even_4/config.yml
