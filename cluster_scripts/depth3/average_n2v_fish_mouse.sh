#!/bin/sh
#SBATCH --job-name=average-depth3-n2v-fish-mouse
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o average-depth3-n2v-fish-mouse.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf
#python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_mouse/avg16_gauss30/image/config.yml
python src/train_model.py experiments/average/main_3_sub_3/n2v/fish_mouse/avg16_gauss30/pixel/config.yml
