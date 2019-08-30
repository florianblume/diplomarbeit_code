#!/bin/sh
#SBATCH --job-name=baseline-depth3-fish-mouse-i
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-05:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o baseline-depth3-fish-mouse-i.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments/baseline/main_3/n2c/joined/fish_mouse/avg16_gauss30_1/config.yml
python src/train_model.py experiments/baseline/main_3/n2c/joined/fish_mouse/avg16_gauss30_2/config.yml
python src/train_model.py experiments/baseline/main_3/n2c/joined/fish_mouse/avg16_gauss30_3/config.yml
python src/train_model.py experiments/baseline/main_3/n2c/joined/fish_mouse/avg16_gauss30_4/config.yml
