#!/bin/sh
#SBATCH --job-name=baseline-depth3-n2v
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-04:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o baseline-depth3-n2v.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

#python src/train_model.py experiments/baseline/main_3/n2v/fish/avg16/config.yml
#python src/train_model.py experiments/baseline/main_3/n2v/fish/raw/config.yml

#python src/train_model.py experiments/baseline/main_3/n2v/joined/fish_mouse/avg16_gauss30/config.yml
#python src/train_model.py experiments/baseline/main_3/n2v/joined/fish_simsim/raw_all/config.yml
#python src/train_model.py experiments/baseline/main_3/n2v/joined/simsim/all/config.yml
python src/train_model.py experiments/baseline/main_3/n2v/joined/fish_simsim/raw_all_even/config.yml
