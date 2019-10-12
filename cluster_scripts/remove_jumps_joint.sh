#!/bin/sh
#SBATCH --job-name=remove-jumps-joint
#SBATCH -A p_biomedicalmodel
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-6:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/remove-jumps=joint.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=ml

python src/train_model.py experiments_depth_2/baseline/joined/fish_mouse/avg16_gauss30_0/config.yml
python src/train_model.py experiments_depth_2/baseline/joined/fish_simsim/raw_all_even_0/config.yml
