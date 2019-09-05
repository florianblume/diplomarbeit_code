#!/bin/sh
#SBATCH --job-name=stuff-probabilistic
#SBATCH -A p_biomedicalmodel
#SBATCH --array=0-4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-08:00:00
#SBATCH --mem 64G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=florian.blume@mailbox.tu-dresden.de
#SBATCH -o logs/stuff-probabilistic.log
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=hpdlf

python src/train_model.py experiments_stuff/probabilistic/two_subnets/fish_only/image_${SLURM_ARRAY_TASK_ID}/config.yml
python src/train_model.py experiments_stuff/probabilistic/two_subnets/fish_only/pixel_${SLURM_ARRAY_TASK_ID}/config.yml
