#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/myenv/bin/activate
cd ~/gan_cifar10

export PYTHONUNBUFFERED=1

python scripts/train_gan.py --config configs/$1.yaml
python scripts/evaluate.py --variant $1