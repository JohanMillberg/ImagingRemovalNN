#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p node -N 1
#SBATCH -t 5:00:00
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1

module load python3/3.9.5
module load python_ML_packages/3.9.5-gpu

pip install scipy
pip install cupy-cuda111

echo "Generate images..."
echo "Calculating I matrices..."
python3 main.py 100

echo " "
echo "Finished calculations"
