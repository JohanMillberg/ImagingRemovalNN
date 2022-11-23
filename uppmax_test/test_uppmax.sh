#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p core -n 1
#SBATCH -t 60:00

module load python/3.9.5
module load python_ML_packages/3.9.5-gpu

echo "Starting test"
echo "Run 'test.py'"
python3 ./test.py

echo " "
echo "TestFinished Finished"
