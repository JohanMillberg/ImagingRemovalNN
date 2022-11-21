#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p core -n 1
#SBATCH -t 60:00

module load gcc openmpi
module load python/3.5.0.
module load python_ML_packages/3.9.5-gpu
export OMPI_MCA_btl_openib_allow_ib=1
make

echo "Starting test"

echo "TestFinished Finished"