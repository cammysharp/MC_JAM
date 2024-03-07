#!/bin/bash
#
#SBATCH --job-name=test_mpi
#
#SBATCH --ntasks=1
#SBATCH --time=01:10:00
#SBATCH --mem=10G

ml python-scientific/3.10.4-foss-2022a
source /fred/oz059/cammy/environments/MJ/bin/activate
python mpi_emcee_JAM.py
deactivate
