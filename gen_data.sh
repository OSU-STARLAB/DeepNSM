#!/bin/bash
#SBATCH -J nsm_data_gen
#SBATCH --time=72:00:00
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH -o nsm_data_gen.out
#SBATCH -e nsm_data_gen.err
module load cuda openssl
cd /nfs/hpc/share/baartmar/NSM
source .venv/bin/activate
python generate_dataset.py