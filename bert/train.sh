#!/bin/bash -l
#SBATCH -J Bert            # Display Name
#SBATCH -N 1                # Number of nodes
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -p gpu              # Job-type, can be batch (only CPU) or gpu
#SBATCH -G 1                # Number of GPUs for the job
#SBATCH --time=0-30:00:00   # Time limit
#SBATCH --qos normal


conda activate transformer
nvidia-smi
python main.py
