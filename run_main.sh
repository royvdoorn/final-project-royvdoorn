#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=9:00:00
#SBATCH --mem-per-cpu=20G

cd /gpfs/home6/scur0756/Final_Assignment/final-project-royvdoorn/
mkdir wandb/$SLURM_JOBID

srun apptainer exec --nv /gpfs/work5/0/jhstue005/JHS_data/5lsm0_v1.sif /bin/bash run_container.sh