export WANDB_API_KEY=a592f5ab8f3ee87682f0119deb65558cc06bd357
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
