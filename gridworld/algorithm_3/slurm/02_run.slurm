#!/bin/bash

#SBATCH --job-name=9w5qngfw
#SBATCH --exclude=adriatic[01-06],cheetah01,cheetah02,cheetah03,cheetah04,jaguar01,jaguar02,jaguar04,lotus,titanx[01-03],titanx[04-06],lynx[01-02]
#SBATCH --partition=nolim,main,gnolim,gpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4                 # one CPU core per task
#SBATCH --time=4-00:00:00
#SBATCH --output hello.out            
#SBATCH --ntasks 1                    # how many repetition time for srun

# Run python script with a command line argument
# srun python test.py $SLURM_ARRAY_TASK_ID
srun wandb agent daniel-xiao/algorithm_3_robust_mod/9w5qngfw