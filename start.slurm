#!/bin/bash

#SBATCH --job-name=dqn
#SBATCH --exclude=adriatic[01-06],cheetah01,cheetah02,cheetah03,jaguar01,jaguar02,jaguar04,lotus,ristretto[01-04],titanx[01-03],titanx[04-06],lynx[01-02]
#SBATCH --partition=nolim,main,gnolim,gpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --output dqn.out
#SBATCH --ntasks 1

# Run python script with a command line argument
srun python dqn.py