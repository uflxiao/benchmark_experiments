#!/bin/bash

for i in $seq(1 1000); do
    sbatch start.slurm   &
done
