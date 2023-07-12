#!/bin/bash

for i in $(seq 1 500); do
    sbatch slurm/03_run.slurm   &
done
