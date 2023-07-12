#!/bin/bash

for i in $(seq 1 1000); do
    sbatch slurm/02_run.slurm   &
done

