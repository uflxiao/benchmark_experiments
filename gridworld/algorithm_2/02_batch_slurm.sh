#!/bin/bash

for i in $(seq 1 5); do
    sbatch slurm/02_run.slurm   &
done

