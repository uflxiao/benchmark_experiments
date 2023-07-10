#!/bin/bash

for i in $(seq 1 100); do
    sbatch slurm/01_run.slurm   &
done

