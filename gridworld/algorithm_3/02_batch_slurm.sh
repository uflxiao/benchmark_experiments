#!/bin/bash

for i in $(seq 1 20./0); do
    sbatch slurm/02_run.slurm   &
done

