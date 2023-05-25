#!/bin/bash

for i in $(seq 1 300); do
    sbatch start.slurm   &
done
