#!/bin/bash

for i in $(seq 1 500); do
    sbatch start.slurm   &
done
