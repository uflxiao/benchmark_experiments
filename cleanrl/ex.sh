#!/bin/bash

for i in $(seq 1 100); do
    sbatch start.slurm   &
done
