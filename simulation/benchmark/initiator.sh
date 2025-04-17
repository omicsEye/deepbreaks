#!/bin/bash
for i in {1..10}; do
        echo "Running simulation with $i"
        sbatch --job-name=deepbreaks_sim_$i --output=report/deepbreaks_sim_$i.out --error=report/deepbreaks_sim_$i.err job.sh $i
done
