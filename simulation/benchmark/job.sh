#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --time=10:30:00


# Parameters
python sim.py ${1}