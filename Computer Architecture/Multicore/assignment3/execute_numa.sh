#!/bin/bash
#SBATCH --chdir /scratch/wenli/A3
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 36
#SBATCH --mem 10G
#SBATCH --account cs307
#SBATCH --reservation CS307-WEEKEND3 

echo STARTING AT `date`
numactl --interleave=all ./numa
numactl --cpunodebind=0 --membind=1 ./numa
numactl --localalloc ./numa
echo FINISHED at `date`
