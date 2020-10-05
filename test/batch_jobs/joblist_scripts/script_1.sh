#!/bin/bash --login
#$ -cwd # Job runs in current directory (where you run qsub)
#$ -V # Job inherits environment (settings from loaded modules etc)
#$ -pe smp.pe 24 # request intel nodes and find 12 core slot
#$ -M joshua.burton@manchester.ac.uk
#$ -m e # send email to above address when job ends
export OMP_NUM_THREADS=1 # limits jobs to requested resources
cd ~/scratch/hesdynamics/test/
nosetests -s --nologcapture test_inference.py protein_observations_90_ps3_ds1.npy

wait
