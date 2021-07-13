#!/bin/bash --login
#$ -cwd # Job runs in current directory (where you run qsub)
#$ -V # Job inherits environment (settings from loaded modules etc)
#$ -pe smp.pe 8 # request intel nodes and find 8 core slot
#$ -M joshua.burton@manchester.ac.uk
#$ -m e # send email to above address when job ends
export OMP_NUM_THREADS=1 # limits jobs to requested resources
cd ~/scratch/hesdynamics/test/
python -c "import test_inference; class_instance = test_inference.TestInference(); class_instance.test_multiple_mala_traces_figure_5_coherence('protein_observations_coherence_23.npy')"

wait
