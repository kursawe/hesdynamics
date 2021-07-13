#!python
import argparse
import os
import subprocess
import time
import fnmatch
import numpy as np

def main():
    loading_path = os.path.join(os.path.dirname(__file__),'../data')
    # protein_datasets = np.array([file for file in os.listdir(loading_path) if 'minutes' in file])
    # modified for review to run any failed tests
    protein_datasets = np.array(["protein_observations_ps6_5_cells_15_minutes_2.npy",
                                 "protein_observations_ps6_5_cells_15_minutes_3.npy",
                                 "protein_observations_ps9_1_cells_8_minutes_5.npy"])

    saving_path = os.path.join(os.path.dirname(__file__),'joblist')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for data in protein_datasets:
        with open(os.path.join(saving_path,data.replace('.npy','') + '.txt'),'w') as f:
            f.write('python -c \"import test_inference; class_instance = test_inference.TestInference(); class_instance.test_multiple_mala_traces_figure_5b(\'{i}\')''\"'.format(i=data))
    return saving_path

if __name__ == "__main__":
    main()
