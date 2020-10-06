#!python
import argparse
import os
import subprocess
import time
import fnmatch
import numpy as np

def main():
    loading_path = os.path.join(os.path.dirname(__file__),'../data')
    protein_datasets = np.array([file for file in os.listdir(loading_path) if fnmatch.fnmatch(file,'protein_observations*.npy')])

    saving_path = os.path.join(os.path.dirname(__file__),'joblist')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for data in protein_datasets:
        with open(os.path.join(saving_path,data.replace('.npy','') + '.txt'),'w') as f:
            f.write('python -c \"import test_inference; class_instance = test_inference.TestInference(); class_instance.test_multiple_mala_traces_in_parallel(\'{i}\')''\"'.format(i=data))
    return saving_path

if __name__ == "__main__":
    main()
