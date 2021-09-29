#!python
import argparse
import os
import subprocess
import time
import fnmatch
import numpy as np

def main():
    loading_path = os.path.join(os.path.dirname(__file__),'../data/experimental_data/selected_data_for_mala/')
    saving_path = os.path.join(os.path.dirname(__file__),'joblist')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for experiment_date in ['040417','280317p1','280317p6']:
        for cluster in ['1','2','3','4']:
            with open(os.path.join(saving_path,'parallel_mala_output_protein_observations_'+experiment_date+'_cluster_'+cluster+'.txt'),'w') as f:
                f.write('python -c \"import test_inference; class_instance = test_inference.TestInference(); class_instance.test_mala_multiple_experimental_traces(experiment_date=\'{i}\',cluster=\'{j}\')''\"'.format(i=experiment_date,j=cluster))
    return saving_path

if __name__ == "__main__":
    main()
