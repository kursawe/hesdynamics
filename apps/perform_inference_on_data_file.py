import argparse
import os.path
import numpy as np
import multiprocessing.pool as mp_pool
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

def run_inference_on_file(dataset_file_name):
    '''Run inference on data in dataset_file_name. Will save output based on the filename.

    If filename is located in path/folder_name/ then this function will write files into
    path/output/
    
    Assumes that results from a previous random walk are saved in the same folder as the dataset,
    with the filename 'full_random_walk_180_ps2_ds1.npy'
    
    Parameters:
    -----------
    
    dataset_file_name : string
        path to the dataset containing the protein observations. Needs to be loadable by numpy and contain
        a two-dimensional array where the first column is time and the second column contains protein observations.
    '''    

    parent_folder = os.path.dirname(dataset_file_name)
    filename_with_extension = os.path.basename(dataset_file_name)
    filename_without_extension,_ = os.path.splitext(filename_with_extension)
    saving_base_name = filename_without_extension.replace("protein_observations","parallel_random_walk")
    saving_folder = os.path.join(parent_folder,'..','output')

    protein_observations    = np.load(dataset_file_name)
    previous_run            = np.load(os.path.join(parent_folder,'full_random_walk_180_ps2_ds1.npy'))

    # define parameters for uniform prior distributions
    hyper_parameters = np.array([100,19900,2,4,0,1,0,1,np.log10(0.1),np.log10(60)+1,np.log10(0.1),np.log10(40)+1,5,35]) # uniform
    measurement_variance = 10000.0

    # draw 8 random initial states for the parallel random walk
    from scipy.stats import uniform
    initial_states          = np.zeros((8,7))
    initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
    for initial_state_index in range(initial_states.shape[0]):
        initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([100,2,np.log10(0.1),np.log10(0.1),5]),
                    np.array([20100,4,np.log10(60)+1,np.log10(40)+1,35]))

    # initial covariance based on prior assumptions about the data
    initial_covariance = 0.04*np.diag(np.array([np.var(previous_run[50000:,0]),np.var(previous_run[50000:,1]),
                                                np.var(previous_run[50000:,2]),np.var(previous_run[50000:,3]),
                                                np.var(previous_run[50000:,4])]))
#     number_of_iterations = 350000
    number_of_iterations = 10

    pool_of_processes = mp_pool.ThreadPool(processes = 8)
    process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                      args=(number_of_iterations,protein_observations,hyper_parameters,measurement_variance,0.6,initial_covariance,initial_state),
                                                      kwds=dict(adaptive='true'))
                        for initial_state in initial_states ]
    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()
    list_of_random_walks      = []
    list_of_acceptance_rates  = []
    list_of_acceptance_tuners = []
    chain_counter = 0
    for process_result in process_results:
        this_random_walk, this_acceptance_rate, this_acceptance_tuner = process_result.get()
        print('successful get ', chain_counter)
        list_of_random_walks.append(this_random_walk)
        list_of_acceptance_rates.append(this_acceptance_rate)
        list_of_acceptance_tuners.append(this_acceptance_tuner)
        chain_counter += 1
    pool_of_processes.join()
    print(list_of_acceptance_rates)
    print(list_of_acceptance_tuners)

    for chain_index in range(len(initial_states)):
        np.save(os.path.join(saving_folder, saving_base_name + '_{chain_index}.npy'.format(chain_index = chain_index)),list_of_random_walks[chain_index])


if __name__ == "__main__":
    function_docstring = run_inference_on_file.__doc__
    programtext = "This script performs Bayesian inference on a dataset, using this function:\n\n" + function_docstring
    parser = argparse.ArgumentParser(description = programtext, formatter_class=argparse.RawTextHelpFormatter)

    # joblist argument
    parser.add_argument('-i','--input_dataset', help = 'The path to the file containing the data')

    args = parser.parse_args()

    if args.input_dataset is None:
        print ' *** error: you need to specify a dataset'
        exit()

    dataset_file_name  = os.path.realpath(args.input_dataset)
    
    run_inference_on_file(dataset_file_name)