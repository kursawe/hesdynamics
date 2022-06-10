import unittest
import os.path
import sys
import argparse
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
# mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
# mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
font = {'size'   : 20}
plt.rc('font', **font)
import numpy as np
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from jitcdde import jitcdde,y,t
import time
from scipy.spatial.distance import euclidean
from scipy import stats
# import pymc3 as pm
import arviz as az
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'../../','src'))
import hes5
import hes_inference

number_of_cpus = mp.cpu_count()
font_size = 25
cm_to_inches = 0.3937008
class TestInference(unittest.TestCase):
    
    def test_mala_experimental_data(self,data_filename = 'protein_observations_28hpf_141117_test_cell_1.npy'):
        # load data and true parameter values
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        experiment_name = data_filename[21:data_filename.find('cell')-1]
        protein_at_observations = np.array([np.load(os.path.join(loading_path,data_filename))])
        measurement_variance = np.power(np.round(np.load(loading_path + experiment_name + "_measurement_variance_detrended.npy"),4),2)


        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,None],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/11)],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}


        known_parameters = {k:all_parameters[k] for k in ['protein_degradation_rate'] if k in all_parameters}
        # known_parameters = all_parameters['protein_degradation_rate']

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        mean_protein = mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])

        prior_bounds = np.array([50,2*mean_protein-50,
                                 2,6-2,
                                 np.log(2)/11,np.log(2)/1-np.log(2)/11,
                                 np.log(2)/12,np.log(2)/12,
                                 1.0,120.0-1.0,
                                 0.1,40.0-0.1,
                                 1,11-1]).reshape(-1,2)

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             prior_bounds,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_mala_analysis(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i and 'final' in i]
        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            # mala = mala[[0,1,2,4,5,6,7],:,:]
            mala[:,:,[2,3,4]] = np.exp(mala[:,:,[2,3,4]])
            mala[:,:,[2,3]] = np.log(2) / mala[:,:,[2,3]]
            chains = az.convert_to_dataset(mala)
            # print('\n' + chain_path_string + '\n')
            # print('\nrhat:\n',az.rhat(chains))
            # print('\ness:\n',az.ess(chains))
            az.plot_trace(chains,compact=False); plt.savefig(loading_path + 'traceplot_' + chain_path_string[:-4] + '.png'); plt.close()
            az.plot_posterior(chains); plt.savefig(loading_path + 'posterior_' + chain_path_string[:-4] + '.png'); plt.close()
            plot_histograms(mala,chain_path_string)
            # az.plot_pair(chains,kind='kde'); plt.savefig(loading_path + 'pairplot_' + chain_path_string[:-4] + '.png'); plt.close()
            # np.save(loading_path + chain_path_string,mala)

    def xest_make_experimental_data(self):
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','')
        # import spreadsheets as dataframes
        FCS_df = pd.DataFrame(pd.read_excel(loading_path + "FCS_molec_different_development_stages_Hindbrain.xlsx",header=0))
        experiment_date = '28hpf_141117_test' # 28hpf_141117, 30hpf_170517, 34hpf_160617
        cell_intensity_df = pd.DataFrame(pd.read_excel(loading_path + "CTRL_venus_h2b_ratio_" + experiment_date + ".xls",header=None))

        # convert to numpy arrays for plotting / fitting
        intensities = cell_intensity_df.iloc[:,1:].astype(float).values.flatten()
        intensities = intensities[~(np.isnan(intensities))]
        

        hpf_keys = ['19hpf','29hpf','33hpf','35hpf','48hpf']
        wanted_keys = ['29hpf','33hpf','35hpf']
        FCS_at_hpf = {}
        for key in hpf_keys:
            FCS_at_hpf[key] = FCS_df[key].astype(float).values.flatten()
            FCS_at_hpf[key] = FCS_at_hpf[key][~(np.isnan(FCS_at_hpf[key]))]
        FCS_at_hpf['all_hpf'] = np.hstack(FCS_at_hpf.values())
        FCS_at_hpf['correct_hpf'] = np.hstack(list( map(FCS_at_hpf.get, wanted_keys) ))

        # make qqplots and calculate gradients
        gradients = np.zeros(len(FCS_at_hpf.keys()))
        fig, ax = plt.subplots(1,len(FCS_at_hpf.keys()),figsize=(25*1.5,5*1.5))
        for index, key in enumerate(FCS_at_hpf.keys()):
            x = np.quantile(intensities,np.linspace(0.0,1.0,101))
            y = np.quantile(FCS_at_hpf[key],np.linspace(0.0,1.0,101))
            gradient = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
            gradients[index] = gradient[0]
            ax[index].scatter(x,y)
            ax[index].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
            ax[index].set_ylabel(key + 'FCS')
            ax[index].set_xlabel("Cell intensities")
            gradient_string = "Gradient is " + str(np.round(gradients[index],4))
            ax[index].text(1,1,gradient_string)
        plt.tight_layout()
        plt.savefig(loading_path + 'molecule_qq_plot_test.png')

        # make data from each trace and save
        for cell_index in range(1,cell_intensity_df.shape[1]):
            cell_intensity_values = cell_intensity_df.iloc[2:,[0,cell_index]].astype(float).values
            # remove NaNs
            cell_intensity_values = cell_intensity_values[~np.isnan(cell_intensity_values[:,1])]
            cell_intensity_values[:,0] *= 60 # turn hours to minutes
            cell_intensity_values[:,1] *= gradients[-1] # average of hom, het, het and hom?
            # cell_cluster = int(cell_intensity_df.iloc[0,cell_index])
            np.save(loading_path + 'protein_observations_' + experiment_date + '_cell_' + str(cell_index),
                    cell_intensity_values)


    def xest_detrend_trace(self,experiment_name = '28hpf_141117'):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        data_saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/detrended_data/')
        image_saving_path = os.path.join(os.path.dirname(__file__),'output','experimental_data/detrended_data_images/')
        # loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        # saving_path = os.path.join(os.path.dirname(__file__),'output','detrended_data_images/')

        experimental_data_strings = [i for i in os.listdir(loading_path) if 'npy' in i
                                     and experiment_name in i
                                     and not 'detrended' in i]
        variances = np.zeros(len(experimental_data_strings))
        detrended_variances = np.zeros(len(experimental_data_strings))
        # import pdb; pdb.set_trace()

        # make data from each trace and save
        for index, cell_string in enumerate(experimental_data_strings):
            protein = np.load(loading_path + cell_string)
            protein[:,0] -= protein[0,0] # start time = 0
            mean_protein = np.mean(protein[:,1])
            protein_around_mean = protein[:,1] - mean_protein
            times = protein[:,0]
            detrended_protein, y_gpr, y_std = hes5.detrend_experimental_data(protein,length_scale=270)
            variances[index] = np.var(protein[:,1])
            detrended_variances[index] = np.var(detrended_protein[:,1])
            np.save(data_saving_path + cell_string[:-4] + "_detrended.npy",detrended_protein)

            # plot
            lw = 2
            fig, ax = plt.subplots(1,1,figsize=(1.4*13, 1.4*5.19))
            # plot data without trend and mean
            ax.plot(times,protein[:,1], c='k', label='data')
            X_plot = np.linspace(0, np.int(times[-1]), np.int(times[-1])+1)[:, None]
            ax.plot(times, detrended_protein[:,1], color='#20948B', lw=lw,label='detrended data')
            ax.plot(X_plot, mean_protein + y_gpr, '#F18D9E', lw=lw,label='trend')
            ax.set_xlabel('Time (mins)',fontsize=font_size*1.2)
            ax.set_ylabel('Protein',fontsize=font_size*1.2)
            ax.legend()
            # plot original data and detrended data
            plt.tight_layout()
            plt.savefig(image_saving_path + cell_string[:-4] + "_detrended.png")
        measurement_variance = np.sqrt(0.1*np.mean(detrended_variances))
        np.save(data_saving_path + experiment_name + '_measurement_variance_detrended.npy',measurement_variance)

def run_mala_for_dataset(data_filename,
                        protein_at_observations,
                        measurement_variance,
                        number_of_parameters,
                        known_parameters,
                        prior_bounds,
                        step_size = 1,
                        number_of_chains = 8,
                        number_of_samples = 80000):
    """
    A function which gives a (hopefully) decent MALA output for a given dataset with known or
    unknown parameters. If a previous output already exists, this will be used to create a
    proposal covariance matrix, otherwise one will be constructed with a two step warm-up
    process.
    """
    # make sure all data starts from time "zero"
    for i in range(protein_at_observations.shape[0]):
        protein_at_observations[i,:,0] -= protein_at_observations[i,0,0]

    mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])

    # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
    if os.path.exists(os.path.join(
                      os.path.dirname(__file__),
                      'output','final_parallel_mala_output_' + data_filename)):
        print("Posterior samples already exist, sampling directly without warm up...")

        mala_output = np.load(os.path.join(os.path.dirname(__file__),
                              'output','final_parallel_mala_output_' + data_filename))
        previous_number_of_samples = mala_output.shape[1]
        previous_number_of_chains = mala_output.shape[0]

        samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,3] = np.array([np.log(np.log(2)/12)])
        initial_states[:,(0,1,2,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                prior_bounds,
                                                                proposal_covariance,
                                                                1,
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

    else:
        # warm up chain
        print("New data set, initial warm up with " + str(np.int(number_of_samples*0.3)) + " samples...")
        # Initialise by minimising the function using a constrained optimization method witout gradients
        print("Optimizing using Powell's method for initial guess...")
        from scipy.optimize import minimize
        initial_guess = np.array([mean_protein,4.5,np.log(2)/5,np.log(2)/12,1.0,1.0,5.0])
        optimiser = minimize(hes_inference.calculate_log_likelihood_at_parameter_point,
                             initial_guess,
                             args=(protein_at_observations,measurement_variance),
                             bounds=np.array([(0.3*mean_protein,1.3*mean_protein),
                                              (2.0,5.0),
                                              (np.log(2)/11,np.log(2)/1),
                                              (np.log(2)/12,np.log(2)/12),
                                              (1.0,120.0),
                                              (1.0,40.0),
                                              (1.0,11.0)]),
                             method='Powell')

        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(0,1,6)] = optimiser.x[[0,1,6]]
        initial_states[:,(2,3,4,5)] = np.log(optimiser.x[[2,3,4,5]])

        print("Warming up with " + str(int(number_of_samples*0.3)) + " samples...")
        initial_burnin_number_of_samples = np.int(0.3*number_of_samples)
        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                initial_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                prior_bounds,
                                                                np.power(np.diag([2*mean_protein,4,8,1.0,1.0,10]),2),# initial variances are width of prior squared, only include unknown parameters
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,initial_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','first_parallel_mala_output_' + data_filename),
        array_of_chains)

        print("Second warm up with " + str(int(number_of_samples*0.7)) + " samples...")
        second_burnin_number_of_samples = np.int(0.7*number_of_samples)

        samples_with_burn_in = array_of_chains[:,int(initial_burnin_number_of_samples/2):,:].reshape(int(initial_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,3] = np.array([np.log(np.log(2)/12)])
        initial_states[:,(0,1,2,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                second_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                prior_bounds,
                                                                proposal_covariance,
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,second_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','second_parallel_mala_output_' + data_filename),
        array_of_chains)

        # sample directly
        print("Now sampling directly...")
        samples_with_burn_in = array_of_chains[:,int(second_burnin_number_of_samples/2):,:].reshape(int(second_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,3] = np.array([np.log(np.log(2)/12)])
        initial_states[:,(0,1,2,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                prior_bounds,
                                                                proposal_covariance,
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

def plot_histograms(output, path):
    output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])
    alpha = 0.35
    fig, ax = plt.subplots(1,6,figsize=(1.4*17,1.4*3.02))
    parameters = [3,4,2,0,5,1]
    parameter_names = np.array(["$\\alpha_m$ (1/min)",
                                "$\\alpha_p$ (1/min)",
                                "mRNA half-life (min)",
                                "$P_0$",
                                "$\\tau$ (mins)",
                                "$h$",])

    for index, parameter in enumerate(parameters):
        if parameter in [0,1,2,3]:
        #     hist, bins = np.histogram(np.exp(output[:,parameter]),density=True,bins=20)
        #     logbins = np.geomspace(bins[0],bins[-1],20)
        #     # import pdb; pdb.set_trace()
        #     ax[index].hist(output[:,parameter],density=True,bins=logbins,alpha=alpha,color='#20948B')
        #     ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
        #     ax[index].set_xlim(0,6)
        #     ax[index].set_ylim(0,1.0)
        # if parameter == 3:
            ax[index].hist(output[:,parameter],density=True,bins=np.geomspace(np.min(output[:,parameter]),np.max(output[:,parameter]),20),alpha=alpha,color='#20948B')
            ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
        else:
            ax[index].hist(output[:,parameter],density=True,bins=20,alpha=alpha,color='#20948B')
            ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
    # ax[1].set_ylim(0,5)
    ax[0].set_ylabel("Probability",fontsize=font_size*1.2)

    def format_tick_labels(x, pos):
            return '{:.0e}'.format(x)
    from matplotlib.ticker import FuncFormatter
    ax[2].yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

    # ax[2].set_yticklabels()
    plt.tight_layout()
    plt.savefig(path + "_posteriors.png")