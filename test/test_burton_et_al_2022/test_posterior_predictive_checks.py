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

    def xest_summary_statistics_for_posterior(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i and 'final' in i and 'png' not in i]

        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            mala[:,:,[2,3,4]] = np.exp(mala[:,:,[2,3,4]])

            # need samples in the order (alpha_m, alpha_p, P_0, tau, h, nu_m, mu_p)
            mala = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])

            correct_order_samples = np.zeros((mala.shape[0],7))
            permute = [3,4,0,5,1,2]
            for index, col in enumerate(permute):
                correct_order_samples[:,index] = mala[:,col]

            correct_order_samples[:,6] = np.log(2)/12
            # take 200 random posterior samples
            n = 12
            indices = np.random.choice(mala.shape[0],n,replace=False)
            # obtain summary statistics for these samples
            import pdb; pdb.set_trace()
            summary_statistics = hes5.calculate_summary_statistics_at_parameters(correct_order_samples[indices,:],number_of_cpus=2)
                # number_of_traces_per_sample = 1,
                # number_of_cpus = 6,#number_of_available_cores,
                # model = 'langevin',
                # timestep = 1.0,
                # simulation_duration = 1000,
                # power_spectrum_smoothing_window = 0.1
            import pdb; pdb.set_trace()

    def xest_plot_posterior_predictions(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        data_loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i and 'final' in i and 'png' not in i]

        measurement_variance = np.load(data_loading_path + 'detrended_data/28hpf_141117_test_measurement_variance_detrended.npy')

        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            mala[:,:,[2,3,4]] = np.exp(mala[:,:,[2,3,4]])
            data = np.load(data_loading_path + chain_path_string[27:])
            data[:,0] -= data[0,0]


            # need samples in the order (alpha_m, alpha_p, P_0, tau, h, nu_m, mu_p)
            mala = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])

            correct_order_samples = np.zeros((mala.shape[0],7))
            permute = [3,4,0,5,1,2]
            for index, col in enumerate(permute):
                correct_order_samples[:,index] = mala[:,col]

            correct_order_samples[:,6] = np.log(2)/12
            # take n random posterior samples
            n = 2000
            indices = np.random.choice(mala.shape[0],n,replace=False)
            
            random_samples = correct_order_samples[indices,:]
            
            fig, ax = plt.subplots(figsize=(10,10))
            
            
            for index, sample in enumerate(random_samples):
                trace = hes5.generate_langevin_trajectory(duration = data[-1,0]+1,
                            repression_threshold = sample[2],
                            hill_coefficient = sample[4],
                            mRNA_degradation_rate = sample[5],
                            protein_degradation_rate = np.log(2)/12,
                            basal_transcription_rate = sample[0],
                            translation_rate = sample[1],
                            transcription_delay = sample[3],
                            initial_mRNA = 0,
                            initial_protein = 0,
                            equilibration_time = 500.0,
                            extrinsic_noise_rate = 0.0,
                            transcription_noise_amplification = 1.0,
                            timestep = 1.0
                        )
                trace[:,2] = np.maximum(0,trace[:,2] + measurement_variance*np.random.randn(trace.shape[0]))
                if index == 0:
                    ax.plot(data[:,0],trace[::np.int(data[1,0]),2],color='#F18D9E',alpha=0.05,label="Posterior prediction")
                else:
                    ax.plot(data[:,0],trace[::np.int(data[1,0]),2],color='#F18D9E',alpha=0.05)
            ax.plot(data[:,0],data[:,1],
                    color='black',label='Data')
            ax.set_xlabel('Time (mins)')
            ax.set_ylabel('Protein Molecules')
            leg = plt.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
            # print(chain_path_string)
            plt.tight_layout()
            # plt.show()
            plt.savefig(loading_path + chain_path_string[27:-4] + '_ppc.png')



    def test_plot_posterior_means(self,experiment_name='34hpf'):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        mbs_chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i and experiment_name in i and 'MBS' in i and 'png' not in i]
        ctrl_chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i and experiment_name in i and 'CTRL' in i and 'png' not in i]
        alpha = 0.35
        fig, ax = plt.subplots(1,6,figsize=(1.4*17,1.4*3.02))
        parameters = [3,4,2,0,5,1]
        parameter_names = np.array(["$\\alpha_m$ (1/min)",
                                    "$\\alpha_p$ (1/min)",
                                    "mRNA half-life (min)",
                                    "$P_0$",
                                    "$\\tau$ (mins)",
                                    "$h$",])
        mbs_mala_means = np.zeros((len(mbs_chain_path_strings),len(parameter_names)))
        ctrl_mala_means = np.zeros((len(ctrl_chain_path_strings),len(parameter_names)))
        for chain_path_index, chain_path_string in enumerate(mbs_chain_path_strings):
            mala = np.load(loading_path + chain_path_string)
            mala[:,:,[2,3,4]] = np.exp(mala[:,:,[2,3,4]])
            mala[:,:,2] = np.log(2) / mala[:,:,2]
            # plot_histograms(mala,loading_path + chain_path_string[:-4],save=False)

            mala = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
            mbs_mala_means[chain_path_index,:] = np.mean(mala,axis=0)

        for chain_path_index, chain_path_string in enumerate(ctrl_chain_path_strings):
            mala = np.load(loading_path + chain_path_string)
            mala[:,:,[2,3,4]] = np.exp(mala[:,:,[2,3,4]])
            mala[:,:,2] = np.log(2) / mala[:,:,2]
            # plot_histograms(mala,loading_path + chain_path_string[:-4],save=False)

            mala = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
            ctrl_mala_means[chain_path_index,:] = np.mean(mala,axis=0)

        for index, parameter in enumerate(parameters):
            ax[index].boxplot([ctrl_mala_means[:,parameters[index]],
                               mbs_mala_means[:,parameters[index]]])
            ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
            ax[index].set_xticks(np.arange(1,3))
            ax[index].set_xticklabels(['CTRL','MBS'])
        ax[0].set_ylabel("Posterior Means",fontsize=font_size*1.2)

        def format_tick_labels(x, pos):
                return '{:.0e}'.format(x)
        from matplotlib.ticker import FuncFormatter
        ax[2].yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

        # ax[2].set_yticklabels()
        plt.suptitle(experiment_name)
        plt.tight_layout()
        plt.savefig(loading_path + "posterior_means_" + experiment_name + ".png")