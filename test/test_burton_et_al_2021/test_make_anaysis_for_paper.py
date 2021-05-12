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
import pymc3 as pm
import arviz as az
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

number_of_cpus = mp.cpu_count()
font_size = 25
cm_to_inches = 0.3937008
class TestInference(unittest.TestCase):

    def xest_make_experimental_data(self):
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','')
        # import spreadsheets as dataframes
        het_hom_df = pd.DataFrame(pd.read_excel(loading_path + "HES5Molnumber_FCS_E10_5_Het_Hom.xlsx",header=None))
        experiment_date = '280317p6'
        cell_intensity_df = pd.DataFrame(pd.read_excel(loading_path + experiment_date + "_VH5_corrected_tissmean_Zpos_correctstartt.xlsx",header=None))

        # convert to numpy arrays for plotting / fitting
        intensities = cell_intensity_df.iloc[2:,2:].astype(float).values.flatten()
        intensities = intensities[~(np.isnan(intensities))]
        het_molecule_numbers = het_hom_df.iloc[1:,0].astype(float).values.flatten()
        het_molecule_numbers = het_molecule_numbers[~(np.isnan(het_molecule_numbers))]
        hom_molecule_numbers = het_hom_df.iloc[1:,1].astype(float).values.flatten()
        hom_molecule_numbers = hom_molecule_numbers[~(np.isnan(hom_molecule_numbers))]
        het_and_hom = np.concatenate((het_molecule_numbers,hom_molecule_numbers))


        # make qqplots and calculate gradients
        # fig, ax = plt.subplots(1,3,figsize=(20,10))
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(het_molecule_numbers,np.linspace(0.0,1.0,101))
        gradient_het = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[0].scatter(x,y)
        # ax[0].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[0].set_ylabel("Het Molecule numbers")
        # ax[0].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_het[0],2))
        # ax[0].text(0.5,71000,gradient_string)
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(hom_molecule_numbers,np.linspace(0.0,1.0,101))
        gradient_hom = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[1].scatter(x,y)
        # ax[1].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[1].set_ylabel("Hom Molecule numbers")
        # ax[1].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_hom[0],2))
        # ax[1].text(0.5,120000,gradient_string)
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(het_and_hom,np.linspace(0.0,1.0,101))
        gradient_het_and_hom = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[2].scatter(x,y)
        # ax[2].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[2].set_ylabel("Het and Hom Molecule numbers")
        # ax[2].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_het_and_hom[0],2))
        # ax[2].text(0.5,120000,gradient_string)
        # plt.tight_layout()
        # plt.savefig(saving_path + 'molecule_qq_plot.png')

        np.save(loading_path + 'selected_data_for_mala/' + experiment_date + "_measurement_variance.npy",np.std((gradient_hom,gradient_het,gradient_het_and_hom)))

        # # make data from each trace and save
        # for cell_index in range(1,cell_intensity_df.shape[1]):
        #     cell_intensity_values = cell_intensity_df.iloc[2:,[0,cell_index]].astype(float).values
        #     # remove NaNs
        #     cell_intensity_values = cell_intensity_values[~np.isnan(cell_intensity_values[:,1])]
        #     cell_intensity_values[:,0] *=60 # turn hours to minutes
        #     cell_intensity_values[:,1] *= gradient_hom # average of hom, het, het and hom?
        #     cell_cluster = int(cell_intensity_df.iloc[0,cell_index])
        #     np.save(loading_path + 'protein_observations_' + experiment_date + '_cell_' + str(cell_index) + '_cluster_' + str(cell_cluster),
        #             cell_intensity_values)

    def xest_detrend_trace(self,data_filename = 'protein_observations_040417_cell_2_cluster_4.npy'):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data_parnian/detrended_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','experimental_data_parnian/detrended_data_images/')
        # loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        # saving_path = os.path.join(os.path.dirname(__file__),'output','detrended_data_images/')

        experimental_data_strings = [i for i in os.listdir(loading_path) if 'npy' in i
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
            detrended_protein, y_gpr, y_std = hes5.detrend_experimental_data(protein,length_scale=500)
            variances[index] = np.var(protein[:,1])
            detrended_variances[index] = np.var(detrended_protein[:,1])
            np.save(loading_path + cell_string[:-4] + "_detrended.npy",detrended_protein)

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
            plt.savefig(saving_path + cell_string[:-4] + "_detrended.png")
            # import pdb; pdb.set_trace()
        measurement_variance = np.sqrt(0.1*np.mean(detrended_variances))
        np.save(loading_path + 'measurement_variance_detrended.npy',measurement_variance)

    def xest_kalman_mala_likelihood_single_parameters(self):
        # load data and true parameter values
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'output','')

        protein_at_observations = np.array([np.load(saving_path + 'protein_observations_90_ps3_ds1.npy')])
        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])
        true_parameter_values = np.load(os.path.join(saving_path,'ps3_parameter_values.npy'))
        # mala_output = np.load(saving_path + 'mala_output_1.npy')
        number_of_samples = 200
        measurement_variance = np.power(true_parameter_values[-1],2)

        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        initial_position = true_parameter_values[[0,1,2,3,4,5,6]]
        initial_position[[4,5]] = np.log(initial_position[[4,5]])
        # proposal_covariance = np.cov(mala_output.T)
        step_size = [55000.0,0.45,0.01,0.013,90.0]
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(np.log(2)/30)],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/90)],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}


        for index in range(5):
            if index == 0: # repression_threshold
                known_parameters = {k:all_parameters[k] for k in ('hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_repression.npy")
                x_values = np.linspace(3300,3600,len(likelihood))
                normal = np.trapz(np.exp(likelihood),x_values)
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$P_0$",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.vlines(true_parameter_values[0],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'repression_likelihood_mala.png')
                plt.clf()

            elif index == 1: # hill_coefficient
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_hill.npy")
                x_values = np.linspace(2.0,6.0,1000)
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$h$",fontsize=font_size)
                ax.vlines(true_parameter_values[1],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.set_ylabel('Probability',fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'hill_likelihood_mala.png')
                plt.clf()
            elif index == 2: # transcription_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_transcription.npy")
                x_values = np.linspace(0.01,60.0,200)
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.vlines(true_parameter_values[4],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlabel("$\\alpha_m$ (1/min)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'transcription_likelihood_mala.png')
                plt.clf()

            elif index == 3: # translation_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_translation.npy")
                x_values = np.linspace(1.1,1.55,len(likelihood))
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$\\alpha_p$ (1/min)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.vlines(true_parameter_values[5],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'translation_likelihood_mala.png')
                plt.clf()
            elif index == 4: # transcription_delay
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_delay.npy")
                x_values = np.linspace(5.0,40.0,36)
                unique_values, unique_indices = np.unique(likelihood,return_index=True)
                unique_indices = np.sort(unique_indices)
                normal = np.trapz(np.exp(likelihood[unique_indices]),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood[unique_indices])/normal,label='Likelihood')
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.vlines(true_parameter_values[6],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.set_xlabel("$\\tau$ (mins)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'delay_likelihood_mala.png')
                plt.clf()

    def xest_relationship_between_steady_state_mean_and_variance(self):
        model_parameters = [10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0]
        mean = hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[0],
                                                  hill_coefficient=model_parameters[1],
                                                  mRNA_degradation_rate=model_parameters[2],
                                                  protein_degradation_rate=model_parameters[3],
                                                  basal_transcription_rate=model_parameters[4],
                                                  translation_rate=model_parameters[5])

        LNA_mRNA_variance = np.power(hes5.calculate_approximate_mRNA_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                           hill_coefficient=model_parameters[1],
                                                                                                           mRNA_degradation_rate=model_parameters[2],
                                                                                                           protein_degradation_rate=model_parameters[3],
                                                                                                           basal_transcription_rate=model_parameters[4],
                                                                                                           translation_rate=model_parameters[5],
                                                                                                           transcription_delay=model_parameters[6]),2)

        LNA_protein_variance = np.power(hes5.calculate_approximate_protein_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                                 hill_coefficient=model_parameters[1],
                                                                                                                 mRNA_degradation_rate=model_parameters[2],
                                                                                                                 protein_degradation_rate=model_parameters[3],
                                                                                                                 basal_transcription_rate=model_parameters[4],
                                                                                                                 translation_rate=model_parameters[5],
                                                                                                                 transcription_delay=model_parameters[6]),2)

        print('mean =',mean)
        print('mRNA_variance/mRNA_mean =',LNA_mRNA_variance/mean[0])
        print('protein_variance/protein_mean =',LNA_protein_variance/mean[1])

    def xest_generate_multiple_protein_observations(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        saving_path = os.path.join(os.path.dirname(__file__), 'data/figure_5','')
        ps_strings = ["ps6","ps9"]
        time = 736
        durations = [i*time for i in range(1,6)]
        frequencies = [5,8,12,15]
        batches = [1,2,3,4,5]
        for batch in batches:
            for ps_string in ps_strings:
                for obs_index, observation_duration in enumerate(durations):
                    for observation_frequency in frequencies:
                        parameters = np.load(loading_path + ps_string + "_parameter_values.npy")
                        no_of_observations = np.int(observation_duration/observation_frequency)

                        true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                                      repression_threshold = parameters[0],
                                                                      hill_coefficient = parameters[1],
                                                                      mRNA_degradation_rate = parameters[2],
                                                                      protein_degradation_rate = parameters[3],
                                                                      basal_transcription_rate = parameters[4],
                                                                      translation_rate = parameters[5],
                                                                      transcription_delay = parameters[6],
                                                                      equilibration_time = 1000)

                        ## the F constant matrix is left out for now
                        protein_at_observations = true_data[:,(0,2)]
                        protein_at_observations[:,1] += np.random.randn(true_data.shape[0])*parameters[-1]
                        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
                        np.save(saving_path + 'protein_observations_' + ps_string + '_{i}_cells_{j}_minutes_{k}.npy'.format(i=obs_index+1,j=observation_frequency,k=batch),
                        protein_at_observations[0::observation_frequency,:])

    def xest_compute_likelihood_at_multiple_parameters(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_at_observations = np.array([np.load(saving_path + 'protein_observations_90_ps3_ds1.npy')])
        true_parameter_values = np.load(saving_path + 'ps3_parameter_values.npy')
        number_of_evaluations = 50
        likelihood_at_multiple_parameters = np.zeros(number_of_evaluations)
        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])
        print(mean_protein)

        repression_threshold = true_parameter_values[0]
        hill_coefficient = true_parameter_values[1]
        mRNA_degradation_rate    = np.log(2)/30
        protein_degradation_rate = np.log(2)/90
        basal_transcription_rate = true_parameter_values[4]
        translation_rate = true_parameter_values[5]
        transcription_delay = true_parameter_values[6]
        measurement_variance = np.power(100,2)
        # import pdb; pdb.set_trace()

        for index, parameter in enumerate(np.linspace(1.1,1.55,number_of_evaluations)):
            likelihood_at_multiple_parameters[index] = -hes_inference.calculate_log_likelihood_at_parameter_point(model_parameters=np.array([repression_threshold,
                                                                                                                                            hill_coefficient,
                                                                                                                                            mRNA_degradation_rate,
                                                                                                                                            protein_degradation_rate,
                                                                                                                                            basal_transcription_rate,
                                                                                                                                            parameter,
                                                                                                                                            transcription_delay]),
                                                                                                                 protein_at_observations=protein_at_observations,
                                                                                                                 measurement_variance = measurement_variance)


        np.save(os.path.join(os.path.dirname(__file__), 'output','likelihood_translation.npy'),likelihood_at_multiple_parameters)

    def xest_multiple_mala_traces_figure_5(self,data_filename = 'protein_observations_ps6_fig5_5.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_fig')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
        measurement_variance = np.power(true_parameter_values[-1],2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(true_parameter_values[2])],
                          'protein_degradation_rate' : [3,np.log(true_parameter_values[3])],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_multiple_mala_traces_figure_5b(self,data_filename = 'protein_observations_ps6_fig5_3_cells_15_minutes.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path+'/figure_5',data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('ps') + 3
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))

        measurement_variance = np.power(true_parameter_values[-1],2)
        # define known parameters
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(true_parameter_values[2])],
                          'protein_degradation_rate' : [3,np.log(true_parameter_values[3])],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_mala_experimental_data(self,data_filename = 'protein_observations_040417_cell_52_cluster_4_detrended.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        experiment_date = data_filename[data_filename.find('ns_')+3:data_filename.find('_cel')]
        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,np.log(np.log(2)/30)],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/90)],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_mala_multiple_experimental_traces(self,experiment_date='280317p1',cluster='1'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        data_filename = experiment_date + '_cluster_' + cluster + '.npy'
        protein_at_observations = np.array([np.load(os.path.join(saving_path,i)) for i in os.listdir(saving_path) if 'detrended' in i
                                                                               if 'cluster_'+cluster in i
                                                                               if experiment_date in i])

        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,np.log(np.log(2)/30)],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/90)],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_mala_analysis(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output/figure_5','')
        chain_path_strings = [i for i in os.listdir(loading_path) if 'ps6_1_cells_8_minutes_3' in i
                                                                  if '.npy' in i]
        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            mala = mala[[0,1,2,4,5,6,7],:,:]
            # mala[:,:,[2,3]] = np.exp(mala[:,:,[2,3]])
            chains = az.convert_to_dataset(mala)
            print('\n' + chain_path_string + '\n')
            print('\nrhat:\n',az.rhat(chains))
            print('\ness:\n',az.ess(chains))
            az.plot_trace(chains); plt.savefig(loading_path + 'traceplot_' + chain_path_string[:-4] + '.png'); plt.close()
            az.plot_posterior(chains); plt.savefig(loading_path + 'posterior_' + chain_path_string[:-4] + '.png'); plt.close()
            az.plot_pair(chains,kind='kde'); plt.savefig(loading_path + 'pairplot_' + chain_path_string[:-4] + '.png'); plt.close()
            # np.save(loading_path + chain_path_string,mala)

    def xest_identify_oscillatory_parameters(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(loading_path + '.npy')
        prior_samples = np.load(loading_path + '_parameters.npy')

        ps_string = "ps10"
        coherence_bands = [[0.01,0.03],[0.2,0.25]]
        observation_duration  = 600
        observation_frequency = 5
        no_of_observations    = np.int(observation_duration/observation_frequency)
        protein_at_observations = np.zeros((2,observation_duration//5,2))
        for i, coherence_band in enumerate(coherence_bands):
            accepted_indices = np.where(np.logical_and(model_results[:,0]>30000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]>0.05, #standard deviation
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                        np.logical_and(model_results[:,3]>coherence_band[0], #coherence
                                        np.logical_and(model_results[:,3]<coherence_band[1], #coherence
                                                       prior_samples[:,4]<6))))))) # hill coefficient

            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]

            this_parameter = my_posterior_samples[0]
            this_results = my_model_results[0]

            print('this basal transcription rate')
            print(this_parameter[0])
            print('this translation rate')
            print(this_parameter[1])
            print('this repression threshold')
            print(this_parameter[2])
            print('this transcription_delay')
            print(this_parameter[3])
            print('this hill coefficient')
            print(this_parameter[4])
            print('coherence')
            print(this_results[3])

            saving_path = os.path.join(os.path.dirname(__file__), 'data','')

            measurement_variance = 1000

            parameters = np.array([this_parameter[2],
                                   this_parameter[4],
                                   np.log(2)/30,
                                   np.log(2)/90,
                                   this_parameter[0],
                                   this_parameter[1],
                                   this_parameter[3],
                                   this_results[3],
                                   measurement_variance])



            # np.save(saving_path + ps_string + "_parameter_values.npy",parameters)


            # parameters = np.load(loading_path + ps_string + "_parameter_values.npy")

            true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                          repression_threshold = parameters[0],
                                                          hill_coefficient = parameters[1],
                                                          mRNA_degradation_rate = parameters[2],
                                                            protein_degradation_rate = parameters[3],
                                                          basal_transcription_rate = parameters[4],
                                                          translation_rate = parameters[5],
                                                          transcription_delay = parameters[6],
                                                          equilibration_time = 1000)
            # np.save(saving_path + 'true_data_' + ps_string + '.npy',
            #         true_data)

            ## the F constant matrix is left out for now
            protein_at_observations[i] = true_data[::5,(0,2)]
            protein_at_observations[i,:,1] += np.random.randn(true_data.shape[0]//5)*parameters[-1]
            protein_at_observations[i,:,1] = np.maximum(protein_at_observations[i,:,1],0)
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds1.npy',
            #             protein_at_observations[0:900:10,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds2.npy',
            #             protein_at_observations[0:900:5,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds3.npy',
            #             protein_at_observations[0:1800:10,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds4.npy',
            #             protein_at_observations[0:1800:5,:])

        my_figure = plt.figure(figsize=(9.83,5.54))
        plt.scatter(np.arange(0,observation_duration,5),protein_at_observations[0,:,1],marker='o',s=12,c='#F18D9E',label='Low Coherence')
        plt.scatter(np.arange(0,observation_duration,5),protein_at_observations[1,:,1],marker='o',s=12,c='#8d9ef1',label='High Coherence')
        plt.plot(np.arange(0,observation_duration,5),protein_at_observations[0,:,1],'--',c='#F18D9E',alpha=0.5)
        plt.plot(np.arange(0,observation_duration,5),protein_at_observations[1,:,1],'--',c='#8d9ef1',alpha=0.5)
        plt.xlabel('Time (mins)',fontsize=font_size)
        plt.ylabel('Protein Molecules',fontsize=font_size)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        my_figure.savefig(saving_path + 'coherence_comparison.pdf')

def run_mala_for_dataset(data_filename,
                         protein_at_observations,
                         measurement_variance,
                         number_of_parameters,
                         known_parameters,
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
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
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
        initial_guess = np.array([mean_protein,5.0,np.log(2)/30,np.log(2)/90,1.0,1.0,10.0])
        optimiser = minimize(hes_inference.calculate_log_likelihood_at_parameter_point,
                             initial_guess,
                             args=(protein_at_observations,measurement_variance),
                             bounds=np.array([(0.3*mean_protein,1.3*mean_protein),
                                              (2.0,5.0),
                                              (np.log(2)/30,np.log(2)/30),
                                              (np.log(2)/90,np.log(2)/90),
                                              (0.01,120.0),
                                              (0.01,40.0),
                                              (1.0,40.0)]),
                             method='Powell')

        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,6)] = optimiser.x[[0,1,6]]
        initial_states[:,(4,5)] = np.log(optimiser.x[[4,5]])

        print("Warming up...")
        initial_burnin_number_of_samples = np.int(0.3*number_of_samples)
        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                initial_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                np.power(np.diag([2*mean_protein,4,9,8,39]),2),# initial variances are width of prior squared
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
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                second_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
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
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
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
