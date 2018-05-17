import unittest
import os.path
import sys
import matplotlib as mpl
import matplotlib.gridspec 
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import pandas as pd
import seaborn as sns

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestMakePaperAnalysis(unittest.TestCase):
                                 
    def xest_make_abc_samples(self):
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (1,40),
                        'repression_threshold' : (0,120000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_extended',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_extended_abc_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))

    def xest_make_agnostic_abc_samples(self):
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (1,40),
                        'repression_threshold' : (0,120000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        'noise_strength' : (0,100)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_agnostic',
                                                                prior_bounds = prior_bounds,
                                                                model = 'agnostic',
                                                                logarithmic = True)
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 6))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_agnostic_abc_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
        
    def xest_plot_posterior_distributions(self):
        
        option = 'deterministic'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000, #cell_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                         np.logical_and(model_results[:,0]<65000, #cell_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                         np.logical_and(model_results[:,0]<65000, #cell_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #cell number
                                         np.logical_and(model_results[:,5]<65000, #cell_number
                                         np.logical_and(model_results[:,6]<0.15,  #standard deviation
                                                        model_results[:,6]>0.05))))
        else:
            ValueError('could not identify posterior option')
#       
        my_posterior_samples = prior_samples[accepted_indices]
        
#         pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
#         pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','pairplot_extended_abc_' + option + '.pdf'))

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        my_posterior_samples[:,2]/=10000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient'])

        sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(151)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1.0)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,np.log10(40))
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,2.0)
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.22)
        plt.gca().set_xlim(0,12)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
#         plt.gca().set_ylim(0,0.035)
        plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_paper_' + option + '.pdf'))

    def xest_plot_example_deterministic_trace(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        # same plot as before for different transcription ("more_mrna") - not yet
        # our preferred hes5 values

        accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #cell number
                                    np.logical_and(model_results[:,5]<65000, #cell_number
                                    np.logical_and(model_results[:,6]<0.15,  #standard deviation
                                                   model_results[:,6]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
 
        this_parameter = my_posterior_samples[0]
        my_trajectory = hes5.generate_deterministic_trajectory( duration = 1000 + 5*1500,
                                                                repression_threshold = this_parameter[2],
                                                                mRNA_degradation_rate = np.log(2)/30,
                                                                protein_degradation_rate = np.log(2)/90,
                                                                translation_rate = this_parameter[1],
                                                                basal_transcription_rate = this_parameter[0],
                                                                hill_coefficient = this_parameter[4],
                                                                transcription_delay =  this_parameter[3],
                                                                initial_mRNA = 3,
                                                                initial_protein = this_parameter[2])

        my_trajectory = my_trajectory[my_trajectory[:,0]>1000]
        my_trajectory[:,0] -= 1000
        
        self.assertGreaterEqual(np.min(my_trajectory),0.0)
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,1]*100, label = 'mRNA*100', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_deterministic_oscillating_trajectory.pdf'))

    def xest_plot_agnostic_oscillating_variation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')

        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000,
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]>0.3)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        
        this_parameter = my_posterior_samples[0,:]
        # same plot as before for different transcription ("more_mrna") - not yet
        # our preferred hes5 values
        my_trajectory = hes5.generate_agnostic_noise_trajectory( duration = 1500,
                                                                 repression_threshold = this_parameter[2],
                                                                 mRNA_degradation_rate = np.log(2)/30,
                                                                 protein_degradation_rate = np.log(2)/90,
                                                                 translation_rate = this_parameter[1],
                                                                 basal_transcription_rate = this_parameter[0],
                                                                 transcription_delay = this_parameter[3],
                                                                 hill_coefficient = this_parameter[4],
                                                                 noise_strength = this_parameter[5],
                                                                 initial_mRNA = 3,
                                                                 initial_protein = this_parameter[2],
                                                                 equilibration_time = 1000.0)
        
        self.assertGreaterEqual(np.min(my_trajectory),0.0)
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*100, label = 'mRNA*100', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_agnostic_oscillating_trajectory.pdf'))

    def xest_plot_agnostic_not_oscillating_variation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')

        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000,
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]<0.05)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        
        this_parameter = my_posterior_samples[0,:]
        # same plot as before for different transcription ("more_mrna") - not yet
        # our preferred hes5 values
        my_trajectory = hes5.generate_agnostic_noise_trajectory( duration = 1500,
                                                                 repression_threshold = this_parameter[2],
                                                                 mRNA_degradation_rate = np.log(2)/30,
                                                                 protein_degradation_rate = np.log(2)/90,
                                                                 translation_rate = this_parameter[1],
                                                                 basal_transcription_rate = this_parameter[0],
                                                                 transcription_delay = this_parameter[3],
                                                                 hill_coefficient = this_parameter[4],
                                                                 noise_strength = this_parameter[5],
                                                                 initial_mRNA = 3,
                                                                 initial_protein = this_parameter[2],
                                                                 equilibration_time = 1000.0)
        
        self.assertGreaterEqual(np.min(my_trajectory),0.0)
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*100, label = 'mRNA*100', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_agnostic_not_oscillating_trajectory.pdf'))

    def xest_plot_agnostic_posterior_distributions(self):
        
        option = 'full'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000, #cell_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                         np.logical_and(model_results[:,0]<65000, #cell_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                         np.logical_and(model_results[:,0]<65000, #cell_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
        else:
            ValueError('could not identify posterior option')
#       
        my_posterior_samples = prior_samples[accepted_indices]
        
        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_agnostic_abc_' +  option + '.pdf'))

#         my_posterior_samples[:,2]/=10000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'Noise strength'])

        sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(161)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1.0)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(162)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,np.log10(40))
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,2.0)
#         plt.yticks([])
 
        my_figure.add_subplot(163)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.22)
        plt.gca().set_xlim(0,12)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(164))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
#         plt.gca().set_ylim(0,0.035)
        plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(165))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift.append(my_figure.add_subplot(166))
        sns.distplot(data_frame['Noise strength'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_ylim(0,0.35)
#         plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','agnostic_inference' + option + '.pdf'))
 
    def xest_plot_amplitude_distribution(self):
        option = 'agnostic_mean_and_coherence'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                                       model_results[:,0]<65000)) #cell_number
        elif option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000, #cell_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
            accepted_indices = np.where(model_results[:,3]>0.3)  #standard deviation
        elif option == 'mean_and_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'mean_and_period': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                                       model_results[:,2]<300))))  #standard deviation
        elif option == 'mean_and_period_and_coherence': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                        np.logical_and(model_results[:,2]<300,
                                                       model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'agnostic_prior':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = (range(len(prior_samples)),)
        elif option == 'agnostic_mean':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                                       model_results[:,0]<65000)) #cell_number
        elif option == 'agnostic_mean_and_coherence':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')

            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'not_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                         np.logical_and(model_results[:,0]<65000, #cell_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            ValueError('could not identify posterior option')

        my_posterior_samples = prior_samples[accepted_indices]
        print 'so many posterior samples'
        print len(my_posterior_samples)
        my_model_results = model_results[accepted_indices]

        my_posterior_samples[:,2]/=10000

        sns.set()
#         sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_standard_deviations = my_model_results[:,1]
        sns.distplot(all_standard_deviations,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     )
#                      bins = 20)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Standard deviation/mean HES5")
        plt.xlim(0,0.4)
        plt.axvline(0.05)
        plt.axvline(0.15)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_standard_deviation_' + option + '.pdf'))
 
    def xest_visualise_model_regimes(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 10
        figuresize = (6,2.5)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(1, 3 )

        coherence_bands = [[0,0.1],
                           [0.3,0.4],
                           [0.8,0.9]]
        
        panel_labels = {0: 'A', 1: 'B', 2: 'C'}

        for coherence_index, coherence_band in enumerate(coherence_bands):
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000, #cell_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>coherence_band[0],
                                                       model_results[:,3]<coherence_band[1]))))))

            my_posterior_results = model_results[accepted_indices]
            my_posterior_samples = prior_samples[accepted_indices]
       
            this_double_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec = outer_grid[coherence_index],
                    height_ratios= [number_of_traces, 1])
#                     wspace = 5.0)
            this_inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                    subplot_spec=this_double_grid[0], hspace=0.0)
            this_parameter = my_posterior_samples[0]
            this_results = my_posterior_results[0]

            for subplot_index in range(number_of_traces):
                this_axis = plt.Subplot(my_figure, this_inner_grid[subplot_index])
                my_figure.add_subplot(this_axis)
                this_trace = hes5.generate_langevin_trajectory(
                                                         duration = 1500,
                                                         repression_threshold = this_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = this_parameter[3],
                                                         basal_transcription_rate = this_parameter[0],
                                                         translation_rate = this_parameter[1],
                                                         initial_mRNA = 10,
                                                         hill_coefficient = this_parameter[4],
                                                         initial_protein = this_parameter[2],
                                                         equilibration_time = 1000)
                plt.plot(this_trace[:,0], this_trace[:,2]/1e4)
                plt.ylim(3,9)
#                 this_axis.locator_params(axis='y', tight = True, nbins=1)
#                 this_axis.locator_params(axis='y', nbins=2)
                this_axis.locator_params(axis='x', tight = True, nbins=3)
                plt.yticks([])
                this_axis.tick_params(axis='both', length = 1)
                if subplot_index == 0:
                    plt.title('Coherence: ' + '{:.2f}'.format(this_results[3]) + 
                              r', $\alpha_m =$ ' + '{:.2f}'.format(this_parameter[0]) +
                              r', $n =$ ' + '{:.2f}'.format(this_parameter[4]) +
                              '\n' + r'$\alpha_p =$ ' + '{:.2f}'.format(this_parameter[1]) + 
                              r', $p_0 = $ ' + '{:.2f}'.format(this_parameter[2]) + 
                              r', $\tau = $ ' + '{:.2f}'.format(this_parameter[3]),
                              fontsize = 5)
                    plt.gca().text(-0.2, 2.1, panel_labels[coherence_index], transform=plt.gca().transAxes)
                if subplot_index < number_of_traces - 1:
                    this_axis.xaxis.set_ticklabels([])
                if subplot_index !=9 or coherence_index != 0: 
                    this_axis.yaxis.set_ticklabels([])
                else:
                    plt.yticks([3,9])
                if coherence_index == 0 and subplot_index == 4:
                    plt.ylabel('Expression/1e4 ', labelpad = 15)
            plt.xlabel('Time [min]', labelpad = 2)
            plt.yticks([3,9])
            this_axis = plt.Subplot(my_figure, this_double_grid[1])
            my_figure.add_subplot(this_axis)
            plt.xlabel('Frequency [1/min]', labelpad = 2)
            _, these_traces = hes5.generate_multiple_langevin_trajectories(number_of_trajectories = 200,
                                                         duration = 1500*5,
                                                         repression_threshold = this_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = this_parameter[3],
                                                         basal_transcription_rate = this_parameter[0],
                                                         translation_rate = this_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = this_parameter[2],
                                                         hill_coefficient = this_parameter[4],
                                                         equilibration_time = 1000)
            this_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_traces)
            smoothened_power_spectrum = hes5.smoothen_power_spectrum(this_power_spectrum)
            plt.plot(this_power_spectrum[:,0], this_power_spectrum[:,1])
            this_axis.locator_params(axis='x', tight = True, nbins=3)
            this_axis.tick_params(axis='both', length = 1)
            if coherence_index == 0:
                plt.ylabel('Power', labelpad = 15)
            max_index = np.argmax(smoothened_power_spectrum[:,1])
            max_power_frequency = smoothened_power_spectrum[max_index,0]
            left_frequency = max_power_frequency*0.9
            right_frequency = max_power_frequency*1.1
            plt.axvline(left_frequency, color = 'black')
            plt.axvline(right_frequency, color = 'black')
            plt.xlim(0.0,0.01)
            plt.yticks([])
            plt.axhline(3)

        plt.tight_layout()
        my_figure.subplots_adjust(hspace = 0.7)
            
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','model_visualisation_for_paper.pdf'))
        
    def xest_plot_prior_for_paper(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        my_posterior_samples = prior_samples[accepted_indices]

        my_posterior_samples[:,2]/=10000
        prior_samples[:,2]/=10000

        data_frame = pd.DataFrame( data = prior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient'])

        sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(151)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,0.8)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,np.log10(40))
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation rate \n [1/min]")
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.22)
        plt.gca().set_xlim(0,12)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
        plt.gca().set_ylim(0,0.035)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','prior_for_paper.pdf'))

    def xest_plot_period_distribution_for_paper(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        real_data = [ 6.4135025721, 6.9483225932, 2.6887457703, 3.8620874625, 3.2559540745,
                      4.4568030424, 5.2120783369, 4.3169191105, 4.2472576997, 2.7684001434,
                      3.6331949226, 5.365000329,  1.1181243755, 4.2130976958, 6.3381760719,
                      2.466899605,  4.7849990718, 5.2029517316, 4.2038143391, 3.9909362984,
                      3.2734490618, 4.3116631965, 5.3199423883] 
        
        ## the values that verionica sent initially
#          
#         real_data = [2.0075009033, 5.1156200644, 7.7786868129, 6.4328452748, 7.441794935,
#                      7.0127707313, 2.6890681359, 3.4454911902, 3.8689181126, 3.2493764293,
#                      6.3817264371, 5.8903734106, 4.5034984657, 3.4247641996, 4.4767623623, 
#                      4.1803337503, 5.2752672662, 6.9038758003, 4.3200156205, 4.2588402084, 
#                      6.1428930891, 5.4124817274, 5.0135377758, 2.8156245427, 5.5008033408, 
#                      3.6331974295, 5.295813407,  1.1181243876, 5.5984263674, 4.2800118281, 
#                      6.7713656265, 3.4585300534, 6.3727670575, 2.4668994841, 6.3725171059,
#                      4.8021898758, 4.8108333392, 5.9935335349, 6.2570622822, 5.2284704987,
#                      4.2143881493, 4.0659270434, 3.9990674449, 4.4410420437, 6.7406002947,
#                      5.0648853886, 1.8765732885, 3.307425174,  5.6208186717, 4.3185605778,
#                      5.186842823,  5.6310823986, 7.4402931009]

        sns.set(font_scale = 1.5)
        font = {'size'   : 28}
        plt.rc('font', **font)

        all_periods = my_model_results[:,2]
# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                   'Data' : np.array(real_data)*60})
        
#         print('mean period is')
#         print(np.mode(all_periods[all_periods<600]))
#         import pdb; pdb.set_trace()
        my_figure = plt.figure(figsize= (5,3))
        sns.boxplot(data = [all_periods[all_periods<600], np.array(real_data)*60])
        plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.ylabel('Period [min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_for_paper.pdf'))
 
    def xest_plot_agnostic_period_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                    prior_samples[:,-1]<20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        real_data = [ 6.4135025721, 6.9483225932, 2.6887457703, 3.8620874625, 3.2559540745,
                      4.4568030424, 5.2120783369, 4.3169191105, 4.2472576997, 2.7684001434,
                      3.6331949226, 5.365000329,  1.1181243755, 4.2130976958, 6.3381760719,
                      2.466899605,  4.7849990718, 5.2029517316, 4.2038143391, 3.9909362984,
                      3.2734490618, 4.3116631965, 5.3199423883] 
        
        ## the values that verionica sent initially
#          
#         real_data = [2.0075009033, 5.1156200644, 7.7786868129, 6.4328452748, 7.441794935,
#                      7.0127707313, 2.6890681359, 3.4454911902, 3.8689181126, 3.2493764293,
#                      6.3817264371, 5.8903734106, 4.5034984657, 3.4247641996, 4.4767623623, 
#                      4.1803337503, 5.2752672662, 6.9038758003, 4.3200156205, 4.2588402084, 
#                      6.1428930891, 5.4124817274, 5.0135377758, 2.8156245427, 5.5008033408, 
#                      3.6331974295, 5.295813407,  1.1181243876, 5.5984263674, 4.2800118281, 
#                      6.7713656265, 3.4585300534, 6.3727670575, 2.4668994841, 6.3725171059,
#                      4.8021898758, 4.8108333392, 5.9935335349, 6.2570622822, 5.2284704987,
#                      4.2143881493, 4.0659270434, 3.9990674449, 4.4410420437, 6.7406002947,
#                      5.0648853886, 1.8765732885, 3.307425174,  5.6208186717, 4.3185605778,
#                      5.186842823,  5.6310823986, 7.4402931009]

        my_posterior_samples[:,2]/=10000

        real_data = np.array(real_data)*60
        sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]/60
        sns.distplot(all_periods[all_periods<10],
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 10)
#         plt.gca().set_xlim(-1,2)
        plt.axvline(np.mean(real_data)/60)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
        plt.xlim(1,10)
        plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_agnostic.pdf'))
 
    def xest_plot_period_distribution_for_paper(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        real_data = [ 6.4135025721, 6.9483225932, 2.6887457703, 3.8620874625, 3.2559540745,
                      4.4568030424, 5.2120783369, 4.3169191105, 4.2472576997, 2.7684001434,
                      3.6331949226, 5.365000329,  1.1181243755, 4.2130976958, 6.3381760719,
                      2.466899605,  4.7849990718, 5.2029517316, 4.2038143391, 3.9909362984,
                      3.2734490618, 4.3116631965, 5.3199423883] 
        
        ## the values that verionica sent initially
#          
#         real_data = [2.0075009033, 5.1156200644, 7.7786868129, 6.4328452748, 7.441794935,
#                      7.0127707313, 2.6890681359, 3.4454911902, 3.8689181126, 3.2493764293,
#                      6.3817264371, 5.8903734106, 4.5034984657, 3.4247641996, 4.4767623623, 
#                      4.1803337503, 5.2752672662, 6.9038758003, 4.3200156205, 4.2588402084, 
#                      6.1428930891, 5.4124817274, 5.0135377758, 2.8156245427, 5.5008033408, 
#                      3.6331974295, 5.295813407,  1.1181243876, 5.5984263674, 4.2800118281, 
#                      6.7713656265, 3.4585300534, 6.3727670575, 2.4668994841, 6.3725171059,
#                      4.8021898758, 4.8108333392, 5.9935335349, 6.2570622822, 5.2284704987,
#                      4.2143881493, 4.0659270434, 3.9990674449, 4.4410420437, 6.7406002947,
#                      5.0648853886, 1.8765732885, 3.307425174,  5.6208186717, 4.3185605778,
#                      5.186842823,  5.6310823986, 7.4402931009]

        my_posterior_samples[:,2]/=10000

        real_data = np.array(real_data)*60
        sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]/60
        sns.distplot(all_periods[all_periods<10],
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     bins = 10)
#         plt.gca().set_xlim(-1,2)
        plt.axvline(np.mean(real_data)/60)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
        plt.xlim(1,10)
        plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_for_paper.pdf'))
 
    def xest_plot_agnostic_mrna_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #time_delay
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        my_posterior_samples[:,2] /= 10000

        sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_mrna = my_model_results[:,4]
        sns.distplot(all_mrna,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True)
#                      norm_hist = True,
#                      bins = 10)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("mean mRNA number")
#         plt.xlim(1,80)
#         plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_mrna_distribution_agnostic.pdf'))
 
    def xest_plot_agnostic_noise_amplitude_correlation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                    model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                   model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        my_figure = plt.figure(figsize= (4.5,2.5))
        plt.scatter(prior_samples[:,-1],model_results[:,1], lw = 0, s = 1, zorder = 0)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel('Noise strength [1/min]')
        plt.ylabel('std/mean')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','agnostic_noise_vs_amplitude.pdf'),dpi = 400)

    def xest_plot_mrna_distribution_for_paper(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        my_posterior_samples[:,2] /= 10000

        sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_mrna = my_model_results[:,4]
        sns.distplot(all_mrna[all_mrna<80],
                     kde = False,
                     rug = False,
                     norm_hist = True)
#                      norm_hist = True,
#                      bins = 10)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("mean mRNA number")
        plt.xlim(1,80)
        plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_mrna_distribution_for_paper.pdf'))
 
    def xest_plot_period_distribution_differently(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.1)))))

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        real_data = [ 6.4135025721, 6.9483225932, 2.6887457703, 3.8620874625, 3.2559540745,
                      4.4568030424, 5.2120783369, 4.3169191105, 4.2472576997, 2.7684001434,
                      3.6331949226, 5.365000329,  1.1181243755, 4.2130976958, 6.3381760719,
                      2.466899605,  4.7849990718, 5.2029517316, 4.2038143391, 3.9909362984,
                      3.2734490618, 4.3116631965, 5.3199423883] 
        
        ## the values that verionica sent initially
#          
#         real_data = [2.0075009033, 5.1156200644, 7.7786868129, 6.4328452748, 7.441794935,
#                      7.0127707313, 2.6890681359, 3.4454911902, 3.8689181126, 3.2493764293,
#                      6.3817264371, 5.8903734106, 4.5034984657, 3.4247641996, 4.4767623623, 
#                      4.1803337503, 5.2752672662, 6.9038758003, 4.3200156205, 4.2588402084, 
#                      6.1428930891, 5.4124817274, 5.0135377758, 2.8156245427, 5.5008033408, 
#                      3.6331974295, 5.295813407,  1.1181243876, 5.5984263674, 4.2800118281, 
#                      6.7713656265, 3.4585300534, 6.3727670575, 2.4668994841, 6.3725171059,
#                      4.8021898758, 4.8108333392, 5.9935335349, 6.2570622822, 5.2284704987,
#                      4.2143881493, 4.0659270434, 3.9990674449, 4.4410420437, 6.7406002947,
#                      5.0648853886, 1.8765732885, 3.307425174,  5.6208186717, 4.3185605778,
#                      5.186842823,  5.6310823986, 7.4402931009]

        my_posterior_samples[:,2]/=10000

        real_data = np.array(real_data)*60
        sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (6,4))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                   'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]
        my_figure.add_subplot(211)
        sns.distplot(all_periods[all_periods<600],
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    bins = 10)
#         plt.gca().set_xlim(-1,2)
        plt.axvline(np.mean(real_data))
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [min]")
        plt.xlim(50,600)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        my_figure.add_subplot(212)
        sns.distplot(real_data,
                     kde = False,
                     rug = False,
                     norm_hist = True)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(0,2.3)
#         plt.gca().set_ylim(0,2.0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.xticks([0,1,2], [r'$10^0$',r'$10^1$',r'$10^2$'])
        plt.xlabel("Measured period [min]")
        plt.ylabel("Occurrence", labelpad = 20)
        plt.xlim(50,600)
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_differently.pdf'))
 
    def xest_plot_mrna_distribution_for_paper(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                    model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        sns.set(font_scale = 1.5)
        font = {'size'   : 28}
        plt.rc('font', **font)

        all_mrna_levels = my_model_results[:,4]
# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                   'Data' : np.array(real_data)*60})
        my_figure = plt.figure(figsize= (5,3))
        sns.distplot(all_mrna_levels,
                     kde = False,
                     rug = False,
                     norm_hist = True)
#         plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.xlabel('<mRNA>')
        plt.ylabel('Likelihood')
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
        plt.ylim(0,0.05)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_mrna_distribution_for_paper.pdf'))
 
    def xest_make_degradation_rate_sweep(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_sweep_results = hes5.conduct_parameter_sweep_at_parameters('protein_degradation_rate',
                                          my_posterior_samples,
                                          number_of_sweep_values = number_of_parameter_points,
                                          number_of_traces_per_parameter = number_of_trajectories,
                                          relative = False)

        np.save(os.path.join(os.path.dirname(__file__), 'output','extended_degradation_sweep.npy'),
                    my_sweep_results)

    def test_make_relative_delay_parameter_variation(self):
        number_of_parameter_points = 2
        number_of_trajectories = 1
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_parameter_sweep_at_parameters('time_delay',
                                                                                my_posterior_samples,
                                                                                number_of_parameter_points,
                                                                                number_of_trajectories,
                                                                                relative = True)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output','extended_relative_sweeps_' + 'time_delay' + '.npy'),
                    my_parameter_sweep_results)

    def xest_make_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','extended_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_model_prediction(self):
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

        parameter_names = ['protein_degradation_rate']

        x_labels = dict()
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'narrowed_relative_sweeps_' + parameter_name + '.npy'))
            
            increase_indices = np.where(my_parameter_sweep_results[:,9,3] < 300)
 
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
#             my_sweep_parameters = my_posterior_samples[increase_indices]
            
            x_coord = -0.4
            y_coord = 1.1
            my_figure = plt.figure( figsize = (4.5, 1.5) )
            this_axis = my_figure.add_subplot(121)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='teal', alpha = 0.02, zorder = 0)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            plt.gca().set_rasterization_zorder(1)
            plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(122)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            plt.gca().set_rasterization_zorder(1)
            plt.gca().text(x_coord, y_coord, 'B', transform=plt.gca().transAxes)
            this_axis.set_ylim(0,1)
#             this_axis.set_ylim(0,0.5)
#             this_axis.set_ylim(0,0.25)
            
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','model_prediction_' + parameter_name + '.pdf'), dpi = 400)
 
    def xest_plot_bifurcation_implementation(self):
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

        my_figure = plt.figure( figsize = (6.5, 1.5) )

        my_figure.add_subplot(131)
        my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                                          'narrowed_degradation_sweep.npy'))
        
        x_coord = -0.3
        y_coord = 1.05
        for results_table in my_degradation_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
        plt.axvline( np.log(2)/90, color = 'darkblue' )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Coherence')
        plt.ylim(0,1)
        plt.xlim(0,np.log(2)/15.)
        plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)

        my_figure.add_subplot(132)
        hill_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'narrowed_relative_sweeps_hill_coefficient.npy'))
        for results_table in hill_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel('rel. Hill coefficient')
        plt.axvline( 1.0, color = 'darkblue' )
        plt.gca().text(x_coord, y_coord, 'B', transform=plt.gca().transAxes)
        plt.ylim(0,1)               
        plt.xlim(0.1,2)

        my_figure.add_subplot(133)
        delay_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'narrowed_relative_sweeps_time_delay.npy'))
        for results_table in delay_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().set_rasterization_zorder(1)
        plt.axvline( 1.0, color = 'darkblue')
        plt.xlabel('rel. Transcription delay')
        plt.gca().text(x_coord, y_coord, 'C', transform=plt.gca().transAxes)
        plt.ylim(0,1)
        plt.xlim(0.1,2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','bifurcation_illustration.pdf'), dpi = 400)
 
    def xest_plot_bayes_factors_for_models(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        number_of_absolute_samples = len(accepted_indices[0])
        print 'base model accepted that many indices'
        print number_of_absolute_samples
        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'Transcription rate'
        x_labels['translation_rate'] = 'Translation rate'
        x_labels['repression_threshold'] = 'Repression threshold' 
        x_labels['time_delay'] = 'Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'mRNA degradation'
        x_labels['protein_degradation_rate'] = 'Protein degradation'
        x_labels['hill_coefficient'] = 'Hill coefficient'

        decrease_ratios = dict()
        increase_ratios = dict()
        bardata = []
        for parameter_name in parameter_names:
            print 'investigating ' + parameter_name
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'narrowed_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print 'these accepted base samples are'
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                    my_parameter_sweep_results[:,9,4] < 0.1))[0])
            print number_of_absolute_samples
            
            decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                        my_parameter_sweep_results[:,4,4] > 0.1)))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                        my_parameter_sweep_results[:,14,4] > 0.1)))

            increase_ratios[parameter_name] = len(increase_indices[0])/float(number_of_absolute_samples)
                
        increase_bars = [increase_ratios[parameter_name] for parameter_name
                         in parameter_names]

        decrease_bars = [decrease_ratios[parameter_name] for parameter_name
                         in parameter_names]

        increase_positions = np.arange(len(increase_bars))
        decrease_positions = np.arange(len(decrease_bars)) + len(increase_bars)
        all_positions = np.hstack((increase_positions, decrease_positions))
        
        all_bars = np.array( increase_bars + decrease_bars)

        labels_up = [x_labels[parameter_name] + ' up' for parameter_name in parameter_names]
        labels_down = [x_labels[parameter_name] + ' down' for parameter_name in parameter_names]
        
        all_labels = labels_up + labels_down
        sorting_indices = np.argsort(all_bars)
        sorted_labels = [all_labels[sorting_index] for
                         sorting_index in sorting_indices]
        sorted_bars = np.sort(all_bars)
        sorted_bars/= np.sum(sorted_bars)

        my_figure = plt.figure( figsize = (4.5, 1.5) )
        plt.bar(all_positions, sorted_bars[::-1])
        sorted_labels.reverse()
#         plt.xticks( all_positions + 0.4 , 
        plt.xticks( all_positions, 
                    sorted_labels,
                    rotation = 30,
                    fontsize = 5,
                    horizontalalignment = 'right')
        plt.xlim(all_positions[0] - 0.5,)
        plt.gca().locator_params(axis='y', tight = True, nbins=5)
        plt.ylim(0,sorted_bars[-1]*1.2)
        plt.ylabel('Likelihood')

        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output',
                                     'likelihood_plot_for_paper.pdf'))

    def xest_plot_power_spectra_before(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        number_of_absolute_samples = len(accepted_indices[0])
        
        # where is coherence less than 0.1 or period larger than 600
        new_accepted_indices = np.where(np.logical_or(accepted_model_results[:,2] > 600,
                                                      accepted_model_results[:,3] < 0.1))
        
        these_posterior_samples = my_posterior_samples[new_accepted_indices]
        #downsample to 100
#         fewer_samples = these_posterior_samples[:1000]
        fewer_samples = these_posterior_samples
        
        power_spectra = hes5.calculate_power_spectra_at_parameter_points(fewer_samples)
        
        my_figure = plt.figure( figsize = (4.5, 1.5) )
        for power_spectrum in power_spectra[:,1:].transpose():
            plt.plot(power_spectra[:,0], power_spectrum, ls = 'solid',
                     color ='teal', alpha = 0.02, zorder = 0)
        plt.plot(power_spectra[:,0], np.mean(power_spectra[:,1:],axis = 1), ls = 'solid',
                     color ='blue')
#         plt.gca().locator_params(axis='y', tight = True, nbins=5)
#         plt.ylim(0,sorted_bars[-1]*1.2)
        plt.xlim(0.0,0.01)
        plt.gca().set_rasterization_zorder(1)
        plt.ylabel('Power')
        plt.xlabel('Frequency [1/min]')

        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output',
                                       'power_spectra_before.pdf'), dpi = 400)

    def xest_plot_power_spectra_before_and_after(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        index_to_parameter_name_lookup = {0: 'basal_transcription_rate',
                                          1: 'translation_rate',
                                          2: 'repression_threshold',
                                          3: 'time_delay',
                                          4: 'hill_coefficient',
                                          5: 'mRNA_degradation_rate',
                                          6: 'protein_degradation_rate'}

        parameter_name_to_index_lookup = {'basal_transcription_rate':0,
                                          'translation_rate'        :1, 
                                          'repression_threshold'    :2,
                                          'time_delay'              :3,
                                          'hill_coefficient'        :4,
                                          'mRNA_degradation_rate'   :5,
                                          'protein_degradation_rate':6 }


        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']
        
        for parameter_name in parameter_names:
            print 'investigating ' + parameter_name
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'narrowed_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print 'these accepted base samples are'
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                    my_parameter_sweep_results[:,9,4] < 0.1))[0])
            print number_of_absolute_samples
            
            decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                        my_parameter_sweep_results[:,4,4] > 0.1)))

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                        my_parameter_sweep_results[:,14,4] > 0.1)))

            decrease_parameters_before = my_posterior_samples[decrease_indices]
            increase_parameters_before = my_posterior_samples[increase_indices]
            print('number of accepted samples is ' + str(len(decrease_indices[0])))
            print('number of accepted samples is ' + str(len(increase_indices[0])))
            print('these are the before parameters')
            print decrease_parameters_before
            print increase_parameters_before
            
#             if len(decrease_parameters_before) > 10:
#                 decrease_parameters_before = decrease_parameters_before[:10]
#             if len(increase_parameters_before) > 10:
#                 increase_parameters_before = increase_parameters_before[:10]

            dummy_zeros = np.zeros((decrease_parameters_before.shape[0],2))
            decrease_parameters_after = np.hstack((decrease_parameters_before,dummy_zeros))
            dummy_zeros = np.zeros((increase_parameters_before.shape[0],2))
            increase_parameters_after = np.hstack((increase_parameters_before,dummy_zeros))
            decrease_parameters_after[:,-2] = np.log(2.)/30.
            increase_parameters_after[:,-2] = np.log(2.)/30.
            decrease_parameters_after[:,-1] = np.log(2.)/90.
            increase_parameters_after[:,-1] = np.log(2.)/90.
            
            print 'these are the increase parameters after'
            print increase_parameters_after
            parameter_index = parameter_name_to_index_lookup[parameter_name]
            
            try:
                print'hello1'
                reference_decrease_parameters = decrease_parameters_after[:,parameter_index]
                print reference_decrease_parameters
                decreased_parameters = reference_decrease_parameters*0.5
                print'hello2'
                decrease_parameters_after[:,parameter_index] = decreased_parameters
                print'hello3'
                decrease_spectra_before = hes5.calculate_power_spectra_at_parameter_points(decrease_parameters_before)
                print'hello4'
                decrease_spectra_after = hes5.calculate_power_spectra_at_parameter_points(decrease_parameters_after)
                print'hello5'
            except Exception, e: 
                print repr(e)
                decrease_spectra_before = np.array([[0,0],[0,0]])
                decrease_spectra_after = np.array([[0,0],[0,0]])

            try:   
                print'hello1'
                reference_increase_parameters = increase_parameters_after[:,parameter_index]
                print reference_increase_parameters
                increased_parameters = reference_increase_parameters*1.5
                print'hello2'
                increase_parameters_after[:,parameter_index] = increased_parameters
                print'hello3'
                increase_spectra_before = hes5.calculate_power_spectra_at_parameter_points(increase_parameters_before)
                print'hello4'
                increase_spectra_after = hes5.calculate_power_spectra_at_parameter_points(increase_parameters_after)
                print'hello5'
            except Exception, e: 
                print repr(e)
                increase_spectra_before = np.array([[0,0],[0,0]])
                increase_spectra_after = np.array([[0,0],[0,0]])

            my_figure = plt.figure( figsize = (6, 4.5) )
            my_figure.add_subplot(221)
            for power_spectrum in decrease_spectra_before[:,1:].transpose():
                plt.plot(decrease_spectra_before[:,0], power_spectrum, ls = 'solid',
                         color ='teal', alpha = 0.2, zorder = 0)
            plt.plot(decrease_spectra_before[:,0], 
                     np.mean(decrease_spectra_before[:,1:],axis = 1), ls = 'solid',
                     color ='blue')
            plt.xlim(0.0,0.01)
            plt.axvline(1/300.0)
            plt.axvline(1/600.0)
#             plt.gca().set_rasterization_zorder(1)
            plt.ylabel('Power')
            plt.title(parameter_name + ' decrease before')
            plt.xlabel('Frequency [1/min]')

            my_figure.add_subplot(222)
            for power_spectrum in decrease_spectra_after[:,1:].transpose():
                plt.plot(decrease_spectra_after[:,0], power_spectrum, ls = 'solid',
                         color ='teal', alpha = 0.2, zorder = 0)
            plt.plot(decrease_spectra_after[:,0], 
                     np.mean(decrease_spectra_after[:,1:],axis = 1), ls = 'solid',
                     color ='blue')
            plt.xlim(0.0,0.01)
            plt.axvline(1/300.0)
            plt.axvline(1/600.0)
#             plt.gca().set_rasterization_zorder(1)
            plt.ylabel('Power')
            plt.title(parameter_name + ' decrease after')
            plt.xlabel('Frequency [1/min]')

            my_figure.add_subplot(223)
            for power_spectrum in increase_spectra_before[:,1:].transpose():
                plt.plot(increase_spectra_before[:,0], power_spectrum, ls = 'solid',
                         color ='teal', alpha = 0.2, zorder = 0)
            plt.plot(increase_spectra_before[:,0], 
                     np.mean(increase_spectra_before[:,1:],axis = 1), ls = 'solid',
                     color ='blue')
            plt.xlim(0.0,0.01)
            plt.axvline(1/300.0)
            plt.axvline(1/600.0)
#             plt.gca().set_rasterization_zorder(1)
            plt.ylabel('Power')
            plt.title(parameter_name + ' increase before')
            plt.xlabel('Frequency [1/min]')

            my_figure.add_subplot(224)
            for power_spectrum in increase_spectra_after[:,1:].transpose():
                plt.plot(increase_spectra_after[:,0], power_spectrum, ls = 'solid',
                         color ='teal', alpha = 0.2, zorder = 0)
            plt.plot(increase_spectra_after[:,0], 
                     np.mean(increase_spectra_after[:,1:],axis = 1), ls = 'solid',
                     color ='blue')
            plt.xlim(0.0,0.01)
            plt.axvline(1/300.0)
            plt.axvline(1/600.0)
            plt.gca().set_rasterization_zorder(1)
            plt.ylabel('Power')
            plt.title(parameter_name + ' increase after')
            plt.xlabel('Frequency [1/min]')

            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output',
                                     parameter_name + 'likelihood_plot_spectra_investigation.pdf'))
