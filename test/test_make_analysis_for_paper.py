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

        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (1,40),
                        'repression_threshold' : (0,120000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        'noise_strength' : (0,20)}

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
        
    def xest_plot_phase_space(self):
        # phase space: we have two options: mrna vs protein or 
        # protein vs dprotein (or mrna vs dmrna)
        # let's start w protein vs mrna
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        # same plot as before for different transcription ("more_mrna") - not yet
        # our preferred hes5 values

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]>0.8)))))

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

#         my_stochastic_trajectory =  hes5.generate_langevin_trajectory( duration = 1000 + 5*1500,
        my_stochastic_trajectory =  hes5.generate_langevin_trajectory( duration = 1000+ 5*1500,
                                                                repression_threshold = this_parameter[2],
                                                                mRNA_degradation_rate = np.log(2)/30,
                                                                protein_degradation_rate = np.log(2)/90,
                                                                translation_rate = this_parameter[1],
                                                                basal_transcription_rate = this_parameter[0],
                                                                hill_coefficient = this_parameter[4],
                                                                transcription_delay =  this_parameter[3],
                                                                initial_mRNA = 3,
                                                                initial_protein = this_parameter[2],
                                                                equilibration_time = 0)
#                                                                 equilibration_time = 1000.0)


#         my_trajectory = my_trajectory[my_trajectory[:,0]>1000]
#         my_trajectory[:,0] -= 1000
        
        figuresize = (4,5)
        my_figure = plt.figure(figsize = figuresize)
        my_figure.add_subplot(311)
        plt.plot(my_trajectory[:,0],my_trajectory[:,2], color = 'black', alpha = 0.3)
        plt.xlabel('Time')
        plt.ylabel('det. Protein')
        my_figure.add_subplot(312)
        plt.plot(my_stochastic_trajectory[:,0],my_stochastic_trajectory[:,2], color = 'black', alpha = 0.3)
        plt.xlabel('Time')
        plt.ylabel('stoch. Protein')
        my_figure.add_subplot(313)
        plt.plot(my_trajectory[:-1,1],my_trajectory[:-1,2], color = 'black', alpha = 0.3)
        plt.plot(my_stochastic_trajectory[:,1],my_stochastic_trajectory[:,2], color = 'blue', alpha = 0.3)
        plt.xlabel('mRNA')
        plt.ylabel('Protein')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_phase_space_analysis.pdf'))

    def xest_plot_posterior_distributions(self):
        
        option = 'amplitude'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                       model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                                       model_results[:,1]>0.05)))  #standard deviation
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #protein number
                                         np.logical_and(model_results[:,5]<65000, #protein_number
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

    def xest_plot_deterministic_posterior_distributions(self):
        
        option = 'deterministic'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #protein number
                                         np.logical_and(model_results[:,5]<65000, #protein_number
#                                          np.logical_and(model_results[:,6]<0.15,  #standard deviation
#                                                         model_results[:,6]>0.05))))
                                                        model_results[:,6]>0.05)))
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

        print('minimum time delay is')
        print(np.min(data_frame['Transcription delay']))
        print('minimum hill coefficient is')
        print(np.min(data_frame['Hill coefficient']))

        sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(151)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),10)
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
        # plt.gca().set_ylim(0,2.0)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,np.log10(40),10)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,np.log10(40))
#         plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation rate \n [1/min]")
#         plt.gca().set_ylim(0,4.0)
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 5)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
#         plt.gca().set_ylim(0,0.22)
        plt.gca().set_xlim(0,12)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
#         time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = 3)
                     # bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
#         plt.gca().set_ylim(0,0.035)
#         plt.gca().set_ylim(0,0.08)
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
                     bins = 5)
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_ylim(0,0.7)
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

        accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #protein number
                                    np.logical_and(model_results[:,5]<65000, #protein_number
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

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
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

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
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

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
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
 
    def xest_plot_amplitude_distribution_for_paper(self):
        option = 'lower_amplitude'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
            accepted_indices = np.where(model_results[:,3]>0.3)  #standard deviation
        elif option == 'mean_and_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'mean_and_period': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                                       model_results[:,2]<300))))  #standard deviation
        elif option == 'mean_and_period_and_coherence': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                        np.logical_and(model_results[:,2]<300,
                                                       model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'lower_amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                                       model_results[:,1]>0.05)))  #standard deviation
        elif option == 'agnostic_prior':
            saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = (range(len(prior_samples)),)
        elif option == 'agnostic_mean':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'agnostic_mean_and_coherence':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')

            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'not_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
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
        print 'maximal standard_deviation is'
        print(np.max(all_standard_deviations))
        print 'number of samples above 0.15'
        print(np.sum(all_standard_deviations>0.15))
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
        plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_single_standard_deviation_' + option + '.pdf'))
 
    def xest_plot_amplitude_distribution(self):
        option = 'lower_amplitude'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        measured_data = [0.115751936, 0.09571043,  0.070593436, 0.074172953, 0.079566358, 0.04600834, 
                         0.079873319, 0.097029606, 0.084070369, 0.105528875, 0.12579082,  0.042329269, 
                         0.064591498, 0.059602288, 0.057944518, 0.051163091, 0.058111095, 0.102434224, 
                         0.080997961, 0.070390139, 0.047127818, 0.095665455, 0.048707284, 0.083330235, 
                         0.072446835, 0.059289326, 0.175901785, 0.08870091,  0.060774517, 0.119311781, 
                         0.071923541, 0.106271586, 0.063191815, 0.068603169, 0.051063533, 0.074326763, 
                         0.030455154, 0.09777155,  0.07789995,  0.052264432, 0.107642115, 0.078060039, 
                         0.053932836, 0.04064868,  0.080203462, 0.102682858, 0.085553023, 0.050921194, 
                         0.107150422, 0.075111352, 0.085250494, 0.06022623,  0.055863624, 0.070855159, 
                         0.072975538, 0.038283748, 0.05842959,  0.069960347, 0.075625282, 0.033601918, 
                         0.10112012,  0.069907351, 0.047498028, 0.054963426, 0.015357264, 0.091893038, 
                         0.030862283, 0.012518025, 0.038223482, 0.05825977,  0.072195839, 0.020020349, 
                         0.05988876,  0.054678433, 0.08156298,  0.075856751, 0.080105646, 0.084244903, 
                         0.060850253, 0.079889701, 0.114204526, 0.048641408, 0.087017989, 0.072664986, 
                         0.135295363, 0.044380981, 0.024025198, 0.068262356, 0.019802578, 0.064603775, 
                         0.076865303, 0.083760066, 0.059606547, 0.05627585,  0.050701138, 0.064442271, 
                         0.073845055, 0.086630591, 0.034115231, 0.036910128, 0.05845354,  0.055185653, 
                         0.081778966, 0.041642038, 0.032706612, 0.034264942, 0.076971854, 0.046987517, 
                         0.060216471, 0.091438729, 0.0341048,   0.072119114, 0.050266261, 0.076173687, 
                         0.059316138, 0.07362588,  0.043229577, 0.056437502, 0.042911643, 0.072583345, 
                         0.069809296, 0.063362361, 0.051916029, 0.042110911, 0.071238566, 0.069599676, 
                         0.056064602, 0.055051899, 0.063226639, 0.076379692, 0.158771206, 0.037536219, 
                         0.055238055, 0.074217076, 0.094215882, 0.057284261, 0.066521902, 0.075479027, 
                         0.0921231,   0.078040383, 0.07767914,  0.053502299, 0.083650072, 0.084202846, 
                         0.065188768, 0.057116998, 0.079006745, 0.058366725, 0.062152612, 0.062281059, 
                         0.036391176, 0.079608123, 0.05814215,  0.084222668, 0.071304801, 0.09422804, 
                         0.106918005, 0.110727013, 0.10753385,  0.078788611, 0.07298067,  0.078655859, 
                         0.045046025, 0.061084624, 0.085156637, 0.109648343, 0.06425073,  0.096245619,
                         0.056215123, 0.085664518, 0.066525248, 0.088294766, 0.055145696, 0.075250338, 
                         0.04822837,  0.019409385, 0.047170987, 0.030422279, 0.0818539, 0.07351729, 
                         0.083877723] 

        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
            accepted_indices = np.where(model_results[:,3]>0.3)  #standard deviation
        elif option == 'mean_and_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'mean_and_period': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                                       model_results[:,2]<300))))  #standard deviation
        elif option == 'mean_and_period_and_coherence': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                        np.logical_and(model_results[:,2]<300,
                                                       model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'lower_amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                                       model_results[:,1]>0.05)))  #standard deviation
        elif option == 'agnostic_prior':
            saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = (range(len(prior_samples)),)
        elif option == 'agnostic_mean':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'agnostic_mean_and_coherence':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')

            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'not_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
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
        my_figure = plt.figure(figsize= (4.5,3))
        my_figure.add_subplot(211)

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_standard_deviations = my_model_results[:,1]
        plt.axvline(np.mean(all_standard_deviations))
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
        plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        my_figure.add_subplot(212)
        sns.distplot(measured_data,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     )
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Standard deviation/mean HES5")
        plt.axvline(np.mean(measured_data))
        print 'maximal measured value'
        print np.max(measured_data)
        plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_standard_deviation_' + option + '.pdf'))
 
        my_boxplot_figure = plt.figure(figsize = [4,2.5])
        sns.boxplot(data = [all_standard_deviations, measured_data])
        plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.ylabel('Period [min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_standard_deviation_boxplot_' + option + '.pdf'))
 
    def xest_visualise_model_regimes(self):
        # sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 10
        figuresize = (6,2.5)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(1, 3 )

        coherence_bands = [[0,0.05],
                           [0.45,0.47],
                           [0.85,0.9]]
        
        panel_labels = {0: 'A', 1: 'B', 2: 'C'}

        for coherence_index, coherence_band in enumerate(coherence_bands):
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
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
                plt.plot(this_trace[:,0], this_trace[:,2]/1e4, lw = 1)
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
                plt.xlim(0,1500)
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
            plt.plot(this_power_spectrum[:,0], this_power_spectrum[:,1], lw = 1)
            this_axis.locator_params(axis='x', tight = True, nbins=3)
            this_axis.tick_params(axis='both', length = 1)
            if coherence_index == 0:
                plt.ylabel('Power', labelpad = 15)
            max_index = np.argmax(smoothened_power_spectrum[:,1])
            max_power_frequency = smoothened_power_spectrum[max_index,0]
            left_frequency = max_power_frequency*0.9
            right_frequency = max_power_frequency*1.1
            plt.axvline(left_frequency, color = 'black', lw = 1 )
            plt.axvline(right_frequency, color = 'black', lw = 1 )
            plt.xlim(0.0,0.01)
            plt.yticks([])
            # plt.axhline(3)

        plt.tight_layout()
        my_figure.subplots_adjust(hspace = 0.7)
            
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','model_visualisation_for_paper.pdf'))
        
    def xest_plot_prior_for_paper(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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

    def xest_plot_period_distribution_boxplot(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
#         accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
#                                                    model_results[:,0]<65000))  #standard deviation
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

#         accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
#                                     np.logical_and(model_results[:,0]<65000, #protein_number
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3)))) #time_delay

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
        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]/60
        print('mean is')
        print(np.mean(all_periods[all_periods<10]))
        print('median is')
        print(np.median(all_periods[all_periods<10]))
        print('data mean is')
        print(np.mean(real_data)/60)
        period_histogram, bins = np.histogram(all_periods[all_periods<10], bins = 400) 
        maximum_index = np.argmax(period_histogram)
        print('max bin is')
        print bins[maximum_index]
        print bins[maximum_index+1]
        print bins[maximum_index+2]
        print bins[maximum_index-1]
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
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_for_paper.pdf'))
 
    def xest_plot_agnostic_mrna_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
#                                                     model_results[:,1]>0.05)))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    model_results[:,3]>0.3))))#time_delay
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #time_delay
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        my_posterior_samples[:,2] /= 10000

        sns.set()
#         sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
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
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_agnostic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                    model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        my_figure = plt.figure(figsize= (4.5,2.5))
        plt.scatter(my_posterior_samples[:,-1],my_model_results[:,1], lw = 0, s = 1, zorder = 0, alpha = 0.2)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel('Noise strength [1/min]')
        plt.ylabel('std/mean')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','agnostic_noise_vs_amplitude.pdf'),dpi = 400)

    def xest_plot_mrna_distribution_for_paper(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     model_results[:,3]>0.3))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        my_posterior_samples[:,2] /= 10000

        weird_index = np.where(my_model_results[:,4]>200)
        weird_results = my_model_results[weird_index]
        weird_posterior = my_posterior_samples[weird_index]
        print weird_results
        print weird_posterior
        sns.set()
#         sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_mrna = my_model_results[:,4]
        print('minimum and maximum are')
        print(np.min(all_mrna))
        print(np.max(all_mrna))
        print('so many samples above 100')
        print(np.sum(all_mrna>100))
        mrna_histogram, bins = np.histogram(all_mrna, bins = 400) 
        maximum_index = np.argmax(mrna_histogram)
        print('max bin is')
        print bins[maximum_index]
        print bins[maximum_index+1]
        print bins[maximum_index+2]
        print bins[maximum_index-1]

#         sns.distplot(all_mrna[all_mrna<80],
        sns.distplot(all_mrna,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True,
#                      norm_hist = True,
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("mean mRNA number")
        plt.xlim(0,100)
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
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    # np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))) #standard deviation

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

    def xest_make_relative_delay_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    # np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_posterior_samples = my_posterior_samples[:10]
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
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
#                                     np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))) #standard deviation

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

    def xest_make_amplitude_plot(self):
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

        parameter_names = ['protein_degradation_rate']

        x_labels = dict()
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                            'extended_degradation_sweep.npy'))
#                                                               'extended_relative_sweeps_' + parameter_name + '.npy'))
            
            increase_indices = np.where(my_parameter_sweep_results[:,9,3] < 300)
 
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
#             my_sweep_parameters = my_posterior_samples[increase_indices]
            
            x_coord = -0.4
            y_coord = 1.1
            my_figure = plt.figure( figsize = (4.5, 1.5) )
            this_axis = my_figure.add_subplot(121)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,2], color ='teal', alpha = 0.02, zorder = 0)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('COV')
            plt.gca().set_rasterization_zorder(1)
            plt.axvline( np.log(2)/90, color = 'darkblue' )
            plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)
            plt.xlim(0,np.log(2)/15.)
#             this_axis.set_ylim(0,1)
#             this_axis.set_ylim(0,0.2)
            plt.title('stochastic')
#             this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(122)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,7], color = 'teal', alpha = 0.02, zorder = 0)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
#             this_axis.set_ylabel('Coherence')
            plt.title('deterministic')
            plt.gca().set_rasterization_zorder(1)
            plt.axvline( np.log(2)/90, color = 'darkblue' )
            plt.gca().text(x_coord, y_coord, 'B', transform=plt.gca().transAxes)
            plt.xlim(0,np.log(2)/15.)
#             this_axis.set_ylim(0,0.2)
#             this_axis.set_ylim(0,0.5)
#             this_axis.set_ylim(0,0.25)
            
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','amplitude_model_prediction_' + parameter_name + '.pdf'), dpi = 400)
 
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
#         my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'data',
#                                                           'narrowed_degradation_sweep.npy'))
        
        my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                                          'extended_degradation_sweep.npy'))
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
                                                              'extended_relative_sweeps_hill_coefficient.npy'))
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
                                                              'extended_relative_sweeps_time_delay.npy'))
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
                                 'output','extended_bifurcation_illustration.pdf'), dpi = 400)
 
    def xest_plot_bayes_factors_for_models(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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
            print 'these decrease samples are'
            number_of_decrease_samples = len(decrease_indices[0])
            print number_of_decrease_samples

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                        my_parameter_sweep_results[:,14,4] > 0.1)))

            increase_ratios[parameter_name] = len(increase_indices[0])/float(number_of_absolute_samples)
            print 'these increase samples are'
            number_of_increase_samples = len(increase_indices[0])
            print number_of_increase_samples
                
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
        plt.xlim(all_positions[0] - 0.5, all_positions[-1] + 0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=4)
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

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
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

    def xest_plot_stochastic_amplification(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]>0.05,
                                                    model_results[:,3]>0.3))))
#                                                     model_results[:,1]>0.05)))
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                    model_results[:,1]>0.05))))

        
        my_posterior_samples = prior_samples[accepted_indices]
        accepted_model_results = model_results[accepted_indices]
        
        this_parameter = my_posterior_samples[0]
        print('parameter is')
        print(this_parameter)
        number_of_trajectories = 3
        hes5_mRNA_trajectories, hes5_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_trajectories,
                                                                                        duration = 2000,
                                                         repression_threshold = this_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = this_parameter[1],
                                                         basal_transcription_rate = this_parameter[0],
                                                         transcription_delay = this_parameter[3],
                                                         initial_mRNA = 3,
                                                         initial_protein = this_parameter[2],
                                                         hill_coefficient = this_parameter[4],
                                                         equilibration_time = 0)
#
        deterministic_trajectory = hes5.generate_deterministic_trajectory(duration = 2000, 
                                                         repression_threshold = this_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = this_parameter[1],
                                                         basal_transcription_rate = this_parameter[0],
                                                         transcription_delay = this_parameter[3],
                                                         initial_mRNA = 3,
                                                         initial_protein = this_parameter[2],
#                                                                           repression_threshold = 31400,
#                                                          mRNA_degradation_rate = np.log(2)/30,
#                                                          protein_degradation_rate = np.log(2)/90,
#                                                          translation_rate = 29,
#                                                          basal_transcription_rate = 11,
#                                                          transcription_delay = 29,
#                                                          initial_mRNA = 3,
#                                                          initial_protein = 31400,
                                                         hill_coefficient = this_parameter[4],
                                                         for_negative_times = 'no_negative')[:-1]
        

        mean_hes5_protein_trajectory = np.mean(hes5_protein_trajectories[:,1:], axis = 1)
        mean_hes5_rna_trajectory = np.mean(hes5_mRNA_trajectories[:,1:], axis = 1)
        figuresize = (4,2.5)
        my_figure = plt.figure(figsize = figuresize)
        plt.plot( hes5_protein_trajectories[:,0],
                  hes5_protein_trajectories[:,1]/10000, color = 'black',
                  lw = 0.5, alpha = 0.2, label = 'stochastic' )
        for trajectory_index in range(2,number_of_trajectories+1):
#             plt.plot( hes5_mRNA_trajectories[:,0],
#                       hes5_mRNA_trajectories[:,trajectory_index]*1000., color = 'black',
#                       lw = 0.5, alpha = 0.1 )
            plt.plot( hes5_protein_trajectories[:,0],
                      hes5_protein_trajectories[:,trajectory_index]/10000, color = 'black',
                      lw = 0.5, alpha = 0.2 )
#         plt.plot( hes5_mRNA_trajectories[:,0],
#                   mean_hes5_rna_trajectory*1000., label = 'mRNA*1000', color = 'blue',
#                   lw = 0.5 )
        plt.plot( deterministic_trajectory[:,0], deterministic_trajectory[:,2]/10000,
                  lw = 0.5, label = 'deterministic' )
#         plt.plot( hes5_protein_trajectories[:,0],
#                   mean_hes5_protein_trajectory, label = 'Protein', color = 'blue', ls = '--',
#                   lw = 0.5, dashes = [1,1] )
        plt.xlabel('Time [min]')
        plt.ylabel('Hes5 expression/1e4')
        plt.xlim(0,2000)
#         plt.ylim(0,10)
#         plt.legend(bbox_to_anchor=(1.05, 1.1), loc = 'upper right')
        plt.legend(loc = 'upper right')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','stochastic_amplficiation.pdf'))
 
    def xest_deterministic_bifurcation(self):
        ##at this parameter point the system should oscillate
        protein_degradation = 0.03
        mrna_degradation = 0.03
        transcription_delay = 18.5
        basal_transcription_rate = 1.0
        translation_rate = 1.0
        repression_threshold = 100.0
        hill_coefficient = 5
        
        is_oscillatory = hes5.is_parameter_point_deterministically_oscillatory( repression_threshold = repression_threshold, 
                                                              hill_coefficient = hill_coefficient, 
                                                              mRNA_degradation_rate = mrna_degradation, 
                                                              protein_degradation_rate = protein_degradation, 
                                                              basal_transcription_rate = basal_transcription_rate,
                                                              translation_rate = translation_rate,
                                                              transcription_delay = transcription_delay)

        self.assert_(is_oscillatory)

        ## at this parameter point the system should not oscillate
        protein_degradation = np.log(2)/90.0
        mrna_degradation = np.log(2)/30.0
        transcription_delay = 29
        basal_transcription_rate = 1.0
        translation_rate = 320.0
        repression_threshold = 60000
        hill_coefficient = 5
        
        is_oscillatory = hes5.is_parameter_point_deterministically_oscillatory( repression_threshold = repression_threshold, 
                                                              hill_coefficient = hill_coefficient, 
                                                              mRNA_degradation_rate = mrna_degradation, 
                                                              protein_degradation_rate = protein_degradation, 
                                                              basal_transcription_rate = basal_transcription_rate,
                                                              translation_rate = translation_rate,
                                                              transcription_delay = transcription_delay)

        self.assert_(not is_oscillatory)

    def xest_stochastic_bifurcation(self):
        ##at this parameter point the system should oscillate
        protein_degradation = 0.03
        mrna_degradation = 0.03
        transcription_delay = 18.5
        basal_transcription_rate = 1.0
        translation_rate = 1.0
        repression_threshold = 100.0
        hill_coefficient = 5
        
        is_oscillatory = hes5.is_parameter_point_stochastically_oscillatory( repression_threshold = repression_threshold, 
                                                              hill_coefficient = hill_coefficient, 
                                                              mRNA_degradation_rate = mrna_degradation, 
                                                              protein_degradation_rate = protein_degradation, 
                                                              basal_transcription_rate = basal_transcription_rate,
                                                              translation_rate = translation_rate,
                                                              transcription_delay = transcription_delay )

        self.assert_(is_oscillatory)

        ## at this parameter point the system should not oscillate stochastically
        protein_degradation = np.log(2)/90.0
        mrna_degradation = np.log(2)/30.0
        transcription_delay = 34
        basal_transcription_rate = 0.64
        translation_rate = 17.32
        repression_threshold = 88288.6
        hill_coefficient = 5.59
        
        is_oscillatory = hes5.is_parameter_point_stochastically_oscillatory( repression_threshold = repression_threshold, 
                                                              hill_coefficient = hill_coefficient, 
                                                              mRNA_degradation_rate = mrna_degradation, 
                                                              protein_degradation_rate = protein_degradation, 
                                                              basal_transcription_rate = basal_transcription_rate,
                                                              translation_rate = translation_rate,
                                                              transcription_delay = transcription_delay)

        self.assert_(not is_oscillatory)

    def xest_investigate_lna_prediction_at_low_degradation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]<0.03))))  #standard deviation
        
        posterior_samples = prior_samples[accepted_indices]

        sample = posterior_samples[0]

        #First: run the model for 100 minutes
#         my_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
#                                                          repression_threshold = 100,
#                                                          mRNA_degradation_rate = 0.03,
#                                                          protein_degradation_rate = 0.03,
#                                                          transcription_delay = 19,
#                                                          initial_mRNA = 3,
#                                                          initial_protein = 100)
# #                                                          integrator = 'PyDDE',
# #                                                          for_negative_times = 'no_negative' )
        print 'experimental values for mrna and protein degradation are'
        print np.log(2)/30
        print np.log(2)/90
        theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( repression_threshold = sample[2], 
                                                                 hill_coefficient = sample[4], 
#                                                                  mRNA_degradation_rate = np.log(2)/30, 
#                                                                  protein_degradation_rate = np.log(2)/90, 
                                                                 mRNA_degradation_rate = 0.001, 
                                                                 protein_degradation_rate = 0.001, 
                                                                 basal_transcription_rate = sample[0],
                                                                 translation_rate = sample[1],
                                                                 transcription_delay = sample[3] )
        
        coherence, period = hes5.calculate_coherence_and_period_of_power_spectrum( theoretical_power_spectrum )
        print 'theoretical coherence and period are'
        print coherence
        print period

        full_parameter_point = np.array([sample[0],
                                sample[1],
                                sample[2],
                                sample[3],
                                sample[4],
                                0.001,
                                0.001])
#                                 np.log(2)/30,
#                                 np.log(2)/90])

        real_power_spectrum = hes5.calculate_power_spectrum_at_parameter_point( full_parameter_point )

        #Second, plot the model

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(theoretical_power_spectrum[:,0], 
                 theoretical_power_spectrum[:,1])
        plt.plot(real_power_spectrum[:,0], 
                 real_power_spectrum[:,1])
        plt.xlabel('Frequency [1/min]')
        plt.ylabel('Power')
        plt.xlim(0,0.01)
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','weird_power_spectrum.pdf'))

    def test_plot_oscillation_probability_data(self):
        option = 'deterministic'

        X = np.load(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_protein_degradation_values_' + option + '.npy'))
        Y = np.load(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_mrna_degradation_values_' + option + '.npy'))
        expected_coherence = np.load(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_values_' + option + '.npy'))
        if option == 'stochastic':
            oscillation_probability = np.load(os.path.join(os.path.dirname(__file__),
                                           'output','oscillation_probability_values_' + option + '.npy'))
        else: 
            oscillation_probability = expected_coherence
        
        plt.figure(figsize = (4.5,2.5))
#         plt.contourf(X,Y,oscillation_probability, 100, lw=0, rasterized = True)
#         plt.pcolormesh(X,Y,oscillation_probability, lw = 0, rasterized = True, shading = 'gouraud')
#         plt.pcolormesh(X,Y,oscillation_probability, lw = 0, rasterized = True)
        plt.pcolor(X,Y,oscillation_probability)
        plt.xlabel("Protein degradation [1/min]")
        plt.ylabel("mRNA degradation [1/min]")
        this_colorbar = plt.colorbar()
        this_colorbar.ax.set_ylabel('Probability of oscillation')
        plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_probability_' + option + '.pdf'))

        plt.figure(figsize = (4.5,2.5))
#         plt.contourf(X,Y,expected_coherence, 100, lw=0, rasterized = True)
#         plt.pcolormesh(X,Y,expected_coherence, lw = 0, rasterized = True, shading = 'gouraud')
#         plt.pcolormesh(X,Y,expected_coherence, lw = 0, rasterized = True)
        plt.pcolor(X,Y,expected_coherence)
        plt.xlabel("Protein degradation [1/min]")
        plt.ylabel("mRNA degradation [1/min]")
        this_colorbar = plt.colorbar()
        this_colorbar.ax.set_ylabel('Expected coherence')
        plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_' + option + '.pdf'))
 
    def xest_make_oscillation_probability_plot(self):
        option = 'stochastic'
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                   model_results[:,1]>0.05)))  #standard deviation
        
        posterior_samples = prior_samples[accepted_indices]

        resolution_per_direction = 100
        mRNA_degradation_values = np.linspace(0.001,np.log(2)/20, resolution_per_direction)
        protein_degradation_values = np.linspace(0.001,np.log(2)/20, resolution_per_direction)
        oscillation_probability = np.zeros((len(mRNA_degradation_values),len(protein_degradation_values)))
        expected_coherence = np.zeros((len(mRNA_degradation_values),len(protein_degradation_values)))
        for protein_degradation_index, protein_degradation in enumerate(protein_degradation_values):
            for mRNA_degradation_index, mRNA_degradation in enumerate(mRNA_degradation_values):
                total_number_of_samples = 0.0
                oscillating_samples = 0.0
                coherence_sum = 0.0
                for sample in posterior_samples:
                    if option == 'deterministic':
                        this_sample_oscillates = hes5.is_parameter_point_deterministically_oscillatory( repression_threshold = sample[2], 
                                                                                 hill_coefficient = sample[4], 
                                                                                 mRNA_degradation_rate = mRNA_degradation, 
                                                                                 protein_degradation_rate = protein_degradation, 
                                                                                 basal_transcription_rate = sample[0],
                                                                                 translation_rate = sample[1],
                                                                                 transcription_delay = sample[3])
                        
                        coherence = this_sample_oscillates

                    elif option == 'stochastic':
                        power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( repression_threshold = sample[2], 
                                                                                 hill_coefficient = sample[4], 
                                                                                 mRNA_degradation_rate = mRNA_degradation, 
                                                                                 protein_degradation_rate = protein_degradation, 
                                                                                 basal_transcription_rate = sample[0],
                                                                                 translation_rate = sample[1],
                                                                                 transcription_delay = sample[3])
                    
                        coherence, period = hes5.calculate_coherence_and_period_of_power_spectrum( power_spectrum )

                        max_index = np.argmax(power_spectrum[:,1])
                        
                        if max_index > 0:
                            this_sample_oscillates = True
                        else:
                            this_sample_oscillates = False
                            

                    else: 
                        raise ValueError('option not recognised')

                    if this_sample_oscillates: 
                        oscillating_samples +=1
                    coherence_sum += coherence
                    total_number_of_samples += 1
                probability_to_oscillate = oscillating_samples/total_number_of_samples
                oscillation_probability[protein_degradation_index, mRNA_degradation_index] = probability_to_oscillate
                expected_coherence[protein_degradation_index, mRNA_degradation_index] = coherence_sum/total_number_of_samples
        
        X, Y = np.meshgrid(protein_degradation_values, mRNA_degradation_values)

        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_protein_degradation_values_' + option + '.npy'), X)
        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_mrna_degradation_values_' + option + '.npy'), Y)
        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_values_' + option + '.npy'), expected_coherence)
        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_probability_values_' + option + '.npy'), oscillation_probability)
        
        plt.figure(figsize = (4.5,2.5))
        plt.contourf(X,Y,oscillation_probability, 100, lw=0, rasterized = True)
#         plt.pcolormesh(X,Y,oscillation_probability, lw = 0, rasterized = True, shading = 'gouraud')
        plt.xlabel("Protein degradation [1/min]")
        plt.ylabel("mRNA degradation [1/min]")
        this_colorbar = plt.colorbar()
        this_colorbar.ax.set_ylabel('Oscillation probability')
        plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_probability_' + option + '.pdf'))

        plt.figure(figsize = (4.5,2.5))
        plt.contourf(X,Y,expected_coherence, 100, lw=0, rasterized = True)
#         plt.pcolormesh(X,Y,expected_coherence, lw = 0, rasterized = True, shading = 'gouraud')
        plt.xlabel("Protein degradation [1/min]")
        plt.ylabel("mRNA degradation [1/min]")
        this_colorbar = plt.colorbar()
        this_colorbar.ax.set_ylabel('Oscillation coherence')
        plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillation_coherence_' + option + '.pdf'))
        
    def xest_plot_lna_std_vs_model_results(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                   model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]

        theoretical_standard_deviation = np.zeros(len(posterior_samples))
        for sample_index, sample in enumerate(posterior_samples):
            this_standard_deviation = hes5.calculate_approximate_standard_deviation_at_parameter_point(basal_transcription_rate = sample[0],
                                                                translation_rate = sample[1],
                                                                repression_threshold = sample[2],
                                                                transcription_delay = sample[3],
                                                                mRNA_degradation_rate = np.log(2)/30,
                                                                protein_degradation_rate = np.log(2)/90,
                                                                hill_coefficient = sample[4]
                                                                )

            steady_state_mrna, steady_state_protein = hes5.calculate_steady_state_of_ode(basal_transcription_rate = sample[0],
                                                                translation_rate = sample[1],
                                                                repression_threshold = sample[2],
                                                                mRNA_degradation_rate = np.log(2)/30,
                                                                protein_degradation_rate = np.log(2)/90,
                                                                hill_coefficient = sample[4]
                                                                )

            relative_standard_deviation = this_standard_deviation/steady_state_protein

            theoretical_standard_deviation[ sample_index ] = relative_standard_deviation 

        error_ratios = theoretical_standard_deviation / posterior_results[:,1]
        relative_errors = np.abs(error_ratios - 1)
        number_of_poor_samples = np.sum(relative_errors>0.1)
        print 'ratio of poor approximations is'
        print number_of_poor_samples/float(len(posterior_samples))
        print 'total  number is'
        print number_of_poor_samples
        plt.figure(figsize = (4.5,2.5))
        plt.scatter(theoretical_standard_deviation, posterior_results[:,1], s = 1)
        plt.plot([0.0,0.25],1.1*np.array([0.0,0.25]))
        plt.plot([0.0,0.25],0.9*np.array([0.0,0.25]))
        plt.xlabel("LNA")
        plt.ylabel("CLE")
        plt.title("Relative standard deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','LNA_validation.pdf'))

        # now, we need to plot where the outliers are:
        outlier_mask = relative_errors > 0.1
        outlier_samples = posterior_samples[outlier_mask]
        outlier_results = posterior_results[outlier_mask]

        print 'outlier coherences are'
        print outlier_results[:,3]
        outlier_samples[:,2]/=10000
        print 'minimal outlier coherence is'
        print np.min(outlier_results[:,3])
        print 'posterior samples with coherence above 0.44'
        print np.sum(posterior_results[:,3]>0.5)

        data_frame = pd.DataFrame( data = outlier_samples,
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
                                    'output','LNA_outliers.pdf'))

        #what happens in the cases when the CLE has higher std than the LNA?
        outlier_mask = error_ratios<0.9
        outlier_samples = posterior_samples[outlier_mask]
        outlier_results = posterior_results[outlier_mask]

        print 'outlier coherences are'
        print outlier_results[:,3]
        outlier_samples[:,2]/=10000
        print 'minimal outlier coherence is'
        print np.min(outlier_results[:,3])
        print 'posterior samples with coherence above 0.44'
        print np.sum(posterior_results[:,3]>0.5)

        data_frame = pd.DataFrame( data = outlier_samples,
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
                                    'output','LNA_other_outliers.pdf'))

    def xest_calculate_variance(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                   model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]

        # first: calculate theoretical standard deviation
        # then: compare to model result
        sample = posterior_samples[5]
        sample_result = posterior_results[5]
        
        mRNA_degradation_rate = np.log(2)/30
        protein_degradation_rate = np.log(2)/90
        basal_transcription_rate = sample[0]
        translation_rate = sample[1]
        repression_threshold = sample[2]
        transcription_delay = sample[3]
        hill_coefficient = sample[4]

        actual_frequencies = np.linspace(0,0.01,1000)
        pi_frequencies = actual_frequencies*2*np.pi
        steady_state_mrna, steady_state_protein = hes5.calculate_steady_state_of_ode( repression_threshold = float(repression_threshold),
                                        hill_coefficient = hill_coefficient,
                                        mRNA_degradation_rate = mRNA_degradation_rate,
                                        protein_degradation_rate = protein_degradation_rate, 
                                        basal_transcription_rate = basal_transcription_rate,
                                        translation_rate = translation_rate)
    
        steady_state_hill_function_value = 1.0/(1.0 + np.power( steady_state_protein/float(repression_threshold),
                                                                hill_coefficient ))
        
        steady_state_hill_derivative = -hill_coefficient*np.power(steady_state_protein/float(repression_threshold), 
                                                                hill_coefficient - 1)/(repression_threshold*
                                        np.power(1.0+np.power(steady_state_protein/float(repression_threshold),
                                                            hill_coefficient),2))
    
    #     steady_state_hill_derivative = -hill_coefficient/float(repression_threshold)*np.power(
    #                                      1.0 + steady_state_protein/float(repression_threshold),
    #                                                     hill_coefficient)
    
        power_spectrum_values = ( translation_rate*translation_rate*
                           ( basal_transcription_rate * steady_state_hill_function_value +
                             mRNA_degradation_rate*steady_state_mrna) 
                          +
                           ( np.power(pi_frequencies,2) + mRNA_degradation_rate*mRNA_degradation_rate)*
                             ( translation_rate*steady_state_mrna + protein_degradation_rate*steady_state_protein)
                          )/(np.power(- np.power(pi_frequencies,2) +
                                      protein_degradation_rate*mRNA_degradation_rate
                                      - basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                      np.cos(pi_frequencies*transcription_delay),2) 
                             +
                             np.power((protein_degradation_rate+mRNA_degradation_rate)*
                                      pi_frequencies +
                                      basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                      np.sin(pi_frequencies*transcription_delay), 2)
                             )
                             
        power_spectrum = np.vstack((pi_frequencies, power_spectrum_values)).transpose()
        integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])

        power_spectrum_intercept = integral
        variance = 0.5/np.pi*power_spectrum_intercept

        relative_standard_deviation = np.sqrt(variance)/steady_state_protein
        
        print 'variance over mean over real value'
        print (power_spectrum_intercept/steady_state_protein)/sample_result[1]
        print 'std over mean'
        print (np.sqrt(power_spectrum_intercept)/steady_state_protein)/sample_result[1]

