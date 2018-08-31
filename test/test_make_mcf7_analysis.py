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

class TestMakeMCF7Analysis(unittest.TestCase):
                                 
    def xest_make_abc_samples(self):
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.01,60),
                        'translation_rate' : (0.01,1),
                        'repression_threshold' : (0,7000),
                        'time_delay' : (5,80),
                        'hill_coefficient' : (2,10),
                        'protein_degradation_rate' : ( np.log(2)/(3.85*60), np.log(2)/(3.85*60) ),
                        'mRNA_degradation_rate' : ( np.log(2)/41.0, np.log(2)/41.0) }

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_MCF7_altered',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 7))

        # plot distribution of accepted parameter samples
#         pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
#         pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','pairplot_mcf7_abc_' +  str(total_number_of_samples) + '_'
#                                       + str(acceptance_ratio) + '.pdf'))

    def xest_plot_posterior_distributions(self):
        
        option = 'coherence'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_MCF7_altered')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        upper_delay_bound = 80
        upper_hill_bound = 10
#         upper_delay_bound = 40
#         upper_hill_bound = 6
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                       model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000,
                                        np.logical_and(model_results[:,4]>40,
                                        np.logical_and(model_results[:,4]<60, #mrna number
                                                       model_results[:,1]>0.05)))))  #standard deviation
#                                                        model_results[:,1]>0.05)))  #standard deviation
        elif option == 'prior':
            accepted_indices = range(len(prior_samples))
        elif option == 'coherence_and_hill':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000,
                                        np.logical_and(model_results[:,4]>40,
                                        np.logical_and(model_results[:,4]<60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.3,
                                                       prior_samples[:,4]<5)))))))  #standard deviation
        elif option == 'coherence_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000,
                                        np.logical_and(model_results[:,4]>40,
                                        np.logical_and(model_results[:,4]<60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.3,
                                                       model_results[:,2]<6*60)))))))  #standard deviation
        elif option == 'coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000,
                                        np.logical_and(model_results[:,4]>40,
                                        np.logical_and(model_results[:,4]<60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05, 
                                                       model_results[:,3]>0.3))))))  #standard deviation
        elif option == 'amplitude_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000, #protein_number
#                                         np.logical_and(model_results[:,4]>40,
#                                         np.logical_and(model_results[:,4]>60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,3]>0.15)))) #standard deviation
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
             accepted_indices = np.where(np.logical_and(model_results[:,5]>2000, #protein number
                                         np.logical_and(model_results[:,5]<4000, #protein_number
                                         np.logical_and(model_results[:,9]>40,
                                         np.logical_and(model_results[:,9]<60, #mrna number
                                                        model_results[:,6]>0.05)))))  #standard deviation
        else:
            ValueError('could not identify posterior option')
#       
        my_posterior_samples = prior_samples[accepted_indices]
        print my_posterior_samples[:,1]
        
        print('minimal hill')
        print(np.min(my_posterior_samples[:,4]))
        print('miximal time delay')
        print(np.min(my_posterior_samples[:,3]))
        print('maximal time delay')
        print(np.max(my_posterior_samples[:,3]))
        
#         pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
#         pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','pairplot_extended_abc_' + option + '.pdf'))

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        my_posterior_samples[:,2]/=1000

        data_frame = pd.DataFrame( data = my_posterior_samples[:,:5],
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
        plt.gca().set_ylim(0,6)
#         plt.gca().set_ylim(0,1)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(-2,0,20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-2,0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,4.0)
#         plt.gca().set_ylim(0,1.0)
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
        plt.gca().set_ylim(0,0.3)
        plt.gca().set_xlim(0,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,upper_delay_bound,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,upper_delay_bound)
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
        plt.gca().set_xlim(2,upper_hill_bound)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_mcf7_' + option + '.pdf'))

    def xest_plot_mcf7_period_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_MCF7_altered')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation
#                                                    model_results[:,1]>0.05))) #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,
#                                                    model_results[:,3]>0.15)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]/60
#         all_periods = my_model_results[:,2]
        print('mean is')
#         print(np.mean(all_periods[all_periods<10]))
        print(np.mean(all_periods))
        print('median is')
#         print(np.median(all_periods[all_periods<10]))
        print(np.median(all_periods))
        print('minimum is')
        print(np.min(all_periods))
#         period_histogram, bins = np.histogram(all_periods[all_periods<10], bins = 400) 
        period_histogram, bins = np.histogram(all_periods, bins = 400) 
        maximum_index = np.argmax(period_histogram)
        print('max bin is')
# # # #         print bins[maximum_index]
# # #         print bins[maximum_index+1]
# #         print bins[maximum_index+2]
#         print bins[maximum_index-1]
# #         sns.distplot(all_periods[np.logical_and(all_periods<1000,
#                                                 all_periods>100)],
        sns.distplot(all_periods,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
        plt.xlim(4,12)
        plt.ylim(0,0.2)
#         plt.ylim(0,0.0003)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mcf7_period_distribution.pdf'))
 
    def xest_plot_mcf7_coherence_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_MCF7_altered')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_coherences = my_model_results[:,3]
        print('largest coherence is')
        print(np.max(all_coherences))
        sns.distplot(all_coherences,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled coherence")
        plt.xlim(0.2,)
        plt.ylim(0,5)
#         plt.xlim(0,20)
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mcf7_coherence_distribution.pdf'))
 
    def xest_plot_mrna_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_mrna_values = my_model_results[:,4]
        sns.distplot(all_mrna_values,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled mrna")
#         plt.xlim(0,20)
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mcf7_mrna_distribution.pdf'))
 
    def xest_investigate_individual_mcf7_trace(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_MCF7_altered')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000, #protein_number
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
#                                     np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.3,
#                                                    prior_samples[:,4]<5)))))))  #standard deviation
                                                    model_results[:,1]>0.05))))))  #standard deviation
#                                                     model_results[:,1]>0.05)))))  #standard deviation
 
        my_posterior_samples = prior_samples[accepted_indices]
        my_results = model_results[accepted_indices]
        index = 0
        my_sample = my_posterior_samples[index,:]
        this_result = my_results[index,:]

        my_trajectory = hes5.generate_stochastic_trajectory( duration = 3500,
                                                             basal_transcription_rate = my_sample[0],
                                                             translation_rate = my_sample[1],
                                                             repression_threshold = my_sample[2],
                                                             transcription_delay = my_sample[3],
                                                             hill_coefficient = my_sample[4],
                                                             mRNA_degradation_rate = my_sample[5],
                                                             protein_degradation_rate = my_sample[6],
                                                             initial_mRNA = 3,
                                                             initial_protein = my_sample[2],
                                                             equilibration_time = 2000 )

        #Second, plot the model

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.title('Coherence: ' + '{:.2f}'.format(this_result[3]) + 
                  r', $\alpha_m =$ ' + '{:.2f}'.format(my_sample[0]) +
                  r', $n =$ ' + '{:.2f}'.format(my_sample[4]) +
                  '\n' + r'$\alpha_p =$ ' + '{:.2f}'.format(my_sample[1]) + 
                  r', $p_0 = $ ' + '{:.2f}'.format(my_sample[2]) + 
                  r', $\tau = $ ' + '{:.2f}'.format(my_sample[3]) +
                  ', Period: ' + '{:.2f}'.format(this_result[2]/60),
                  fontsize = 5)
        plt.plot(my_trajectory[:,0]/60, 
                 my_trajectory[:,1]*10, label = 'mRNA*10', color = 'black')
        plt.plot(my_trajectory[:,0]/60,
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time [h]')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','stochastic_mcf7_trajectory.pdf'))
        
    def test_plot_deterministic_trace_at_inferred_parameter(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
#                                    'sampling_results_MCF7')
                                   'sampling_results_MCF7_altered')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        

        accepted_indices = np.where(np.logical_and(model_results[:,5]>2000, #protein number
                                    np.logical_and(model_results[:,5]<4000, #protein_number
                                    np.logical_and(model_results[:,9]>40,
                                    np.logical_and(model_results[:,9]<60, #mrna number
                                                   model_results[:,6]>0.05)))))  #standard deviation
#                                     np.logical_and(model_results[:,7]<6*60,
#                                                    model_results[:,6]>0.05))))))  #standard deviation
 
        my_posterior_samples = prior_samples[accepted_indices]
        my_results = model_results[accepted_indices]
        index = 0
        my_sample = my_posterior_samples[index,:]
#         my_sample[4] = 100
        this_result = my_results[index,:]

        my_trajectory = hes5.generate_deterministic_trajectory( duration = 2500,
                                                             basal_transcription_rate = my_sample[0],
                                                             translation_rate = my_sample[1],
                                                             repression_threshold = my_sample[2],
                                                             transcription_delay = my_sample[3],
                                                             hill_coefficient = my_sample[4],
                                                             mRNA_degradation_rate = my_sample[5],
                                                             protein_degradation_rate = my_sample[6],
                                                             initial_mRNA = 3,
                                                             initial_protein = my_sample[2],
                                                             integrator = 'PyDDE' )

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.title('Coherence: ' + '{:.2f}'.format(this_result[3]) + 
                  r', $\alpha_m =$ ' + '{:.2f}'.format(my_sample[0]) +
                  r', $n =$ ' + '{:.2f}'.format(my_sample[4]) +
                  '\n' + r'$\alpha_p =$ ' + '{:.2f}'.format(my_sample[1]) + 
                  r', $p_0 = $ ' + '{:.2f}'.format(my_sample[2]) + 
                  r', $\tau = $ ' + '{:.2f}'.format(my_sample[3]) +
                  ', Period: ' + '{:.2f}'.format(this_result[7]/60),
                  fontsize = 5)
        plt.plot(my_trajectory[:,0]/60, 
                 my_trajectory[:,1]*10, label = 'mRNA*10', color = 'black')
        plt.plot(my_trajectory[:,0]/60,
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time [h]')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','deterministic_mcf7_trajectory.pdf'))
 
    def xest_plot_cle_vs_gillespie_results(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_MCF7_altered')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation
#                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        print('number of accepted samples is')
        print(len(posterior_samples[:,0]))
        
#         posterior_samples = posterior_samples[:10,:]
#         posterior_results = posterior_results[:10,:]
        gillespie_results = hes5.calculate_gillespie_summary_statistics_at_parameters(posterior_samples)
#         gillespie_results = hes5.calculate_summary_statistics_at_parameters(posterior_samples, model = 'gillespie_sequential')

        saving_path = os.path.join(os.path.dirname(__file__),'output','new_gillespie_posterior_results')
        
        np.save(saving_path + '.npy', gillespie_results)
#         gillespie_results = np.load(saving_path + '.npy')

        gillespie_standard_deviation = gillespie_results[:,1]

        error_ratios = posterior_results[:,1]/gillespie_standard_deviation
        
        print('number of outlier samples')
        print(np.sum(error_ratios>1.1))

        plt.figure(figsize = (4.5,2.5))
        plt.scatter(gillespie_standard_deviation, posterior_results[:,1], s = 0.5)
        plt.plot([0.0,0.07],1.1*np.array([0.0,0.07]), lw = 1, color = 'grey')
        plt.plot([0.0,0.07],0.9*np.array([0.0,0.07]), lw = 1, color = 'grey')
        plt.xlabel("Gillespie")
        plt.ylabel("CLE")
        plt.title("Relative standard deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','mcf7_CLE_validation.pdf'))

    def xest_plot_more_accurate_cle_vs_gillespie_results(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_MCF7_altered')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation
#                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        print('number of accepted samples is')
        print(len(posterior_samples[:,0]))
        
#         posterior_samples = posterior_samples[:10,:]
#         posterior_results = posterior_results[:10,:]
#         gillespie_results = hes5.calculate_gillespie_summary_statistics_at_parameters(posterior_samples)

        saving_path = os.path.join(os.path.dirname(__file__),'output','gillespie_posterior_results')
        
#         np.save(saving_path + '.npy', gillespie_results)
        gillespie_results = np.load(saving_path + '.npy')

        more_accurate_cle_results = hes5.calculate_summary_statistics_at_parameters(posterior_samples)
        gillespie_standard_deviation = gillespie_results[:,1]

        error_ratios = posterior_results[:,1]/gillespie_standard_deviation
        
        print('number of outlier samples')
        print(np.sum(error_ratios>1.1))

        plt.figure(figsize = (4.5,2.5))
        plt.scatter(posterior_results[:,1], more_accurate_cle_results[:,1], s = 0.5)
        plt.plot([0.0,0.07],1.1*np.array([0.0,0.07]), lw = 1, color = 'grey')
        plt.plot([0.0,0.07],0.9*np.array([0.0,0.07]), lw = 1, color = 'grey')
        plt.xlabel("Gillespie")
        plt.ylabel("CLE")
        plt.title("Relative standard deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','mcf7_CLE_validation.pdf'))

    def xest_plot_lna_std_vs_model_results(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_MCF7_altered')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<4000,
                                    np.logical_and(model_results[:,4]>40,
                                    np.logical_and(model_results[:,4]<60, #mrna number
                                                   model_results[:,1]>0.05)))))  #standard deviation
#                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]

        theoretical_standard_deviation = np.zeros(len(posterior_samples))
        for sample_index, sample in enumerate(posterior_samples):
            this_standard_deviation = hes5.calculate_approximate_standard_deviation_at_parameter_point(basal_transcription_rate = sample[0],
                                                                translation_rate = sample[1],
                                                                repression_threshold = sample[2],
                                                                transcription_delay = sample[3],
                                                                mRNA_degradation_rate = np.log(2)/41,
                                                                protein_degradation_rate = np.log(2)/(3.85*60),
                                                                hill_coefficient = sample[4]
                                                                )

            steady_state_mrna, steady_state_protein = hes5.calculate_steady_state_of_ode(basal_transcription_rate = sample[0],
                                                                translation_rate = sample[1],
                                                                repression_threshold = sample[2],
                                                                mRNA_degradation_rate = np.log(2)/41,
                                                                protein_degradation_rate = np.log(2)/(3.85*60),
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
        plt.scatter(theoretical_standard_deviation, posterior_results[:,1], s = 0.5)
        plt.plot([0.0,0.25],1.1*np.array([0.0,0.25]), lw = 1, color = 'grey')
        plt.plot([0.0,0.25],0.9*np.array([0.0,0.25]), lw = 1, color = 'grey')
        plt.xlabel("LNA")
        plt.ylabel("CLE")
        plt.title("Relative standard deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','mcf7_LNA_validation.pdf'))

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

        data_frame = pd.DataFrame( data = outlier_samples[:,:5],
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
                                    'output','mcf7_LNA_outliers.pdf'))

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

        data_frame = pd.DataFrame( data = outlier_samples[:,:5],
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
                                    'output','mcf_7_LNA_other_outliers.pdf'))

