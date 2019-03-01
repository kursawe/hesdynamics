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
import scipy.signal
import pandas as pd
import seaborn as sns

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestNSQuiescence(unittest.TestCase):
                                 
    def xest_a_make_abc_samples(self):
        print('making abc samples')
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (0.01,40),
                        'repression_threshold' : (0,20000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : (np.log(2)/20,np.log(2)/20),
                        'mRNA_degradation_rate' : (np.log(2)/20,np.log(2)/20)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_quiescense',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_extended_abc_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))

    def xest_plot_posterior_distributions(self):
        
        option = 'mean_and_mrna'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'mean_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000, #protein_number
                                                       model_results[:,2]<300)))  #standard deviation
        elif option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
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
#                                          np.logical_and(model_results[:,6]<0.15,  #standard deviation
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

#         my_posterior_samples[:,2]/=1000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation rate',
                                             'protein degradation rate'])

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
        translation_rate_bins = np.linspace(-1,np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-1,np.log10(40))
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
#         plt.gca().set_ylim(0,0.22)
#         plt.gca().set_xlim(0,2)
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
                                    'output','inference_for_mari_' + option + '.pdf'))

    def xest_plot_amplitude_distribution_for_paper(self):
        option = 'mean_and_mrna'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
        elif option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
        elif option == 'mean_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000, #protein_number
                                                       model_results[:,2]<300)))  #standard deviation
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
        print('so many posterior samples')
        print(len(my_posterior_samples))
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
        all_standard_deviations = my_model_results[:,2]
        print('maximal standard_deviation is')
        print(np.max(all_standard_deviations))
        print('number of samples above 0.15')
#         print(np.sum(all_standard_deviations>0.15))
        sns.distplot(all_standard_deviations,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    bins = np.linspace(0,400,20),
                     hist_kws = {'edgecolor' : 'black'},
                     )
#                      bins = 20)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
#         plt.xlabel("Standard deviation/mean HES5")
        plt.xlabel('periods')
        plt.xlim(0,600)
#         plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mari_standard_deviation_' + option + '.pdf'))
 
    def xest_plot_mrna_distribution_for_mari(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                    np.logical_and(model_results[:,0]<12000,
                                                   model_results[:,4]<50))) #protein_number

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

#         my_posterior_samples[:,2] /= 10000

        my_figure = plt.figure(figsize= (4.5,2.5))

        all_mrna = my_model_results[:,4]
        print('minimum and maximum are')
        print(np.min(all_mrna))
        print(np.max(all_mrna))
        print('so many samples above 100')
        print(np.sum(all_mrna>100))
        mrna_histogram, bins = np.histogram(all_mrna, bins = 400) 
        maximum_index = np.argmax(mrna_histogram)
        print('max bin is')
        print(bins[maximum_index])
        print(bins[maximum_index+1])
        print(bins[maximum_index+2])
        print(bins[maximum_index-1])

#         sns.distplot(all_mrna[all_mrna<80],
        sns.distplot(all_mrna,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True,
#                      norm_hist = True,
                     bins = 20)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("mean mRNA number")
        plt.xlim(0,100)
#         plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mrna_distribution_for_mari.pdf'))

    def test_make_relative_parameter_variation(self):
#         number_of_parameter_points = 20
#         number_of_trajectories = 200
        number_of_parameter_points = 3
        number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                    np.logical_and(model_results[:,0]<12000,
                                                   model_results[:,4]<50))) #protein_number

        my_posterior_samples = prior_samples[accepted_indices][:10]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','maris_relative_sweeps_' + parameter_name + '.npy'),
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
        print('base model accepted that many indices')
        print(number_of_absolute_samples)
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
            print('investigating ' + parameter_name)
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'narrowed_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print('these accepted base samples are')
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                    my_parameter_sweep_results[:,9,4] < 0.1))[0])
            print(number_of_absolute_samples)
            
            decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                        my_parameter_sweep_results[:,4,4] > 0.1)))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)
            print('these decrease samples are')
            number_of_decrease_samples = len(decrease_indices[0])
            print(number_of_decrease_samples)

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                        my_parameter_sweep_results[:,14,4] > 0.1)))

            increase_ratios[parameter_name] = len(increase_indices[0])/float(number_of_absolute_samples)
            print('these increase samples are')
            number_of_increase_samples = len(increase_indices[0])
            print(number_of_increase_samples)
                
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
            print('investigating ' + parameter_name)
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'narrowed_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print('these accepted base samples are')
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                    my_parameter_sweep_results[:,9,4] < 0.1))[0])
            print(number_of_absolute_samples)
            
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
            print(decrease_parameters_before)
            print(increase_parameters_before)
            
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
            
            print('these are the increase parameters after')
            print(increase_parameters_after)
            parameter_index = parameter_name_to_index_lookup[parameter_name]
            
            try:
                print('hello1')
                reference_decrease_parameters = decrease_parameters_after[:,parameter_index]
                print(reference_decrease_parameters)
                decreased_parameters = reference_decrease_parameters*0.5
                print('hello2')
                decrease_parameters_after[:,parameter_index] = decreased_parameters
                print('hello3')
                decrease_spectra_before = hes5.calculate_power_spectra_at_parameter_points(decrease_parameters_before)
                print('hello4')
                decrease_spectra_after = hes5.calculate_power_spectra_at_parameter_points(decrease_parameters_after)
                print('hello5')
            except Exception as e: 
                print(repr(e))
                decrease_spectra_before = np.array([[0,0],[0,0]])
                decrease_spectra_after = np.array([[0,0],[0,0]])

            try:   
                print('hello1')
                reference_increase_parameters = increase_parameters_after[:,parameter_index]
                print(reference_increase_parameters)
                increased_parameters = reference_increase_parameters*1.5
                print('hello2')
                increase_parameters_after[:,parameter_index] = increased_parameters
                print('hello3')
                increase_spectra_before = hes5.calculate_power_spectra_at_parameter_points(increase_parameters_before)
                print('hello4')
                increase_spectra_after = hes5.calculate_power_spectra_at_parameter_points(increase_parameters_after)
                print('hello5')
            except Exception as e: 
                print(repr(e))
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
        print('experimental values for mrna and protein degradation are')
        print(np.log(2)/30)
        print(np.log(2)/90)
        theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( repression_threshold = sample[2], 
                                                                 hill_coefficient = sample[4], 
#                                                                  mRNA_degradation_rate = np.log(2)/30, 
#                                                                  protein_degradation_rate = np.log(2)/90, 
                                                                 mRNA_degradation_rate = 0.01, 
                                                                 protein_degradation_rate = 0.01, 
                                                                 basal_transcription_rate = sample[0],
                                                                 translation_rate = sample[1],
                                                                 transcription_delay = sample[3] )
        
        coherence, period = hes5.calculate_coherence_and_period_of_power_spectrum( theoretical_power_spectrum )
        print('theoretical coherence and period are')
        print(coherence)
        print(period)

        full_parameter_point = np.array([sample[0],
                                sample[1],
                                sample[2],
                                sample[3],
                                sample[4],
                                0.01,
                                0.01])
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

    def xest_plot_oscillation_probability_data(self):
#         option = 'deterministic'
        option = 'deterministic'

        X = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_protein_degradation_values_' + option + '.npy'))
        Y = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_mrna_degradation_values_' + option + '.npy'))
        expected_coherence = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_values_' + option + '.npy'))
        if option == 'stochastic':
            oscillation_probability = np.load(os.path.join(os.path.dirname(__file__),
                                           'data','oscillation_probability_values_' + option + '.npy'))
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
                                       'output','oscillation_probability_' + option + '.png'), dpi = 600)

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
                                       'output','oscillation_coherence_' + option + '.png'), dpi = 600)
 
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
                                       'output','oscillation_probability_' + option + '.png'))

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
                                       'output','oscillation_coherence_' + option + '.png'))
        
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
        print('ratio of poor approximations is')
        print(number_of_poor_samples/float(len(posterior_samples)))
        print('total  number is')
        print(number_of_poor_samples)
        plt.figure(figsize = (4.5,2.5))
        plt.scatter(theoretical_standard_deviation, posterior_results[:,1], s = 0.5)
        plt.plot([0.0,0.25],1.1*np.array([0.0,0.25]), lw = 1, color = 'grey')
        plt.plot([0.0,0.25],0.9*np.array([0.0,0.25]), lw = 1, color = 'grey')
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

        print('outlier coherences are')
        print(outlier_results[:,3])
        outlier_samples[:,2]/=10000
        print('minimal outlier coherence is')
        print(np.min(outlier_results[:,3]))
        print('posterior samples with coherence above 0.44')
        print(np.sum(posterior_results[:,3]>0.5))

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

        print('outlier coherences are')
        print(outlier_results[:,3])
        outlier_samples[:,2]/=10000
        print('minimal outlier coherence is')
        print(np.min(outlier_results[:,3]))
        print('posterior samples with coherence above 0.44')
        print(np.sum(posterior_results[:,3]>0.5))

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
        
        print('variance over mean over real value')
        print((power_spectrum_intercept/steady_state_protein)/sample_result[1])
        print('std over mean')
        print((np.sqrt(power_spectrum_intercept)/steady_state_protein)/sample_result[1])

    def xest_get_period_values_from_signal(self):
        time_points = np.linspace(0,1000,100000)
        signal_values = np.sin(2*np.pi/2*time_points) + 10
        period_values = hes5.get_period_measurements_from_signal(time_points,signal_values)
        for period_value in period_values[1:-1]:
            self.assertAlmostEqual(period_value, 2.0, 3)
        
        signal_values = np.sin(2*np.pi/1.42*time_points) + 10
        period_values = hes5.get_period_measurements_from_signal(time_points,signal_values)
        print(period_values)
        # in this case, for whatever weird boundary effect reason the hilbert won't give the right
        # response on the boundaries, let's check the mean instead
        self.assertAlmostEqual(np.mean(period_values), 1.42, 2)

    def xest_get_hilbert_periods_at_representative_model_parameter(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.15,
                                                   model_results[:,3]<0.2)))))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        sample = posterior_samples[2]
        these_mrna_traces, these_protein_traces = hes5.generate_multiple_langevin_trajectories( 200, # number_of_trajectories 
                                                                                           1500*5, #duration 
                                                                                           sample[2], #repression_threshold, 
                                                                                           sample[4], #hill_coefficient,
                                                                                           np.log(2)/30, #mRNA_degradation_rate, 
                                                                                           np.log(2)/90, #protein_degradation_rate, 
                                                                                           sample[0], #basal_transcription_rate, 
                                                                                           sample[1], #translation_rate,
                                                                                           sample[3], #transcription_delay, 
                                                                                           10, #initial_mRNA, 
                                                                                           sample[2], #initial_protein,
                                                                                           1000)
        
        this_power_spectrum,this_coherence, this_period = hes5.calculate_power_spectrum_of_trajectories(these_protein_traces)
        
        # get first set of periods
        all_periods = hes5.get_period_measurements_from_signal(these_protein_traces[:,0],
                                                               these_protein_traces[:,1])
        # and now add the rest
        for trace in these_protein_traces[:,1:].transpose():
            these_periods = hes5.get_period_measurements_from_signal(these_protein_traces[:,0], trace)
            all_periods = np.hstack((all_periods, these_periods))
        
        print(all_periods)
        plt.figure(figsize = (6.5,2.5))
        plt.subplot(121)
        plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1])
        plt.axvline(1/this_period, color = 'purple')
        plt.xlim(0,0.01)
        plt.xlabel('Frequency [1/min]')
        plt.ylabel('Power')
        plt.title('Period: '  + '{:.2f}'.format(this_period/60) + 'h, Coherence: ' + '{:.2f}'.format(this_coherence))
        
        plt.subplot(122)
        plt.hist(all_periods/60, range = (0,10), density = True, edgecolor = 'black')
        plt.axvline(this_period/60, color = 'purple')
#         plt.axvline(np.mean(all_periods)/60, color = 'black')
        plt.xlabel('Period [h]')
#         plt.ylim(0,0.0001)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'representative_hilbert_periods.pdf'))
        
    def xest_kolmogorov_smirnov_on_period_and_stdev(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
 
        experimental_periods = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data',
                                          'experimental_periods.csv'))

        experimental_stdevs = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data',
                                          'experimental_stdevs.csv'))
        
        simulated_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                          'shortened_posterior_hilbert_periods_per_cell_one_sample.npy'))
        
        real_simulated_periods = simulated_periods[simulated_periods<(12.0*60)]

        simulated_stdevs = posterior_results[:,1]
        
        print(experimental_periods)
        print(experimental_stdevs)
        
        period_stats = scipy.stats.ks_2samp(experimental_periods, real_simulated_periods/60)
        print('period kolmogorov-smirnov test is')
        print(scipy.stats.ks_2samp(experimental_periods, real_simulated_periods/60))
        
        stdev_stats = scipy.stats.ks_2samp(experimental_stdevs, simulated_stdevs)
        print('stdev kolmogorov-smirnov test is')
        print(scipy.stats.ks_2samp(experimental_stdevs, simulated_stdevs))
        
        plt.figure(figsize = [6.5,4.5])
        plt.subplot(221)
        sns.boxplot(data = [simulated_stdevs, experimental_stdevs])
        plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.text(0.25,0.2, 'K-S-value: ' + '{:.2f}'.format(stdev_stats[0]) + 
                 '\np-value: ' + '{:.2f}'.format(stdev_stats[1]), fontsize = 8)
        plt.ylabel('Stdev')
        
        plt.subplot(222)
        sns.boxplot(data = [real_simulated_periods/60, experimental_periods])
        plt.xticks([0,1], ['Model', 'Experiment']) 
#         plt.axhline(12)
        plt.text(0.25,10, 'K-S-value: ' + '{:.2f}'.format(period_stats[0]) + 
                 '\np-value: ' + '{:.2e}'.format(period_stats[1]), fontsize = 8)
        plt.ylabel('Period [h]')

        simulated_stdevs.sort()
        experimental_stdevs.sort()

        real_simulated_periods.sort()
        experimental_periods.sort()
        
        print('number of experimental periods:')
        number_of_experimental_periods = len(experimental_periods)
        print(len(experimental_periods))

        print('number or real_simulated_periods')
        number_of_simulated_periods = len(real_simulated_periods)
        print(len(real_simulated_periods))
        
        print('product over sum')
        print(number_of_experimental_periods*number_of_simulated_periods/(number_of_experimental_periods+
                                                                          number_of_simulated_periods))
        
#         plt.figure(figsize = [6.5,2.5])
        plt.subplot(223)
        plt.step(simulated_stdevs, (np.arange(simulated_stdevs.size)+1.0)/simulated_stdevs.size, label = "Model")
        plt.step(experimental_stdevs, (np.arange(experimental_stdevs.size)+1.0)/experimental_stdevs.size, label = "Experiment")
        plt.legend(loc='lower right')
        plt.xlabel('Stdev')
        plt.ylabel('Cumulative probability')
        
        plt.subplot(224)
        plt.step(real_simulated_periods/60, (np.arange(real_simulated_periods.size)+1.0)/real_simulated_periods.size)
        plt.step(experimental_periods, (np.arange(experimental_periods.size)+1.0)/experimental_periods.size)
#         plt.axhline(12)
        plt.xlabel('Period [h]')
        plt.ylabel('Cumulative probability')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                'output','Model_data_period_and_stdev_distribution_comparison.pdf'))
#         plt.tight_layout()
#         plt.savefig(os.path.join(os.path.dirname(__file__),
#                                  'output','Kolmogoriv_w_cumulative_plotted.pdf'))


    def xest_get_shortened_posterior_hilbert_period_distribution_smoothed_per_cell(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, measurement_interval = 12*60, 
                                                                             smoothen = True, per_cell = True)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output',
                                    'shortened_smoothened_posterior_hilbert_periods_per_cell'), hilbert_periods)
        
#         hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_smoothened_posterior_hilbert_periods_per_cell.pdf'))


    def xest_get_shortened_posterior_hilbert_period_distribution_smoothed(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
#         hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, measurement_interval = 12*60, 
#                                                                              smoothen = True)
        
#         np.save(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_smoothened_posterior_hilbert_periods'), hilbert_periods)
        
        hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                    'shortened_smoothened_posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,13), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_smoothened_posterior_hilbert_periods.pdf'))

    def xest_get_shortened_smoothened_posterior_hilbert_period_distribution_one_sample(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
#         hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, 
#                                                                              measurement_interval = 12*60,
#                                                                              per_cell = True,
#                                                                              smoothen = True,
#                                                                              samples_per_parameter_point = 1)
        
#         np.save(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_smoothened_posterior_hilbert_periods_per_cell_one_sample'), hilbert_periods)
        
        hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                    'shortened_smoothened_posterior_hilbert_periods_per_cell_one_sample.npy'))

        hilbert_periods = hilbert_periods[hilbert_periods<10*60]
        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_smoothened_posterior_hilbert_periods_per_cell_one_sample.pdf'))

    def xest_get_shortened_posterior_hilbert_period_distribution_one_sample(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'data',
#                                    'sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_repeated')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, 
                                                                             measurement_interval = 12*60,
                                                                             per_cell = True,
                                                                             samples_per_parameter_point = 1)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output',
                                    'repeated_shortened_posterior_hilbert_periods_per_cell_one_sample'), hilbert_periods)
        
#         hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_posterior_hilbert_periods_per_cell_one_sample.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_posterior_hilbert_periods_per_cell_one_sample.pdf'))

    def xest_get_shortened_posterior_hilbert_period_distribution(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'data',
#                                    'sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_repeated')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, measurement_interval = 12*60)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output',
                                    'shortened_repeated_posterior_hilbert_periods'), hilbert_periods)
        
#         hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_posterior_hilbert_periods.pdf'))

    def xest_get_shortened_posterior_hilbert_period_distribution_per_cell(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'data',
#                                    'sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_repeated')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, measurement_interval = 12*60, 
                                                                             per_cell = True)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output',
                                    'repeated_shortened_posterior_hilbert_periods_per_cell'), hilbert_periods)
        
#         hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
#                                     'shortened_posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'shortened_posterior_hilbert_periods_per_cell.pdf'))

    def xest_get_posterior_hilbert_period_distribution_per_cell(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
        hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples, per_cell = True)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output',
                                    'posterior_hilbert_periods_per_cell'), hilbert_periods)
        
#         hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
#                                     'posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'posterior_hilbert_periods_per_cell.pdf'))
 
    def xest_get_posterior_hilbert_period_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                    model_results[:,1]>0.05)))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        
#         hilbert_periods = hes5.calculate_hilbert_periods_at_parameter_points(posterior_samples)
        
#         np.save(os.path.join(os.path.dirname(__file__), 'output',
#                                    'posterior_hilbert_periods'), hilbert_periods)
        
        hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                    'posterior_hilbert_periods.npy'))

        plt.figure(figsize = (4.5,2.5))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [h]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'posterior_hilbert_periods.pdf'))
 
    def xest_in_silico_power_spectrum(self):
        time_points = np.linspace(0,20000,10000)
        in_silico_data = np.zeros((len(time_points),301))
        in_silico_data[:,0] = time_points 
        for trace_index in range(1,301):
            signal_values = np.sin(2*np.pi/220*time_points) + 10*np.random.rand(len(time_points))
            in_silico_data[:, trace_index] = signal_values

        this_power_spectrum,this_coherence, this_period = hes5.calculate_power_spectrum_of_trajectories(in_silico_data)

        plt.figure(figsize = (6.5,2.5))
#         plt.subplot(121)
        plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1])
        plt.xlim(0,0.01)
        plt.xlabel('Frequency [1/min]')
        plt.ylabel('Power')
        plt.title('Period: '  + '{:.2f}'.format(this_period/60) + 'h, Coherence: ' + '{:.2f}'.format(this_coherence))
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output',
                                   'in_silico_power_spectrum.pdf'))
 
    def xest_try_hilbert_transform(self):
        time_points = np.linspace(0,100,10000)
        signal_values = np.sin(2*np.pi/1.42*time_points)+10
#         time_points = np.linspace(0,15,100)
#         signal_values = np.sin(2*np.pi/2*time_points)
        analytic_signal = scipy.signal.hilbert(signal_values - np.mean(signal_values))
#         analytic_signal = scipy.signal.hilbert(signal_values)
        phase = np.angle(analytic_signal)
        print(np.signbit(phase).astype(int))
        #this will find the index just before zero-crossings from plus to minus
        phase_reset_indices = np.where(np.diff(np.signbit(phase).astype(int))>0)
        phase_reset_times = time_points[phase_reset_indices]
        extracted_periods = np.diff(phase_reset_times)
        print(extracted_periods)
        print(np.mean(extracted_periods))
        
        plt.figure(figsize = (4,2.5))
        plt.plot(time_points, signal_values, label = 'signal')
        plt.plot(time_points, phase, label = 'phase')
        plt.vlines(phase_reset_times, -4,4, color = 'black')
        plt.xlim(0,20)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','initial_hilbert_test.pdf'))
        
        plt.figure(figsize = (4,2.5))
        signal_values = signal_values + np.random.rand(len(signal_values))
#         analytic_signal = scipy.signal.hilbert(signal_values)
        analytic_signal = scipy.signal.hilbert(signal_values - np.mean(signal_values))
        phase = np.angle(analytic_signal)
        phase_reset_indices = np.where(np.diff(np.signbit(phase).astype(int))>0)
        phase_reset_times = time_points[phase_reset_indices]
        extracted_periods = np.diff(phase_reset_times)
        print(extracted_periods)
        print(np.mean(extracted_periods))
        plt.plot(time_points, signal_values, label = 'signal', lw = 0.1)
        plt.plot(time_points, phase, label = 'phase', lw = .1)
        plt.vlines(phase_reset_times, -1,1, color = 'black', zorder = 10, lw = .1)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.xlim(0,20)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','initial_hilbert_test2.pdf'))
        
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.12,
                                                   model_results[:,3]<0.17)))))  #standard deviation

        posterior_samples = prior_samples[accepted_indices]
        posterior_results = model_results[accepted_indices]
        sample = posterior_samples[2]
        these_traces = hes5.generate_langevin_trajectory( 1500*5, #duration 
                                                          sample[2], #repression_threshold, 
                                                          sample[4], #hill_coefficient,
                                                          np.log(2)/30, #mRNA_degradation_rate, 
                                                          np.log(2)/90, #protein_degradation_rate, 
                                                          sample[0], #basal_transcription_rate, 
                                                          sample[1], #translation_rate,
                                                          sample[3], #transcription_delay, 
                                                          10, #initial_mRNA, 
                                                          sample[2], #initial_protein,
                                                          1000)
        
        plt.figure(figsize = (6.5,2.5))
        plt.subplot(121)
        signal_values = these_traces[:,2]
        time_points = these_traces[:,0]
#         analytic_signal = scipy.signal.hilbert(signal_values)
        analytic_signal = scipy.signal.hilbert(signal_values - np.mean(signal_values))
        phase = np.angle(analytic_signal)
        phase_reset_indices = np.where(np.diff(np.signbit(phase).astype(int))>0)
        phase_reset_times = time_points[phase_reset_indices]
        extracted_periods = np.diff(phase_reset_times)
        print(extracted_periods)
        print(np.mean(extracted_periods))
        plt.plot(time_points, signal_values, label = 'signal', lw = 0.1)
        plt.vlines(phase_reset_times, 45000,55000, color = 'black', zorder = 10, lw = .5)
        plt.xlabel("Time [min]")
        plt.ylabel("Expression")
        
        implemented_periods = hes5.get_period_measurements_from_signal(time_points, signal_values)
        
        self.assert_(np.array_equal(implemented_periods, extracted_periods))
        
        plt.subplot(122)
        plt.plot(time_points, phase, label = 'phase', lw = .1)
        plt.vlines(phase_reset_times, -1,1, color = 'black', zorder = 10, lw = .5)
        plt.xlabel("Time [min]")
        plt.ylabel("Phase")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','initial_hilbert_on_data.pdf'))
        
        plt.figure(figsize = (6.5,2.5))
        plt.subplot(121)
        smoothened_signal = scipy.signal.savgol_filter(signal_values, 
                                                            75, 
                                                            3)
#         analytic_signal = scipy.signal.hilbert(signal_values)
        analytic_signal = scipy.signal.hilbert(smoothened_signal - np.mean(smoothened_signal))
        phase = np.angle(analytic_signal)
        phase_reset_indices = np.where(np.diff(np.signbit(phase).astype(int))>0)
        phase_reset_times = time_points[phase_reset_indices]
        extracted_periods = np.diff(phase_reset_times)
        print(extracted_periods)
        print(np.mean(extracted_periods))
        plt.plot(time_points, smoothened_signal, label = 'signal', lw = 0.1)
        plt.vlines(phase_reset_times, 45000,55000, color = 'black', zorder = 10, lw = .5)
        plt.xlabel("Time [min]")
        plt.ylabel("Expression")
        
        implemented_periods = hes5.get_period_measurements_from_signal(time_points, signal_values, smoothen = True)
        
        self.assert_(np.array_equal(implemented_periods, extracted_periods))
        
        plt.subplot(122)
        plt.plot(time_points, phase, label = 'phase', lw = .1)
        plt.vlines(phase_reset_times, -1,1, color = 'black', zorder = 10, lw = .5)
        plt.xlabel("Time [min]")
        plt.ylabel("Phase")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','initial_hilbert_on_data_smoothened.pdf'))
        
    def xest_ngn_hes5_toy_model(self):
        times = np.linspace(0,15,1000)
        noise_1 = np.random.randn(len(times)) 
        noise_2 = np.random.randn(len(times)) 
#         noise_1 = 0
#         noise_2 = 0
        these_hes5_data = 2*np.sin(2*np.pi*times/4.0)-times + 15 + noise_1
        these_ngn_data = times + noise_2
        
        plt.figure(figsize = (6.5,6.5))
        plt.subplot(421)
        plt.plot(times, these_hes5_data, label = 'Hes5')
        plt.plot(times, these_ngn_data, label = 'Ngn')
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.legend(loc = 'upper left')
        plt.subplot(422)
        plt.plot(these_hes5_data, these_ngn_data)
        plt.xlabel('Hes5 expression')
        plt.ylabel('Ngn expression')

        these_ngn_data = 2*np.sin(2*np.pi*times/4.0)+times + noise_2
        plt.subplot(423)
        plt.plot(times, these_hes5_data)
        plt.plot(times, these_ngn_data)
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.subplot(424)
        plt.plot(these_hes5_data, these_ngn_data)
        plt.xlabel('Hes5 expression')
        plt.ylabel('Ngn expression')

        these_ngn_data = 2*np.sin(2*np.pi*times/4.0+np.pi)+times + noise_2
        plt.subplot(425)
        plt.plot(times, these_hes5_data)
        plt.plot(times, these_ngn_data)
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.subplot(426)
        plt.plot(these_hes5_data, these_ngn_data)
        plt.xlabel('Hes5 expression')
        plt.ylabel('Ngn expression')

        these_ngn_data = np.sin(2*np.pi*times/4.0)*times/2 + times/2 + noise_2
        plt.subplot(427)
        plt.plot(times, these_hes5_data)
        plt.plot(times, these_ngn_data)
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.subplot(428)
        plt.plot(these_hes5_data, these_ngn_data)
        plt.xlabel('Hes5 expression')
        plt.ylabel('Ngn expression')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','hes5_ngn_toy_model.pdf'))
