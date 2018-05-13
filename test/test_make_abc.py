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
# import xlrd

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestSimpleHes5ABC(unittest.TestCase):
                                 
    def xest_make_abc(self):
        ## generate posterior samples
        total_number_of_samples = 2000
        acceptance_ratio = 0.03
        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                use_langevin = False )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
        
    def xest_make_abc_on_cluster(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.03
        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 16,
                                                                number_of_cpus = 16,
                                                                use_langevin = False )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_abc_differently(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results')
        acceptance_ratio = 0.03
        total_number_of_samples = 2000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        distance_table = hes5.calculate_distances_to_data(model_results)
    
        my_posterior_samples = hes5.select_posterior_samples( prior_samples, 
                                                  distance_table, 
                                                  acceptance_ratio )

        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        data_frame = pd.DataFrame( data = my_posterior_samples,
                               columns= ['transcription_rate', 
                                         'translation_rate', 
                                         'repression_threshold', 
                                         'transcription_delay'])
        pairplot = sns.pairplot(data_frame)
#         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_dots_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
        
    def xest_plot_abc_in_band(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results')
        acceptance_ratio = 0.03
        total_number_of_samples = 2000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                          np.logical_and(model_results[:,0]<65000, #cell_number
                                                   model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))
        my_posterior_samples = prior_samples[accepted_indices]

        # plot distribution of accepted parameter samples
        data_frame = pd.DataFrame( data = my_posterior_samples,
                               columns= ['transcription_rate', 
                                         'translation_rate', 
                                         'repression_threshold', 
                                         'transcription_delay'])
        pairplot = sns.pairplot(data_frame)
#         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_bands_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
        
    def xest_make_langevin_abc(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02
#         total_number_of_samples = 20
#         acceptance_ratio = 0.5
        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_langevin_200reps' )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_make_langevin_abc_different_prior(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

        prior_bounds = {'basal_transcription_rate' : (0,10),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,150000),
                        'time_delay' : (5,40)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 100,
                                                                saving_name = 'sampling_results_langevin_small_prior',
                                                                prior_bounds = prior_bounds )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_different_prior' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_make_abc_all_parameters(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

        prior_bounds = {'basal_transcription_rate' : (0,100),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,150000),
                        'time_delay' : (5,40),
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
                        'mRNA_degradation_rate': (0.001, 0.04),
                        'protein_degradation_rate': (0.001, 0.04)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_all_parameters_200',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full' )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 6))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_different_prior' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_make_hill_abc(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0,100),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,100000),
                        'time_delay' : (5,40),
                        'hill_coefficient': (2,7)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_hill',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill')
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_abc_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def test_pairplot_prior(self):
#         acceptance_ratio = 0.02
        total_number_of_samples = 200000
        
        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (1,40),
                        'repression_threshold' : (0,120000),
                        'time_delay' : (5,40),
                        'hill_coefficient': (2,6)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),
        prior_samples = hes5.generate_prior_samples( total_number_of_samples, True,
                                                prior_bounds, 'hill', True)


#         pairplot = hes5.plot_posterior_distributions(prior_samples)
#         pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','pairplot_log_prior.pdf'))

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
                                    'output','fixed_prior.pdf'))

    def xest_make_abc_logarithmic_prior_vary_bounds(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,100),
                        'translation_rate' : (1,200),
                        'repression_threshold' : (0,100000),
                        'time_delay' : (5,40),
                        'hill_coefficient': (2,6)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_logarithmic',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill',
                                                                logarithmic = True)
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_abc_logarithmic_prior_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_make_abc_logarithmic_prior(self):
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,100),
                        'translation_rate' : (1,200),
                        'repression_threshold' : (0,100000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_logarithmic',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill',
                                                                logarithmic = True)
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_abc_logarithmic_prior_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_larger_variation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                                   model_results[:,1]>0.3)))  #standard deviation
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        print('coherences are')
        print(model_results[accepted_indices][:,3])
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.diag_axes[0].set_ylim(0,10)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_larger_amplitude.pdf'))
 
    def xest_plot_smaller_mean(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>5500, #cell number
                                    np.logical_and(model_results[:,0]<6500, #cell_number
#                                                    model_results[:,1]>0.3)))  #standard deviation
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]>0.2)))))  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        print('coherences are')
        print(model_results[accepted_indices][:,3])
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.diag_axes[0].set_ylim(0,1000)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_small_mean.pdf'))
 
    def xest_plot_large_amplitude_trace(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                                   model_results[:,1]>0.3)))  #standard deviation
        my_posterior_samples = prior_samples[accepted_indices]
        
        this_parameter = my_posterior_samples[1]

        this_trace = hes5.generate_langevin_trajectory(  duration = 1500,
                                                         repression_threshold = this_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = this_parameter[3],
                                                         basal_transcription_rate = this_parameter[0],
                                                         translation_rate = this_parameter[1],
                                                         initial_mRNA = 10,
                                                         hill_coefficient = this_parameter[4],
                                                         initial_protein = this_parameter[2],
                                                         equilibration_time = 1000
                                                       )

        plt.figure(figsize = (4.5,2.5))
        plt.plot(this_trace[:,0], this_trace[:,2]/1e4)

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','large_amplitude_example.pdf'))

    def xest_plot_logarithmic_prior_bands(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    prior_samples[:,1]<10))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.diag_axes[0].set_ylim(0,1000)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_logarithmic_bands.pdf'))
 
    def xest_plot_logarithmic_prior_oscillating(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,3]>0.3))))) #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.diag_axes[0].set_ylim(0,30)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_logarithmic_oscillating.pdf'))
 
    def xest_plot_logarithmic_prior_not_oscillating(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                    np.logical_and(prior_samples[:,1]<10, #time_delay
                                                   model_results[:,3]>0.3)))))) #coherence
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.diag_axes[0].set_ylim(0,30)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_logarithmic_not_oscillating.pdf'))
 
    def xest_plot_period_distribution_logarithmic_prior(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[:,2]
        plt.hist(all_periods, range = (0,600), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','logarithmic_full_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','logarithmic_full_mrna_distribution.pdf'))

#         accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
#                                     np.logical_and(model_results[:,0]<65000, #cell_number
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,
#                                                     model_results[:,3]>0.1)))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = my_model_results[:,2]
        plt.hist(all_periods, range = (0,600), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','logarithmic_abc_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = my_model_results[:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','logarithmic_abc_mrna_distribution.pdf'))
 
    def xest_plot_period_distribution_for_poster(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    prior_samples[:,1]<10))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

#         real_data = [ 6.4135025721, 6.9483225932, 2.6887457703, 3.8620874625, 3.2559540745,
#                       4.4568030424, 5.2120783369, 4.3169191105, 4.2472576997, 2.7684001434,
#                       3.6331949226, 5.365000329,  1.1181243755, 4.2130976958, 6.3381760719,
#                       2.466899605,  4.7849990718, 5.2029517316, 4.2038143391, 3.9909362984,
#                       3.2734490618, 4.3116631965, 5.3199423883] 
        
        ## the values that verionica sent initially
#          
        real_data = [2.0075009033, 5.1156200644, 7.7786868129, 6.4328452748, 7.441794935,
                     7.0127707313, 2.6890681359, 3.4454911902, 3.8689181126, 3.2493764293,
                     6.3817264371, 5.8903734106, 4.5034984657, 3.4247641996, 4.4767623623, 
                     4.1803337503, 5.2752672662, 6.9038758003, 4.3200156205, 4.2588402084, 
                     6.1428930891, 5.4124817274, 5.0135377758, 2.8156245427, 5.5008033408, 
                     3.6331974295, 5.295813407,  1.1181243876, 5.5984263674, 4.2800118281, 
                     6.7713656265, 3.4585300534, 6.3727670575, 2.4668994841, 6.3725171059,
                     4.8021898758, 4.8108333392, 5.9935335349, 6.2570622822, 5.2284704987,
                     4.2143881493, 4.0659270434, 3.9990674449, 4.4410420437, 6.7406002947,
                     5.0648853886, 1.8765732885, 3.307425174,  5.6208186717, 4.3185605778,
                     5.186842823,  5.6310823986, 7.4402931009]

        sns.set(font_scale = 1.5)
        font = {'size'   : 28}
        plt.rc('font', **font)

        all_periods = my_model_results[:,2]
# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                   'Data' : np.array(real_data)*60})
        my_figure = plt.figure(figsize= (5,3))
        sns.boxplot(data = [all_periods[all_periods<600], np.array(real_data)*60])
        plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.ylabel('Period [min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_period_distribution_for_poster.pdf'))
 
    def xest_plot_period_distribution_for_coherences(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        my_figure = plt.figure(figsize = (6.5,6.5))
        
        coherence_bands = [[0.0,1.0],
                           [0.0,0.1],
                           [0.1,0.2],
                           [0.2,0.3],
                           [0.3,0.4],
#                            [0.4,0.5]]
                           [0.4,0.5],
                           [0.5,0.6],
                           [0.6,0.7]]
        
        for coherence_index, coherence_band in enumerate(coherence_bands):
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                        np.logical_and(model_results[:,0]<65000, #cell_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>coherence_band[0],
                                                       model_results[:,3]<coherence_band[1]))))))  #standard deviation

            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]
            
            my_figure.add_subplot(4,2,coherence_index + 1)

            all_periods = my_model_results[:,2]
            plt.hist(all_periods, range = (0,600), bins = 20)
            plt.title(r'Coherence $\in$ ' 
                        + np.array_str(np.array(coherence_band), precision=1))
            plt.xlabel('Period [min]')
            plt.ylabel('Occurrence')
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','logarithmic_abc_period_distribution_for_coherences.pdf'))
 
    def xest_upsample_hill_abc(self):
        ## generate posterior samples
        total_number_of_samples = 200000
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0,4),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,100000),
                        'time_delay' : (5,40),
                        'hill_coefficient': (2,7)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_hill_low_transcription',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill')
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_abc_low_transcription_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_make_abc_all_parameters_long_delay(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

        prior_bounds = {'basal_transcription_rate' : (0,100),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,150000),
                        'time_delay' : (20,40),
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
                        'mRNA_degradation_rate': (0.001, 0.04),
                        'protein_degradation_rate': (0.001, 0.04)}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 100,
                                                                saving_name = 'sampling_results_all_parameters_long_delay',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full' )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 6))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_full_abc_long_delay.pdf'))
 
    def xest_plot_langevin_abc_differently(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        distance_table = hes5.calculate_distances_to_data(model_results)
    
        my_posterior_samples = hes5.select_posterior_samples( prior_samples, 
                                                  distance_table, 
                                                  acceptance_ratio )

        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        data_frame = pd.DataFrame( data = my_posterior_samples,
                               columns= ['transcription_rate', 
                                         'translation_rate', 
                                         'repression_threshold', 
                                         'transcription_delay'])

        sns.set()
        pairplot = sns.pairplot(data_frame)
#         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_dots_langevin_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_full_abc_in_band(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_200')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_all_parameters.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_full_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_full_mrna_distribution.pdf'))

    def xest_plot_abc_all_parameters_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_200')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]>0.3))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_all_parameters_oscillating.pdf'))
 
    def xest_plot_abc_all_parameters_not_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_200')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.2))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_all_parameters_not_oscillating.pdf'))
 
    def xest_plot_full_abc_in_band_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_long_delay')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                    model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_full_bands_long_delay.pdf'))
 
        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_full_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_full_mrna_distribution.pdf'))

    def xest_plot_full_abc_not_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_long_delay')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,3]<0.3, #coherence
                                                   prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_full_long_delay_not_oscillating.pdf'))
 
    def xest_plot_full_abc_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters_long_delay')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,3]>0.3, #coherence
                                                   prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_full_long_delay_oscillating.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating_different_prior(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_small_prior')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.3))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_different_prior.pdf'))
 
    def xest_plot_different_prior_in_band(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_small_prior')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_bands_different_prior.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_small_prior')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,3]<0.3, #coherence
                                                   prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_long_delay_different_prior.pdf'))
 
    def xest_plot_upsample_hill_abc_in_band(self):
        # generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.axes[-1,0].set_xlim(0,5)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_low_transcription.pdf'))
 
    def xest_plot_hill_abc_in_band(self):
        # generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_bands.pdf'))
 
    def xest_plot_heterozygous_homozygous_comparison(self):
        # generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        number_of_traces_per_sample = 200
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        
#         model_results = hes5.calculate_heterozygous_summary_statistics_at_parameters( my_posterior_samples, 
#                                                                                     number_of_traces_per_sample )

#         np.save(os.path.join(os.path.dirname(__file__), 'output','heterozygous_comparison.npy'), model_results)
        model_results = np.load(os.path.join(os.path.dirname(__file__), 'output','heterozygous_comparison.npy'))

        my_figure = plt.figure( figsize = (6,5) )
        my_figure.add_subplot(221)
        plt.scatter(model_results[:,0,0]/10000, model_results[:,1,0]/10000,
                    color = 'grey', lw = 0, marker = '.')
        plt.plot(model_results[:,0,0]/10000, model_results[:,0,0]/20000,
                 color = 'black')
        plt.xlabel('Homozygous mean')
        plt.ylabel('Allele mean')
        plt.text(0.1, 0.8, 'Slope: 1/2',
                           transform=plt.gca().transAxes)
 
        
        my_figure.add_subplot(222)
        plt.scatter(model_results[:,0,1], model_results[:,1,1],
                    color = 'grey', lw = 0, marker = '.')
        plt.plot(model_results[:,0,1], model_results[:,0,1]*2,
                 color = 'black')
        plt.text(0.1, 0.8, 'Slope: 2',
                           transform=plt.gca().transAxes)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
#         plt.gca().locator_params(axis='y', tight = True, nbins=5)
        plt.xlabel('Homozygous std/mean')
        plt.ylabel('Allele std/mean')
        
        my_figure.add_subplot(223)
        plt.scatter(model_results[:,0,2], model_results[:,1,2],
                    color = 'grey', lw = 0, marker = '.')
        plt.plot(model_results[:,0,2], model_results[:,0,2],
                 color = 'black')
        plt.text(0.1, 0.8, 'Slope: 1',
                           transform=plt.gca().transAxes)
        plt.xlim(0,500)
        plt.ylim(0,)
        plt.xlabel('Homozygous period')
        plt.ylabel('Allele period')
        
        my_figure.add_subplot(224)
        plt.scatter(model_results[:,0,3], model_results[:,1,3],
                    color = 'grey', lw = 0, marker = '.')
        plt.plot(model_results[:,0,3], model_results[:,0,3] - 0.2,
                 color = 'black')
        plt.text(0.1, 0.8, 'Slope: 1, offset: -0.2',
                           transform=plt.gca().transAxes)
        plt.ylim(0,)
        plt.xlabel('Homozygous coherence')
        plt.ylabel('Allele coherence')
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','heterozygous_homozygous_comparison.pdf'))
 
    def xest_plot_langevin_abc_in_band(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_bands.pdf'))
 
    def xest_plot_heterozygous_abc(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]

        sns.set()
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_heterozygous_bands.pdf'))
 
    def xest_plot_langevin_abc_in_band_oscillating_different_prior(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_small_prior')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]>0.3))))) #coherence


        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_oscillating_bands_different_prior.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating_different_prior(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_small_prior')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.3))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_different_prior.pdf'))
 
    def xest_plot_hill_abc_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]>0.2))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_oscillating.pdf'))
 
    def xest_plot_hill_abc_not_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.2))))) #coherence


        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_not_oscillating.pdf'))
 
    def xest_plot_heterozygous_abc_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]>0.2))))) #coherence


        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_heterozygous_oscillating.pdf'))
 
    def xest_plot_langevin_abc_in_band_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]>0.3))))) #coherence


        my_posterior_samples = prior_samples[accepted_indices]

        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        print('number of accepted samples is ' + str(len(my_posterior_samples)))
#         sns.set()
#         # plot distribution of accepted parameter samples
#         data_frame = pd.DataFrame( data = my_posterior_samples,
#                                columns= ['transcription_rate', 
#                                          'translation_rate', 
#                                          'repression_threshold', 
#                                          'transcription_delay'])
#         pairplot = sns.pairplot(data_frame)
#         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_oscillating_bands.pdf'))
 
    def xest_plot_heterozygous_abc_not_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.15))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_heterozygous_not_oscillating.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                                   model_results[:,3]<0.2))))) #coherence

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
#         sns.set()
#         # plot distribution of accepted parameter samples
#         data_frame = pd.DataFrame( data = my_posterior_samples,
#                                columns= ['transcription_rate', 
#                                          'translation_rate', 
#                                          'repression_threshold', 
#                                          'transcription_delay'])
#         pairplot = sns.pairplot(data_frame)
# #         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating.pdf'))
 
    def xest_plot_langevin_abc_in_band_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,3]>0.3, #coherence
                                                   prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
#         sns.set()
#         # plot distribution of accepted parameter samples
#         data_frame = pd.DataFrame( data = my_posterior_samples,
#                                columns= ['transcription_rate', 
#                                          'translation_rate', 
#                                          'repression_threshold', 
#                                          'transcription_delay'])
#         pairplot = sns.pairplot(data_frame)
# #         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_oscillating_bands_long_delay.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,3]<0.2, #coherence
                                                   prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is ' + str(len(my_posterior_samples)))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
#         sns.set()
#         # plot distribution of accepted parameter samples
#         data_frame = pd.DataFrame( data = my_posterior_samples,
#                                columns= ['transcription_rate', 
#                                          'translation_rate', 
#                                          'repression_threshold', 
#                                          'transcription_delay'])
#         pairplot = sns.pairplot(data_frame)
# #         pairplot.map_diag(sns.kdeplot)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_long_delay.pdf'))
 
    def xest_plot_heterozygous_mrna_and_period_distributions(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','heterozygous_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','heterozygous_mrna_distribution.pdf'))
 
    def xest_plot_low_transcription_mrna_and_period_distributions(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_hill_low_transcription_period_distribution.pdf'))
        
#         new_accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
#                                     np.logical_and(model_results[:,0]<65000, #cell_number
#                                     np.logical_and(model_results[:,1]<0.15,  #standard deviation
# #                                                     model_results[:,1]>0.05))))  #standard deviation
#                                                    model_results[:,1]>0.05))))  #standard deviation
# #                                                     prior_samples[:,4]<6))))) #hill
# 
#         new_periods = model_results[new_accepted_indices][:,2]
# 
# 
#         ## need to rerun abc with mrna numbers
#         my_figure = plt.figure(figsize = (4,2.5))
#         plt.hist(new_periods, bins = 200)
# #         plt.hist(new_periods, range = (0,400), bins = 20)
#         plt.xlabel('Period [min]')
#         plt.ylabel('Occurrence')
#         plt.xlim(0,500)
#         plt.tight_layout()
#         plt.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','abc_hill_low_transcription_period_distribution.pdf'))


        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_hill_low_transcription_mrna_distribution.pdf'))
 
    def xest_plot_hill_mrna_and_period_distributions(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_hill_period_distribution.pdf'))
        
        new_accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                     model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    prior_samples[:,4]<6))))) #hill
        new_periods = model_results[new_accepted_indices][:,2]


        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        plt.hist(new_periods, bins = 200)
#         plt.hist(new_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.xlim(0,500)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_hill_6_period_distribution.pdf'))


        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_hill_mrna_distribution.pdf'))
 
    def xest_plot_mrna_and_period_distributions(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay

        ## need to rerun abc with mrna numbers
        my_figure = plt.figure(figsize = (4,2.5))
        all_periods = model_results[accepted_indices][:,2]
        plt.hist(all_periods, range = (0,400), bins = 20)
        plt.xlabel('Period [min]')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_period_distribution.pdf'))

        my_figure = plt.figure(figsize = (4,2.5))
        all_mrna_counts = model_results[accepted_indices][:,4]
        plt.hist(all_mrna_counts, range = (0,150), bins = 50)
        plt.xlabel('Average mRNA count')
        plt.ylabel('Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','abc_mrna_distribution.pdf'))
 
    def xest_plot_langevin_abc_in_band_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
#                                                    model_results[:,1]>0.05))))  #standard deviation
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                                    prior_samples[:,3]>20))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','pairplot_langevin_bands_long_delay.pdf'))
 
    def xest_make_heterozygous_degradation_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 100

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
                                                                                          number_of_parameter_points,
                                                                                          number_of_trajectories)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output','multiple_degradation_sweep_results_new.npy'),
                my_parameter_sweep_results)

    def xest_make_multiple_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 100

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
                                                                                          number_of_parameter_points,
                                                                                          number_of_trajectories)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output','multiple_degradation_sweep_results_new.npy'),
                my_parameter_sweep_results)

    def xest_plot_multiple_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 100

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output','multiple_degradation_sweep_results_new.npy'))
        
#         new_accepted_indices = np.where( my_posterior_samples[:,0] < 10 )
#         my_parameter_sweep_results = my_parameter_sweep_results[new_accepted_indices]

        my_figure = plt.figure( figsize = (6.5, 1.5) )
        my_figure.add_subplot(131)
        for results_table in my_parameter_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,3], color ='black', alpha = 0.01)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(132)
        for results_table in my_parameter_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'black', alpha = 0.01)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Coherence')
        plt.ylim(0,1)
        
        my_figure.add_subplot(133)
        for results_table in my_parameter_sweep_results:
            plt.errorbar(results_table[:,0],
                         results_table[:,1]/10000,
                         yerr = results_table[:,2]/10000*results_table[:,1],
                         color = 'black', alpha = 0.01)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.ylim(0,15)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Expression/1e4')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','multiple_degradation_sweep.pdf'))
        
    def xest_make_all_multiple_parameter_variation_hill_low_transcription(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_parameter_sweeps_hill_low_transcription' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_make_all_multiple_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_parameter_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_make_heterozygous_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_heterozygous_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_make_logarithmic_degradation_rate_sweep(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','logarithmic_degradation_sweep.npy'),
                    my_sweep_results)

    def xest_plot_bifurcation_implementation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
#         sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))
#                                     np.logical_and(model_results[:,1]>0.05,
#                                                    model_results[:,3]<0.1))))) #standard deviation
#                                     np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,1]<10))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        new_accepted_indices = np.where(my_posterior_samples[:,1]<10)

        my_figure = plt.figure( figsize = (6.5, 1.5) )

        my_figure.add_subplot(131)
        my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                                          'logarithmic_degradation_sweep.npy'))
        
        new_accepted_indices = np.where(my_posterior_samples[:,1]<10)
#         my_indices = np.where(np.logical_and(my_degradation_sweep_results[:,3,4]>0.1,
#                                              my_degradation_sweep_results[:,3,4]<0.2))
#         my_degradation_sweep_results = my_degradation_sweep_results[new_accepted_indices]

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
        plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)
#         plt.ylim(0,0.3)

        my_figure.add_subplot(132)
        hill_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'logarithmic_relative_sweeps_hill_coefficient.npy'))
#         my_indices = np.where(np.logical_and(hill_sweep_results[:,9,4]>0.1,
#                                              hill_sweep_results[:,9,4]<0.2))
#         hill_sweep_results = hill_sweep_results[my_indices]
        for results_table in hill_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel('rel. Hill coefficient')
        plt.axvline( 1.0, color = 'darkblue' )
        plt.gca().text(x_coord, y_coord, 'B', transform=plt.gca().transAxes)
#         plt.ylabel('Coherence')
#         plt.ylim(0,0.3)
        plt.ylim(0,1)               

        my_figure.add_subplot(133)
        delay_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'logarithmic_relative_sweeps_time_delay.npy'))
#         delay_sweep_results = delay_sweep_results[my_indices]
        for results_table in delay_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'teal', alpha = 0.02, zorder = 0)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().set_rasterization_zorder(1)
        plt.axvline( 1.0, color = 'darkblue')
        plt.xlabel('rel. Transcription delay')
        plt.gca().text(x_coord, y_coord, 'C', transform=plt.gca().transAxes)
#         plt.ylabel('Coherence')
#         plt.ylim(0,0.3)
        plt.ylim(0,1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','bifurcation_illustration.pdf'), dpi = 400)
 
    def xest_plot_logarithmic_degradation_sweep(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                                          'logarithmic_degradation_sweep.npy'))
        
#         new_accepted_indices = np.where( my_posterior_samples[:,0] < 10 )
#         my_parameter_sweep_results = my_parameter_sweep_results[new_accepted_indices]

        my_figure = plt.figure( figsize = (6.5, 1.5) )
        my_figure.add_subplot(131)
        for results_table in my_parameter_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,3], color ='black', alpha = 0.005)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(132)
        for results_table in my_parameter_sweep_results:
            plt.plot(results_table[:,0],
                     results_table[:,4], color = 'black', alpha = 0.005)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Coherence')
        plt.ylim(0,1)
        
        my_figure.add_subplot(133)
        for results_table in my_parameter_sweep_results:
            plt.errorbar(results_table[:,0],
                         results_table[:,1]/10000,
                         yerr = results_table[:,2]/10000*results_table[:,1],
                         color = 'black', alpha = 0.005)
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.ylim(0,15)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Expression/1e4')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','logarithmic_degradation_sweep.pdf'))
 
    def xest_make_logarithmic_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
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
            np.save(os.path.join(os.path.dirname(__file__), 'output','logarithmic_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_make_hill_relative_parameter_variation_low_transcription_rate(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
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
            np.save(os.path.join(os.path.dirname(__file__), 'output','hill_relative_sweeps_low_transcription' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_make_hill_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
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
            np.save(os.path.join(os.path.dirname(__file__), 'output','hill_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_hill_relative_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
#         model_results = model_results[other_accepted_indices]


#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'hill_relative_sweeps_' + parameter_name + '.npy'))

#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                                       my_parameter_sweep_results[:,
                                                                reference_indices[parameter_name] -1 ,3] < 400))

            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1.0)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,8)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            print('hello?')
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','hill_all_relative_sweep_' + parameter_name + '.pdf'))
 
    def xest_plot_hill_relative_parameter_variation_transitions(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = model_results[other_accepted_indices]


#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'hill_relative_sweeps_' + parameter_name + '.npy'))

          
#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,8)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','hill_relative_sweep_' + parameter_name + '.pdf'))
 
    def xest_make_relative_heterozygous_parameter_variation_low_variance(self):
        number_of_parameter_points = 20
#         number_of_parameter_points = 3
        number_of_trajectories = 200
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.075, #standard deviation
                                                   model_results[:,1]>0.025)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_relative_sweeps_low_variance_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_hill_sweep_low_variance(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.075, #standard deviation
                                                    model_results[:,1]>0.025)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
#         model_results = model_results[other_accepted_indices]


#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_relative_sweeps_low_variance_' + parameter_name + '.npy'))

#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,8)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','hill_relative_sweep_low_variance_' + parameter_name + '.pdf'))
 
    def xest_make_heterozygous_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_heterozygous_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_heterozygous_relative_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = model_results[other_accepted_indices]


#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_heterozygous_relative_sweeps_' + parameter_name + '.npy'))

            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,8)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','heterozygous_relative_sweep_' + parameter_name + '.pdf'))
 
    def xest_investigate_heterozygous_relative_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>25000, #cell number
                                    np.logical_and(model_results[:,0]<35000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = accepted_model_results[other_accepted_indices]
        my_posterior_samples = my_posterior_samples[other_accepted_indices]


#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in ['mRNA_degradation_rate', 'repression_threshold']:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_heterozygous_relative_sweeps_' + parameter_name + '.npy'))

            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            rising_coherence_indices, = np.where(my_parameter_sweep_results[:,15,4] < 0.1)
            
            my_parameter_sweep_results = my_parameter_sweep_results[rising_coherence_indices]
            
            these_samples = my_posterior_samples[rising_coherence_indices]
            these_model_results = model_results[rising_coherence_indices]
            np.set_printoptions(precision=3, suppress = True)
            print(these_model_results)

            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,8)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                              'output',
                              'invistigating_heterozygous_relative_sweep_' + 
                              parameter_name + '.pdf'))
            
            pairplot = hes5.plot_posterior_distributions( these_samples )
            pairplot.axes[3,0].set_xlim(0,2)
            pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                          'output','pairplot_heterozygous_rising_coherence_' + 
                                          parameter_name + '.pdf'))
 
        pairplot2 = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot2.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_heterozygous_coherence_0.2.pdf'))
 
    def xest_plot_low_transcription_relative_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = model_results[other_accepted_indices]

#         parameter_names = ['basal_transcription_rate',
#                            'translation_rate',
#                            'repression_threshold',
#                            'time_delay',
#                            'mRNA_degradation_rate',
#                            'protein_degradation_rate',
#                            'hill_coefficient']
 
        parameter_names = ['repression_threshold']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'hill_relative_sweeps_low_transcription' + parameter_name + '.npy'))
        
            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]

            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                               results_table[:,3], color ='black', alpha = 0.01)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                               results_table[:,4], color = 'black', alpha = 0.01)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
#             this_axis.set_ylim(0,1)
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.01)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','relative_low_transcription_sweep_' + parameter_name + '.pdf'))
 
    def xest_investigate_weird_parameter_behaviour(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data', 'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
        
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
#
        print('number of existing samples is')
        print(len(my_posterior_samples))
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_basal_transcription_rate.npy'))
        
        decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                     my_parameter_sweep_results[:,9,3] > 300),
                                        np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                       my_parameter_sweep_results[:,4,4] > 0.2)))

        accepted_samples = my_posterior_samples[decrease_indices]
        print('number of accepted samples is')
        print(len(accepted_samples))
        pairplot = hes5.plot_posterior_distributions(accepted_samples)
        pairplot.diag_axes[0].set_ylim(0,200)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','pairplot_weird_transcription_rate_behaviour.pdf'))
 
    def xest_investigate_where_repression_threshold_changes_period(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data', 'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
        
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
#
        print('number of existing samples is')
        print(len(my_posterior_samples))
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_repression_threshold.npy'))
        
        my_other_indices = np.where(my_parameter_sweep_results[:,9,4]>0.1)
        print('number of qualifiying samples is')
        print(len(my_other_indices[0]))

        my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,4,4]>my_parameter_sweep_results[:,9,4],
                                             my_parameter_sweep_results[:,9,4]>0.1))
#         my_indices = np.where( my_parameter_sweep_results[:,9,4]>0.2)
        
        print('the average increase is')
        print(np.mean(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the minimal increase is')
        print(np.min(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the maximal increase is')
        print(np.max(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the median increase is')
        print(np.median(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        accepted_samples = my_posterior_samples[my_indices]
        accepted_results = my_posterior_results[my_indices]
        print('number of accepted samples is')
        print(len(accepted_samples))
        print('the minimal period is')
        print(np.min(accepted_results[:,2]))
        print(np.max(accepted_results[:,2]))
        print(np.median(accepted_results[:,2]))
        print('the coherence statistics are')
        print(np.min(accepted_results[:,3]))
        print(np.max(accepted_results[:,3]))
        print(np.median(accepted_results[:,3]))
        print('the mrna statistics are')
        print(np.min(accepted_results[:,4]))
        print(np.max(accepted_results[:,4]))
        print(np.median(accepted_results[:,4]))
        pairplot = hes5.plot_posterior_distributions(accepted_samples)
        pairplot.diag_axes[0].set_ylim(0,200)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','pairplot_repression_threshold_decreases_period.pdf'))
 
    def xest_investigate_where_protein_degradation_changes_period(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data', 'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
        
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
#
        print('number of existing samples is')
        print(len(my_posterior_samples))
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_protein_degradation_rate.npy'))
        
#         my_other_indices = np.where(my_parameter_sweep_results[:,9,4]>0.1)
        my_other_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4]>0.1, my_parameter_sweep_results[:,9,3]<300 ))

        print('number of qualifiying samples is')
        print(len(my_other_indices[0]))

#         my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,4,3]>my_parameter_sweep_results[:,9,3],
#                                             my_parameter_sweep_results[:,9,4]>0.1))
        my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,4,3]>my_parameter_sweep_results[:,9,3],
                              np.logical_and(my_parameter_sweep_results[:,9,4]>0.1, my_parameter_sweep_results[:,9,3]<300 )))

#         my_indices = np.where( my_parameter_sweep_results[:,9,4]>0.2)
        
        print('the average increase is')
        print(np.mean(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the minimal increase is')
        print(np.min(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the maximal increase is')
        print(np.max(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        print('the median increase is')
        print(np.median(my_parameter_sweep_results[:,4,3]/my_parameter_sweep_results[:,9,3]))
        accepted_samples = my_posterior_samples[my_indices]
        accepted_results = my_posterior_results[my_indices]
        print('number of accepted samples is')
        print(len(accepted_samples))
        print('the minimal period is')
        print(np.min(accepted_results[:,2]))
        print(np.max(accepted_results[:,2]))
        print(np.median(accepted_results[:,2]))
        print('the coherence statistics are')
        print(np.min(accepted_results[:,3]))
        print(np.max(accepted_results[:,3]))
        print(np.median(accepted_results[:,3]))
        print('the mrna statistics are')
        print(np.min(accepted_results[:,4]))
        print(np.max(accepted_results[:,4]))
        print(np.median(accepted_results[:,4]))
        pairplot = hes5.plot_posterior_distributions(accepted_samples)
        pairplot.diag_axes[0].set_ylim(0,200)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','pairplot_protein_degradation_decreases_period.pdf'))
 
    def xest_investigate_where_protein_degradation_decreases_coherence(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data', 'sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
        
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
#
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_protein_degradation_rate.npy'))
        
        my_indices = np.where( my_parameter_sweep_results[:,9,3]<300)
        my_parameter_sweep_results = my_parameter_sweep_results[my_indices]
        my_posterior_samples = my_posterior_samples[my_indices]
        print('total number of samples is:')
        print(len(my_parameter_sweep_results))

#         my_indices = np.where(my_parameter_sweep_results[:,14,4]<my_parameter_sweep_results[:,9,4])
#         my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,14,4]<my_parameter_sweep_results[:,9,4],
#                                             my_parameter_sweep_results[:,9,4]>0.1))
        my_indices = np.where(np.logical_not(np.logical_and(my_parameter_sweep_results[:,14,4]>my_parameter_sweep_results[:,9,4],
                                            my_parameter_sweep_results[:,9,3]>my_parameter_sweep_results[:,14,3])))
#         my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,14,4]>my_parameter_sweep_results[:,9,4],
#                                             my_parameter_sweep_results[:,9,3]>my_parameter_sweep_results[:,14,3]))
#         my_indices = np.where(np.logical_and(my_parameter_sweep_results[:,4,4]<my_parameter_sweep_results[:,9,4],
#                                             my_parameter_sweep_results[:,9,3]<my_parameter_sweep_results[:,4,3]))
#         my_indices = np.where( my_parameter_sweep_results[:,9,4]>0.2)
#         my_indices = np.where( my_parameter_sweep_results[:,9,3]<300)
        
        accepted_samples = my_posterior_samples[my_indices]
        accepted_results = my_posterior_results[my_indices]
        print('number of accepted samples is')
        print(len(accepted_samples))
        print('likelihood is')
        print(len(accepted_samples)/float(len(my_parameter_sweep_results)))
        print('the minimal period is')
        print(np.min(accepted_results[:,2]))
        print(np.max(accepted_results[:,2]))
        print(np.median(accepted_results[:,2]))
        print('the coherence statistics are')
        print(np.min(accepted_results[:,3]))
        print(np.max(accepted_results[:,3]))
        print(np.median(accepted_results[:,3]))
        print('the mrna statistics are')
        print(np.min(accepted_results[:,4]))
        print(np.max(accepted_results[:,4]))
        print(np.median(accepted_results[:,4]))
        pairplot = hes5.plot_posterior_distributions(accepted_samples)
        pairplot.diag_axes[0].set_ylim(0,200)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','pairplot_protein_degradation_decreases_coherence.pdf'))
              
    def xest_plot_model_prediction(self):

#         my_posterior_samples = prior_samples[accepted_indices]
        
#         accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
#         my_posterior_samples = prior_samples[other_accepted_indices]
        
#         model_results = model_results[other_accepted_indices]

#         parameter_names = ['basal_transcription_rate',
#                             'translation_rate',
#                             'repression_threshold',
#                             'time_delay',
#                             'mRNA_degradation_rate',
#                             'protein_degradation_rate',
#                             'hill_coefficient']

        parameter_names = ['protein_degradation_rate']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'logarithmic_relative_sweeps_' + parameter_name + '.npy'))
            
#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
            
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
# #                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
# #                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
# #                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
# #                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
# #                                                        my_parameter_sweep_results[:,9,3] < 400)))))
        
            increase_indices = np.where(my_parameter_sweep_results[:,9,3] < 300)
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] > 0.1,
#                                         np.logical_or(my_parameter_sweep_results[:,4,4]>0.25,
#                                                       my_parameter_sweep_results[:,14,4]>0.25))) 
#                                                       my_parameter_sweep_results[:,
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  4] > 0.2,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  4] > 0.2)))
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  3] < 400,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  3] < 400)))
#                                                                                   4] > 0.15))
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
#                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
#                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
#                                                        my_parameter_sweep_results[:,9,3] < 400)))))
 
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
 
    def xest_plot_relative_parameter_variation_for_nancy(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
#         my_posterior_samples = prior_samples[other_accepted_indices]
        
#         model_results = model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

#         parameter_names = ['repression_threshold']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'logarithmic_relative_sweeps_' + parameter_name + '.npy'))
            
            other_accepted_indices = np.where(my_parameter_sweep_results[:,9,3]<300)
            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
#             decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                                      my_parameter_sweep_results[:,9,3] > 300),
#                                         np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
#                                                        my_parameter_sweep_results[:,4,4] > 0.2)))

#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 600,
#             increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                                      my_parameter_sweep_results[:,9,3] > 300),
#                                         np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
#                                                        my_parameter_sweep_results[:,14,4] > 0.2)))
            
#             decrease_results = my_parameter_sweep_results[decrease_indices]
#             increase_results = my_parameter_sweep_results[increase_indices]
#             my_parameter_sweep_results = np.vstack((decrease_results, increase_results))


#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
# #                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
# #                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
# #                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
# #                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
# #                                                        my_parameter_sweep_results[:,9,3] < 400)))))
        
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.25,
#                                         np.logical_or(my_parameter_sweep_results[:,4,4]>0.25,
#                                                       my_parameter_sweep_results[:,14,4]>0.25))) 
#                                                       my_parameter_sweep_results[:,
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  4] > 0.2,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  4] > 0.2)))
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  3] < 400,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  3] < 400)))
#                                                                                   4] > 0.15))
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
#                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
#                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
#                                                        my_parameter_sweep_results[:,9,3] < 400)))))
 
#             my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
#             my_sweep_parameters = my_posterior_samples[increase_indices]
            
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
#             this_axis.set_ylim(0,0.5)
#             this_axis.set_ylim(0,0.25)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','logarithmic_relative_sweep_for_nancy_' + parameter_name + '.pdf'))
 
    def xest_plot_bayes_factor_differences(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        number_of_absolute_samples = len(accepted_indices[0])

        parameter_names = [ 'basal_transcription_rate',
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
        
        for plotting_option in ['boxes', 'samples_only']:
            statistic_names = [ 'basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'hill_coefficient' ]

            my_figure = plt.figure( figsize = (4.5, 7.5) )
            
            decrease_ratios = dict()
            increase_ratios = dict()
            for parameter_name in parameter_names:
                my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'narrowed_relative_sweeps_' + 
                                                              parameter_name + '.npy'))
 
                number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                        my_parameter_sweep_results[:,9,4] < 0.05))[0])

                decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
                                                                        my_parameter_sweep_results[:,9,3] > 600),
                                            np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                            my_parameter_sweep_results[:,4,4] > 0.1)))

                decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)

                increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
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
#             sorted_bars = -np.log(sorted_bars)
            sorted_bars/= np.sum(sorted_bars)
            sorted_bars = sorted_bars[::-1]

            my_figure.add_subplot(611)
            plt.bar(all_positions, sorted_bars)
            sorted_labels.reverse()
            plt.xticks( all_positions + 0.4 , 
                        sorted_labels,
                        rotation = 30,
                        fontsize = 3,
                        horizontalalignment = 'right')
            plt.xlim(all_positions[0] - 0.5,)
            plt.gca().locator_params(axis='y', tight = True, nbins=5)
            plt.ylim(0,sorted_bars[0]*1.2)
            plt.ylabel('Likelihood')
            
            x_positions = dict()
            for index, position in enumerate(all_positions):
                x_positions[sorted_labels[index]] = position

            for statistic_index, statistic_name in enumerate(statistic_names):
                statistic_values = dict()
                for parameter_name in parameter_names:
                    my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                                  'data',
                                                                  'narrowed_relative_sweeps_' + 
                                                                  parameter_name + '.npy'))
 
                    number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                            my_parameter_sweep_results[:,9,4] < 0.05))[0])
                    
                    decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
                                                                            my_parameter_sweep_results[:,9,3] > 600),
                                                np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                                my_parameter_sweep_results[:,4,4] > 0.1)))

                    decrease_statistic_values = my_posterior_samples[decrease_indices, statistic_index]
                    statistic_values[x_labels[parameter_name] + ' down'] = decrease_statistic_values

                    increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
                                                                            my_parameter_sweep_results[:,9,3] > 600),
                                                np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                               my_parameter_sweep_results[:,14,4] > 0.1)))
                    
                    increase_statistic_values = my_posterior_samples[increase_indices, statistic_index]
                    statistic_values[x_labels[parameter_name] + ' up'] = increase_statistic_values

                my_figure.add_subplot(6,1,statistic_index + 2)
                # loop through all parameters up and down combinations
                for label in sorted_labels:
                # get x position and statistic values
                    this_x_position = x_positions[label]
                    these_statistic_values = statistic_values[label]
                    # make x values
                    these_x_positions = this_x_position - 0.2 + 0.4*np.random.rand(len(these_statistic_values.flatten()))
                    if statistic_name.startswith('repression_threshold'):
                        these_statistic_values/=10000
                    # scatter
                    if plotting_option == 'samples_only':
                        print 'hello'
                        plt.scatter(these_x_positions, these_statistic_values, 
                                    marker = '.', lw = 0, color = 'dimgrey', 
#                                     alpha = 0.1, 
                                    s = 1,
                                    zorder = 0)
                    elif plotting_option == 'boxes':
                        print 'hell1'
                        plt.boxplot(these_statistic_values, positions = [this_x_position],
                                    sym = '', widths = [0.7])
                    plt.gca().set_rasterization_zorder(1)
                if statistic_name.startswith('translation_rate') or statistic_name.startswith('basal_transcription_rate'):
                    plt.gca().set_yscale("log")
                else:
                    plt.gca().locator_params(axis='y', tight = True, nbins=5)
#                 plt.gca().xaxis.set_ticks([])
#                 plt.gca().xaxis.set_ticklabels([])
                plt.xticks( all_positions, 
                            sorted_labels,
                            rotation = 30,
                            fontsize = 3,
                            horizontalalignment = 'right')
                plt.xlim(all_positions[0] - 0.5,)
#                 plt.ylim(0,sorted_bars[-1]*1.2)
                plt.ylabel(x_labels[statistic_name], fontsize = 5)

            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                             'output',
                                             'likelihood_plot_extension_' + plotting_option + '.pdf'), dpi = 400)
            
            for reference_point in ['start', 'end']:
                my_figure = plt.figure( figsize = (4.5, 7.5) )
                if reference_point == 'start':
                    statistic_names = ['Expression', 'rel. std.', 'Period', 'Coherence', '<mRNA>']
                else:
                    statistic_names = ['Expression', 'rel. std.', 'Period', 'Coherence']
                for statistic_index, statistic_name in enumerate(statistic_names):
                    statistic_values = dict()
                    for parameter_name in parameter_names:
                        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                                      'data',
                                                                      'narrowed_relative_sweeps_' + 
                                                                      parameter_name + '.npy'))
 
                        number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                                my_parameter_sweep_results[:,9,4] < 0.05))[0])
                        
                        decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
                                                                                my_parameter_sweep_results[:,9,3] > 600),
                                                    np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                                    my_parameter_sweep_results[:,4,4] > 0.1)))

                        if reference_point == 'start':
                            decrease_statistic_values = accepted_model_results[decrease_indices, statistic_index]
                        elif reference_point == 'end':
                            decrease_statistic_values = my_parameter_sweep_results[decrease_indices, 4, statistic_index+1]
                        statistic_values[x_labels[parameter_name] + ' down'] = decrease_statistic_values

                        increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.05,
                                                                                my_parameter_sweep_results[:,9,3] > 600),
                                                    np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                                   my_parameter_sweep_results[:,14,4] > 0.1)))
                        
                        if reference_point == 'start':
                            increase_statistic_values = accepted_model_results[increase_indices, statistic_index]
                        elif reference_point == 'end':
                            increase_statistic_values = my_parameter_sweep_results[increase_indices, 14, statistic_index+1]
                        statistic_values[x_labels[parameter_name] + ' up'] = increase_statistic_values

                    my_figure.add_subplot(5,1,statistic_index + 1)
                    # loop through all parameters up and down combinations
                    for label in sorted_labels:
                    # get x position and statistic values
                        this_x_position = x_positions[label]
                        these_statistic_values = statistic_values[label]
                        # make x values
                        these_x_positions = this_x_position - 0.2 + 0.4*np.random.rand(len(these_statistic_values.flatten()))
                        # scatter
                        if statistic_name.startswith('Expression'):
                            these_statistic_values/=10000
                        if plotting_option == 'samples_only':
                            plt.scatter(these_x_positions, these_statistic_values, 
                                        marker = '.', lw = 0, color = 'dimgrey', 
#                                         alpha = 0.1, 
                                        s = 1,
                                        zorder = 0)
                        elif plotting_option == 'boxes':
                            plt.boxplot(these_statistic_values, positions = [this_x_position],
                                        sym = '', widths = [0.7])
                    plt.gca().set_rasterization_zorder(1)
                    if statistic_name == 'Period':
                        pass
#                         plt.ylim(200,350)
                    plt.gca().locator_params(axis='y', tight = True, nbins=5)
#                     plt.gca().xaxis.set_ticks([])
#                     plt.gca().xaxis.set_ticklabels([])
                    plt.xticks( all_positions, 
                                sorted_labels,
                                rotation = 30,
                                fontsize = 3,
                                horizontalalignment = 'right')
                    plt.xlim(all_positions[0] - 0.5,)
#                     plt.ylim(0,sorted_bars[-1]*1.2)
                    plt.ylabel(statistic_name, fontsize = 5)

                my_figure.tight_layout()
                my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                                 'output',
                                                 'likelihood_plot_extension_' + plotting_option + '_' + reference_point + '.pdf'), dpi = 400)

    def xest_plot_bayes_factors_for_models(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        new_accepted_indices = np.where(my_posterior_samples[:,1]<10)
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

        reference_parameters = dict()
        
        decrease_ratios = dict()
        increase_ratios = dict()
        bardata = []
        ## Increase in coherence
        for parameter_name in parameter_names:
            print 'investigating ' + parameter_name
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print 'these accepted base samples are'
#             print len(np.where(my_parameter_sweep_results[:,9,4] < 0.1)[0])
#             number_of_absolute_samples = len(np.where(my_parameter_sweep_results[:,9,3] > 1000)[0])
#             print number_of_absolute_samples
            
            decrease_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                                        my_parameter_sweep_results[:,4,4] > 0.1))
            
#             decrease_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 1000,
#                                                        my_parameter_sweep_results[:,4,3] < 300))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)

            increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                                        my_parameter_sweep_results[:,14,4] > 0.1))

#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 1000,
#                                                        my_parameter_sweep_results[:,14,3] < 300))
            
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
        plt.xticks( all_positions + 0.4 , 
                    sorted_labels,
                    rotation = 30,
                    fontsize = 3,
                    horizontalalignment = 'right')
        plt.xlim(all_positions[0] - 0.5,)
        plt.gca().locator_params(axis='y', tight = True, nbins=5)
        plt.ylabel('Likelihood')
        plt.ylim(0,sorted_bars[-1]*1.2)
#         plt.ylim(0,1)

        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output',
                                     'likelihood_plot_coherence_increase_from_0.1.pdf'))

        ## Increase in coherence
        for parameter_name in parameter_names:
            print 'investigating ' + parameter_name
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'data',
                                                          'logarithmic_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
#             my_reducing_indices = np.where(my_posterior_samples[:,0]<13)
#             my_parameter_sweep_results = my_parameter_sweep_results[new_accepted_indices]
            print 'these accepted base samples are'
#             print len(np.where(my_parameter_sweep_results[:,9,4] < 0.1)[0])
#             number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 300,
#                                                                     my_parameter_sweep_results[:,9,4] < 0.1))[0])
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
                                                                    my_parameter_sweep_results[:,9,4] < 0.1))[0])
#             number_of_absolute_samples = len(np.where( my_parameter_sweep_results[:,9,4] < 0.2)[0])
#             number_of_absolute_samples = len(np.where( my_parameter_sweep_results[:,9,4] < 0.2)[0])
            print number_of_absolute_samples
            
#             decrease_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                         my_parameter_sweep_results[:,4,4] > 0.1))
            
#             decrease_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 600,
#             decrease_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.2,
#                                         np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
#                                                         my_parameter_sweep_results[:,4,4] > 0.1)))
            decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
                                                        my_parameter_sweep_results[:,4,4] > 0.1)))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)

#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 600,

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
                                                                    my_parameter_sweep_results[:,9,3] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
                                                        my_parameter_sweep_results[:,14,4] > 0.1)))

#             increase_indices = np.where(np.logical_and( my_parameter_sweep_results[:,9,3] < 0.2,
#                                         np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
#                                                         my_parameter_sweep_results[:,14,4] > 0.1)))

#                 np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                         np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
#                                                        my_parameter_sweep_results[:,14,4] > 0.1)))
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                         my_parameter_sweep_results[:,14,4] > 0.1))

#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,3] > 1000,
#                                                        my_parameter_sweep_results[:,14,3] < 300))
            
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
#         sorted_bars = -np.log(sorted_bars)
        sorted_bars/= np.sum(sorted_bars)

        my_figure = plt.figure( figsize = (4.5, 1.5) )
        plt.bar(all_positions, sorted_bars[::-1])
        sorted_labels.reverse()
        plt.xticks( all_positions + 0.4 , 
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
                                     'likelihood_plot_period_decrease_below_six_hours_and_coherence_above_0.1.pdf'))

    def xest_plot_relative_parameter_variation_coherence_increase_logarithmic(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
#         my_posterior_samples = prior_samples[other_accepted_indices]
        
#         model_results = model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

#         parameter_names = ['repression_threshold']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'data',
                                                              'logarithmic_relative_sweeps_' + parameter_name + '.npy'))
            
#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
            
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
# #                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
# #                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
# #                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
# #                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
# #                                                        my_parameter_sweep_results[:,9,3] < 400)))))
        
            increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                        np.logical_or(my_parameter_sweep_results[:,4,4]>0.1,
                                                      my_parameter_sweep_results[:,14,4]>0.1))) 
#                                                       my_parameter_sweep_results[:,
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  4] > 0.2,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  4] > 0.2)))
#                                         np.logical_or(my_parameter_sweep_results[:,
#                                                                                  4,
#                                                                                  3] < 400,
#                                                       my_parameter_sweep_results[:,
#                                                                                  14,
#                                                                                  3] < 400)))
#                                                                                   4] > 0.15))
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
#                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
#                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
#                                                        my_parameter_sweep_results[:,9,3] < 400)))))
 
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
#             my_sweep_parameters = my_posterior_samples[increase_indices]
            
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
#             this_axis.set_ylim(0,1)
#             this_axis.set_ylim(0,0.5)
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','logarithmic_relative_sweep_coherence_increases_large_' + parameter_name + '.pdf'))
 
    def xest_plot_relative_parameter_variation_coherence_increase_low_transcription(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
#         my_posterior_samples = prior_samples[other_accepted_indices]
        
#         model_results = model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

#         parameter_names = ['repression_threshold']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'hill_relative_sweeps_low_transcription' + parameter_name + '.npy'))
            
#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
            
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
# #                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
# #                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
# #                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
# #                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
# #                                                        my_parameter_sweep_results[:,9,3] < 400)))))
        
            increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                                       my_parameter_sweep_results[:,
                                                                                  reference_indices[parameter_name],
                                                                                3] < 400))
#                                                                                   4] > 0.15))
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
#                                                        my_parameter_sweep_results[:,9,4]*8,
#                                         np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                        my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
#                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05))))
#                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
#                                                        my_parameter_sweep_results[:,9,3] < 400)))))
 
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
            my_sweep_parameters = my_posterior_samples[increase_indices]
            
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
#             this_axis.set_ylim(0,1)
#             this_axis.set_ylim(0,0.5)
            this_axis.set_ylim(0,0.25)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_relative_sweep_low_transcription_coherence_increases_' + parameter_name + '.pdf'))
 
    def xest_plot_pairplot_for_coherence_increase(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
#         other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
#         my_posterior_samples = my_posterior_samples[other_accepted_indices]
#         
#         model_results = accepted_model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
#         reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['mRNA_degradation_rate'] = 5
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'hill_relative_sweeps_low_transcription' + parameter_name + '.npy'))
            
#             my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
            increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
                                            my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
                                        np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 
                                                       my_parameter_sweep_results[:,9,4]*1.2,
                                        np.logical_and(my_parameter_sweep_results[:,9,4] < 0.2,
                                                       my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4] > 0.2))))
#   
#             increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,10,4] < 
#                                             my_parameter_sweep_results[:,reference_indices[parameter_name] -1 ,4],
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2,
#                                         np.logical_and(my_parameter_sweep_results[:,reference_indices[parameter_name] -1,3] < 400,
# #                                                        my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.15))))
#                                         np.logical_and(my_parameter_sweep_results[:,20 - reference_indices[parameter_name] -1,4] < 0.05,
#                                                        my_parameter_sweep_results[:,9,3] < 400)))))
#         
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]
            
            my_sweep_parameters = my_posterior_samples[increase_indices]
 
            try:
                my_pairplot = hes5.plot_posterior_distributions(my_sweep_parameters)
                my_pairplot.axes[-1,0].set_xlim(0,4)
#                 plt.style.use('classic')
                my_pairplot.savefig(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'pairplot_low_transcription_coherence_increase_' + parameter_name + '.pdf'))
            except:
                print('could not pairplot ' + parameter_name)

    def xest_plot_traces_for_repression_threshold_decrease(self):
        
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'hill_relative_sweeps_low_transcription' + 
                                                          'repression_threshold.npy'))
            
        increase_indices = np.where(np.logical_and(my_parameter_sweep_results[:,9,4] < 
                                        my_parameter_sweep_results[:,4 ,4],
#                                         my_parameter_sweep_results[:,reference_indices[parameter_name] -1,4] > 0.2))
                                    np.logical_and(my_parameter_sweep_results[:,4,4] > 
                                                   my_parameter_sweep_results[:,9,4]*8,
                                    np.logical_and(my_parameter_sweep_results[:,9,4] < 0.1,
                                                   my_parameter_sweep_results[:,4,4] > 0.2))))
#       
        my_posterior_results = model_results[increase_indices]
        my_posterior_samples = my_posterior_samples[increase_indices]

        number_of_traces = 10
        figuresize = (6,9)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(3, 3 )
        
        repression_threshold_percentages = [1.0, 0.5, 0.25] #corresponds to 1.0 and 0.5
        
        y_limits = [[4,8],
                    [2,6],
                    [0,4]]

        for figure_row_index, repression_threshold_percentage in enumerate(repression_threshold_percentages):
            for parameter_index in range(3):
                this_double_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec = outer_grid[figure_row_index*3 + parameter_index],
                        height_ratios= [number_of_traces, 1])
                this_inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                        subplot_spec=this_double_grid[0], hspace=0.0)
                this_parameter = my_posterior_samples[parameter_index]
                this_results = my_posterior_results[parameter_index]

                _, these_traces = hes5.generate_multiple_langevin_trajectories(number_of_trajectories = 200,
                                                             duration = 1500*5,
                                                             repression_threshold = this_parameter[2]*
                                                                                    repression_threshold_percentage,
                                                             mRNA_degradation_rate = np.log(2)/30.0,
                                                             protein_degradation_rate = np.log(2)/90,
                                                             transcription_delay = this_parameter[3],
                                                             basal_transcription_rate = this_parameter[0],
                                                             translation_rate = this_parameter[1],
                                                             initial_mRNA = 10,
                                                             initial_protein = this_parameter[2]*
                                                                                    repression_threshold_percentage,
                                                             equilibration_time = 1000)
                this_power_spectrum, this_coherence, _ = hes5.calculate_power_spectrum_of_trajectories(these_traces)

                for subplot_index in range(number_of_traces):
                    this_axis = plt.Subplot(my_figure, this_inner_grid[subplot_index])
                    my_figure.add_subplot(this_axis)
                    this_trace = hes5.generate_langevin_trajectory(
                                                             duration = 1500,
                                                             repression_threshold = this_parameter[2]*
                                                                                    repression_threshold_percentage,
                                                             mRNA_degradation_rate = np.log(2)/30.0,
                                                             protein_degradation_rate = np.log(2)/90,
                                                             transcription_delay = this_parameter[3],
                                                             basal_transcription_rate = this_parameter[0],
                                                             translation_rate = this_parameter[1],
                                                             initial_mRNA = 10,
                                                             initial_protein = this_parameter[2]*
                                                                                    repression_threshold_percentage,
                                                             hill_coefficient = this_parameter[4],
                                                             equilibration_time = 1000)

                    plt.plot(this_trace[:,0], this_trace[:,2]/1e4)
                    plt.ylim(y_limits[figure_row_index])
#                     this_axis.locator_params(axis='y', tight = True, nbins=1)
#                     this_axis.locator_params(axis='y', nbins=2)
                    this_axis.locator_params(axis='x', tight = True, nbins=3)
                    plt.yticks([])
                    this_axis.tick_params(axis='both', length = 1)
                    if subplot_index == 0:
                        plt.title('Coherence: ' + '{:.2f}'.format(this_coherence) + 
                                  r', $\alpha_m =$ ' + '{:.2f}'.format(this_parameter[0]) +
                                  '\n' + r'$\alpha_p =$ ' + '{:.2f}'.format(this_parameter[1]) + 
                                  r', $p_0 = $ ' + '{:.2f}'.format(this_parameter[2]) + 
                                  r', $\tau = $ ' + '{:.2f}'.format(this_parameter[3]),
                                  fontsize = 5)
                    if subplot_index < number_of_traces - 1:
                        this_axis.xaxis.set_ticklabels([])
                    if parameter_index !=0: 
                        this_axis.yaxis.set_ticklabels([])
                    if parameter_index == 0 and subplot_index == 5:
                        plt.ylabel('Expression/1e4', labelpad = 15)
                plt.xlabel('Time [min]', labelpad = 2)
                plt.yticks(y_limits[figure_row_index])
                this_axis = plt.Subplot(my_figure, this_double_grid[1])
                my_figure.add_subplot(this_axis)
                plt.xlabel('Frequency [1/min]', labelpad = 2)
                plt.plot(this_power_spectrum[:,0], this_power_spectrum[:,1])
                this_axis.locator_params(axis='x', tight = True, nbins=3)
                this_axis.tick_params(axis='both', length = 1)
                if parameter_index == 0:
                    plt.ylabel('Power', labelpad = 15)
                max_index = np.argmax(this_power_spectrum[:,1])
                max_power_frequency = this_power_spectrum[max_index,0]
                left_frequency = max_power_frequency*0.9
                right_frequency = max_power_frequency*1.1
                plt.axvline(left_frequency, color = 'black')
                plt.axvline(right_frequency, color = 'black')
                plt.xlim(0.0,0.01)
                plt.yticks([])
                plt.axhline(3)

        plt.figtext(0.5,0.98,r'Repression threshold ratio at 1.0', fontsize = 10, 
                rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                horizontalalignment = 'center')
        plt.figtext(0.5,0.65,r'Repression threshold ratio at 0.5', fontsize = 10, 
                rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                horizontalalignment = 'center')
        plt.figtext(0.5,0.32,r'Repression threshold ratio at 0.25', fontsize = 10, 
                rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                horizontalalignment = 'center')
        plt.tight_layout()
        my_figure.subplots_adjust(hspace = 0.5)
            
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','repression_threshold_decrease.pdf'))

    def xest_plot_low_transcription_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill_low_transcription')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_parameter_sweeps_hill_low_transcription' + parameter_name + '.npy'))
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','low_transcription_sweep_' + parameter_name + '.pdf'))
 
    def xest_plot_heterozygous_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_heterozygous_sweeps_' + parameter_name + '.npy'))
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','heterozygous_sweep_' + parameter_name + '.pdf'))
 
    def xest_make_relative_multiple_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories, 
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_relative_parameter_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_relative_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_relative_parameter_sweeps_' + parameter_name + '.npy'))
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_relative_sweep_' + parameter_name + '.pdf'))
 
    def xest_plot_relative_parameter_variation_differently(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_relative_parameter_sweeps_' + parameter_name + '.npy'))
            
            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
        
            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_relative_sweep_low_coherence_' + parameter_name + '.pdf'))
 
    def xest_plot_relative_parameter_variation_coherence_increase(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]
        other_accepted_indices = np.where(accepted_model_results[:,3] < 0.2)
        
        model_results = model_results[other_accepted_indices]

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'rel. Transcription rate'
        x_labels['translation_rate'] = 'rel. Translation rate'
        x_labels['repression_threshold'] = 'rel. Repression threshold' 
        x_labels['time_delay'] = 'rel. Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'rel. mRNA degradation'
        x_labels['protein_degradation_rate'] = 'rel. Protein degradation'
        x_labels['hill_coefficient'] = 'rel. Hill coefficient'

        reference_indices = dict()
        reference_indices['basal_transcription_rate'] = 5
        reference_indices['translation_rate'] = 5
        reference_indices['repression_threshold'] = 5 
        reference_indices['time_delay'] = 15
        reference_indices['mRNA_degradation_rate'] = 15
        reference_indices['protein_degradation_rate'] = 15
        reference_indices['hill_coefficient'] = 15

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_relative_parameter_sweeps_' + parameter_name + '.npy'))
            
            my_parameter_sweep_results = my_parameter_sweep_results[other_accepted_indices]
            
            increase_indices = np.where(my_parameter_sweep_results[:,10,4] < 
                                        my_parameter_sweep_results[:,reference_indices[parameter_name],4])
        
            my_parameter_sweep_results = my_parameter_sweep_results[increase_indices]

            my_figure = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,0.4)
            
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_relative_sweep_coherence_increases_' + parameter_name + '.pdf'))
 
    def xest_plot_all_parameter_variation(self):

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_200reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<10))))) #transcription_rate
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

#         my_parameter_sweep_results = hes5.conduct_protein_degradation_sweep_at_parameters(my_posterior_samples,
#                                                                                           number_of_parameter_points,
#                                                                                           number_of_trajectories)

        parameter_names = ['basal_transcription_rate',
                           'translation_rate',
                           'repression_threshold',
                           'time_delay',
                           'mRNA_degradation_rate',
                           'protein_degradation_rate',
                           'hill_coefficient']
 
        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'Transcription rate [1/min]'
        x_labels['translation_rate'] = 'Translation rate [1/min]'
        x_labels['repression_threshold'] = 'Repression threshold/1e4' 
        x_labels['time_delay'] = 'Transcription delay [min]'
        x_labels['mRNA_degradation_rate'] = 'mRNA degradation [1/min]'
        x_labels['protein_degradation_rate'] = 'Protein degradation [1/min]'
        x_labels['hill_coefficient'] = 'Hill coefficient'

        parameter_indices = dict()
        parameter_indices['basal_transcription_rate'] = 0
        parameter_indices['translation_rate'] = 1
        parameter_indices['repression_threshold'] = 2
        parameter_indices['time_delay'] = 3
        
        reference_parameters = dict()

        for parameter_name in parameter_names:
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                              'output',
                                                              'all_parameter_sweeps_' + parameter_name + '.npy'))
        
            if parameter_name == 'repression_threshold':
                my_parameter_sweep_results[:,:,0] /= 10000
                my_posterior_samples[:,2] /= 10000

            my_figure = plt.figure( figsize = (6.5, 1.5) )
            my_figure2 = plt.figure( figsize = (6.5, 1.5) )
            this_axis = my_figure.add_subplot(131)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,3], color ='black', alpha = 0.05)
            if parameter_name == 'protein_degradation_rate':
                this_axis.axvline( np.log(2)/90 )
            elif parameter_name == 'mRNA_degradation_rate':
                this_axis.axvline( np.log(2)/30 )
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure2.add_subplot(131)
            for results_index, results_table in enumerate(my_parameter_sweep_results):
                if parameter_name == 'protein_degradation_rate':
                    this_parameter = np.log(2)/90
                elif parameter_name == 'mRNA_degradation_rate':
                    this_parameter = np.log(2)/30
                elif parameter_name == 'hill_coefficient':
                    this_parameter = 5
                else:
                    this_parameter = my_posterior_samples[results_index, parameter_indices[parameter_name]]
                this_axis.plot(results_table[:,0] - this_parameter,
                         results_table[:,3], color ='black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(r'$\Delta$' + x_labels[parameter_name])
            this_axis.set_ylabel('Period [min]')
            this_axis.set_ylim(0,700)

            this_axis = my_figure.add_subplot(132)
            for results_table in my_parameter_sweep_results:
                this_axis.plot(results_table[:,0],
                         results_table[:,4], color = 'black', alpha = 0.05)
            if parameter_name == 'protein_degradation_rate':
                this_axis.axvline( np.log(2)/90 )
            elif parameter_name == 'mRNA_degradation_rate':
                this_axis.axvline( np.log(2)/30 )
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
            
            this_axis = my_figure2.add_subplot(132)
            for results_index, results_table in enumerate(my_parameter_sweep_results):
                if parameter_name == 'protein_degradation_rate':
                    this_parameter = np.log(2)/90
                elif parameter_name == 'mRNA_degradation_rate':
                    this_parameter = np.log(2)/30
                elif parameter_name == 'hill_coefficient':
                    this_parameter = 5
                else:
                    this_parameter = my_posterior_samples[results_index, parameter_indices[parameter_name]]
                this_axis.plot(results_table[:,0] - this_parameter,
                         results_table[:,4], color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_xlabel(r'$\Delta$' + x_labels[parameter_name])
            this_axis.set_ylabel('Coherence')
            this_axis.set_ylim(0,1)
 
            this_axis = my_figure.add_subplot(133)
            for results_table in my_parameter_sweep_results:
                this_axis.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            if parameter_name == 'protein_degradation_rate':
                this_axis.axvline( np.log(2)/90 )
            elif parameter_name == 'mRNA_degradation_rate':
                this_axis.axvline( np.log(2)/30 )
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
            my_figure.tight_layout()
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_sweep_' + parameter_name + '.pdf'))

            this_axis = my_figure2.add_subplot(133)
            for results_index, results_table in enumerate(my_parameter_sweep_results):
                if parameter_name == 'protein_degradation_rate':
                    this_parameter = np.log(2)/90
                elif parameter_name == 'mRNA_degradation_rate':
                    this_parameter = np.log(2)/30
                elif parameter_name == 'hill_coefficient':
                    this_parameter = 5
                else:
                    this_parameter = my_posterior_samples[results_index, parameter_indices[parameter_name]]
                this_axis.errorbar(results_table[:,0] - this_parameter,
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000*results_table[:,1],
                             color = 'black', alpha = 0.05)
            this_axis.locator_params(axis='x', tight = True, nbins=4)
            this_axis.set_ylim(0,15)
            this_axis.set_xlabel(r'$\Delta$' + x_labels[parameter_name])
            this_axis.set_ylabel('Expression/1e4')
 
            my_figure2.tight_layout()
            my_figure2.savefig(os.path.join(os.path.dirname(__file__),
                                     'output','multiple_centered_sweep_' + parameter_name + '.pdf'))
 
    def xest_approximate_power_spectrum_numerically(self):
        number_of_traces = 100
        repetition_number = 1
        
        trace_and_repetition_numbers = np.array([[200,1], 
                                                 [200,2], 
                                                 [200,3], 
#                                                  [100,5]]) 
                                                [200,4], 
                                                [200,5], 
                                                [200,6], 
                                                [200,7], 
                                                [200,8], 
                                                [200,9], 
                                                [1000,10], 
                                                [1000,20]])
        
        power_spectra = []
        smoothened_power_spectra = []
        coherences = np.zeros(trace_and_repetition_numbers.shape[0])
        periods = np.zeros(trace_and_repetition_numbers.shape[0])
        index = 0
        for number_of_traces, repetition_number in trace_and_repetition_numbers:
            print(number_of_traces)
            print(repetition_number)
            these_mrna_traces, these_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_number,
                                                         repression_threshold = 31400,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         basal_transcription_rate = 11,
                                                         translation_rate = 29,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 31400,
                                                         equilibration_time = 1000)

            this_power_spectrum, this_coherence, this_period = hes5.calculate_power_spectrum_of_trajectories(these_protein_trajectories)
            
            this_smoothened_power_spectrum = hes5.smoothen_power_spectrum(this_power_spectrum)

            power_spectra.append(this_power_spectrum)
            smoothened_power_spectra.append(this_smoothened_power_spectrum)
            coherences[index] = this_coherence
            periods[index] = this_period
            index += 1 

#         theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
#                                                      basal_transcription_rate = 11,
#                                                      translation_rate = 29,
#                                                      repression_threshold = 31400,
#                                                      transcription_delay = 29,
#                                                      mRNA_degradation_rate = np.log(2)/30,
#                                                      hill_coefficient = 5,
#                                                      protein_degradation_rate = np.log(2)/90)
#         
        figuresize = (6,2.5)
        my_figure = plt.figure(figsize = figuresize)
        my_figure.add_subplot(131)
        for counter, power_spectrum in enumerate(power_spectra):
            if counter == 4:
                plt.plot(power_spectrum[:,0],power_spectrum[:,1]+counter*200, color = 'green', alpha = 0.8)
            elif counter > 6:
                plt.plot(power_spectrum[:,0],power_spectrum[:,1]+counter*200, color = 'blue')
            else:
                plt.plot(power_spectrum[:,0],power_spectrum[:,1]+counter*200, color = 'black', alpha = 0.8)
#         plt.plot(theoretical_power_spectrum[:,0],theoretical_power_spectrum[:,1], color = 'blue', alpha = 0.8)
            plt.plot(smoothened_power_spectra[counter][:,0],
                     smoothened_power_spectra[counter][:,1]+counter*200, color = 'grey', alpha = 0.8)
        plt.xlim(0.000,0.01)
#         plt.ylim(0,100)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Frequency')
        plt.ylabel('Probability + offset')

        my_figure.add_subplot(132)
        plt.plot(trace_and_repetition_numbers[:,1], coherences, color = 'black')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.axvline(5, color = 'green')
        plt.ylim(0,0.5)
        plt.xlabel('Trace length [1500min]')
        plt.ylabel('Coherence estimate')

        my_figure.add_subplot(133)
        plt.plot(trace_and_repetition_numbers[:,1], periods, color = 'black')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.ylim(200,300)
        plt.axvline(5, color = 'green')
        plt.xlabel('Trace length [1500min]')
        plt.ylabel('Period estimate')
       
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','Coherence_measure_test.pdf'))

    def xest_plot_multiple_parameter_variation_differently(self):
        number_of_parameter_points = 20
        number_of_trajectories = 100

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,3]>20)))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]

        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'multiple_degradation_sweep_results.npy'))
        
        #Find all traces where coherence is zero at the entry log(2)/90,
        #i.e. close to at entry three of the results
        
        small_coherence_indices, = np.where(my_parameter_sweep_results[:,3,4] < 0.1)
        small_coherence_index = small_coherence_indices[0]
        this_small_coherence_parameter = my_posterior_samples[small_coherence_index]
        extra_large_coherence_indices, = np.where(my_parameter_sweep_results[:,2,4]>0.1)
        extra_large_coherence_index = extra_large_coherence_indices[0]
        this_extra_large_coherence_parameter = my_posterior_samples[extra_large_coherence_index]
        large_coherence_indices = list( set( range(my_parameter_sweep_results.shape[0])) - 
                                        set( small_coherence_indices).union( 
                                             set( extra_large_coherence_indices ))
                                       ) 
        large_coherence_index = large_coherence_indices[3]
        this_large_coherence_parameter = my_posterior_samples[large_coherence_index]

        my_figure = plt.figure( figsize = (6.5, 3.5) )
        my_figure.add_subplot(231)
        for parameter_index, results_table in enumerate(my_parameter_sweep_results):
            if parameter_index in small_coherence_indices:
                plt.plot(results_table[:,0],
                         results_table[:,3], color ='purple', alpha = 0.1)
            elif parameter_index in extra_large_coherence_indices:
                plt.plot(results_table[:,0],
                         results_table[:,3], color ='blue', alpha = 0.1)
            else:
                plt.plot(results_table[:,0],
                         results_table[:,3], color ='green', alpha = 0.01)
        plt.plot(my_parameter_sweep_results[large_coherence_index,:,0],
                 my_parameter_sweep_results[large_coherence_index,:,3], color ='green')
        plt.plot(my_parameter_sweep_results[extra_large_coherence_index,:,0],
                 my_parameter_sweep_results[extra_large_coherence_index,:,3], color ='blue')
        plt.plot(my_parameter_sweep_results[small_coherence_index,:,0],
                 my_parameter_sweep_results[small_coherence_index,:,3], color ='purple')
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(232)
        for parameter_index, results_table in enumerate(my_parameter_sweep_results):
            if parameter_index in small_coherence_indices:
                plt.plot(results_table[:,0],
                         results_table[:,4], color = 'purple', alpha = 0.1)
            elif parameter_index in extra_large_coherence_indices:
                plt.plot(results_table[:,0],
                         results_table[:,4], color = 'blue', alpha = 0.1)
            else:
                plt.plot(results_table[:,0],
                         results_table[:,4], color = 'green', alpha = 0.01)
        plt.plot(my_parameter_sweep_results[large_coherence_index,:,0],
                 my_parameter_sweep_results[large_coherence_index,:,4], color = 'green')
        plt.plot(my_parameter_sweep_results[extra_large_coherence_index,:,0],
                 my_parameter_sweep_results[extra_large_coherence_index,:,4], color = 'blue')
        plt.plot(my_parameter_sweep_results[small_coherence_index,:,0],
                 my_parameter_sweep_results[small_coherence_index,:,4], color = 'purple')
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Coherence')
        plt.ylim(0,1)
        
        my_figure.add_subplot(233)
        for parameter_index, results_table in enumerate(my_parameter_sweep_results):
            if parameter_index in small_coherence_indices:
                plt.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000,
                             color = 'purple', alpha = 0.1)
            elif parameter_index in extra_large_coherence_indices:
                plt.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000,
                             color = 'blue', alpha = 0.1)
            else:
                plt.errorbar(results_table[:,0],
                             results_table[:,1]/10000,
                             yerr = results_table[:,2]/10000,
                             color = 'green', alpha = 0.01)
        plt.errorbar(my_parameter_sweep_results[extra_large_coherence_index,:,0],
                     my_parameter_sweep_results[extra_large_coherence_index,:,1]/10000,
                     yerr = my_parameter_sweep_results[large_coherence_index,:,2]/10000,
                     color = 'blue')
        plt.errorbar(my_parameter_sweep_results[large_coherence_index,:,0],
                     my_parameter_sweep_results[large_coherence_index,:,1]/10000,
                     yerr = my_parameter_sweep_results[large_coherence_index,:,2]/10000,
                     color = 'green')
        plt.errorbar(my_parameter_sweep_results[small_coherence_index,:,0],
                     my_parameter_sweep_results[small_coherence_index,:,1]/10000,
                     yerr = my_parameter_sweep_results[small_coherence_index,:,2]/10000,
                     color = 'purple')
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.ylim(0,15)
        plt.xlabel('Hes5 degradation [1/min]')
        plt.ylabel('Expression/1e4')
        
        my_figure.add_subplot(234)
        this_small_trace = hes5.generate_langevin_trajectory( duration = 1500, 
                                        repression_threshold = this_small_coherence_parameter[2],
                                        basal_transcription_rate = this_small_coherence_parameter[0],
                                        translation_rate = this_small_coherence_parameter[1],
                                        initial_mRNA = 10.0,
                                        transcription_delay = this_small_coherence_parameter[3],
                                        initial_protein = this_small_coherence_parameter[2],
                                        equilibration_time = 1000.0 )

        plt.plot( this_small_trace[:,0]/100, this_small_trace[:,2]/10000, color = 'purple')
        plt.text(0.1, 0.2, r'$\alpha_m =$ ' + '{:.2f}'.format(this_small_coherence_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(this_small_coherence_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(this_small_coherence_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(this_small_coherence_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
        plt.ylim(0,10)
        plt.xlabel('Time [100min]')
        plt.ylabel('Expression/1e4')

        my_figure.add_subplot(235)
        this_large_trace = hes5.generate_langevin_trajectory( duration = 1500, 
                                        repression_threshold = this_large_coherence_parameter[2],
                                        basal_transcription_rate = this_large_coherence_parameter[0],
                                        translation_rate = this_large_coherence_parameter[1],
                                        initial_mRNA = 10.0,
                                        transcription_delay = this_large_coherence_parameter[3],
                                        initial_protein = this_large_coherence_parameter[2],
                                        equilibration_time = 1000.0 )
        plt.plot( this_large_trace[:,0]/100, this_large_trace[:,2]/10000, color = 'green')
        plt.text(0.1, 0.2, r'$\alpha_m =$ ' + '{:.2f}'.format(this_large_coherence_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(this_large_coherence_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(this_large_coherence_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(this_large_coherence_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
        plt.ylim(0,10)
        plt.xlabel('Time [100min]')
        plt.ylabel('Expression/1e4')

        my_figure.add_subplot(236)
        this_extra_large_trace = hes5.generate_langevin_trajectory( duration = 1500, 
                                        repression_threshold = this_extra_large_coherence_parameter[2],
                                        basal_transcription_rate = this_extra_large_coherence_parameter[0],
                                        translation_rate = this_extra_large_coherence_parameter[1],
                                        initial_mRNA = 10.0,
                                        transcription_delay = this_extra_large_coherence_parameter[3],
                                        initial_protein = this_extra_large_coherence_parameter[2],
                                        equilibration_time = 1000.0 )
        plt.plot( this_extra_large_trace[:,0]/100, this_extra_large_trace[:,2]/10000, color = 'blue')
        plt.text(0.1, 0.2, r'$\alpha_m =$ ' + '{:.2f}'.format(this_extra_large_coherence_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(this_extra_large_coherence_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(this_extra_large_coherence_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(this_extra_large_coherence_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
        plt.ylim(0,10)
        plt.xlabel('Time [100min]')
        plt.ylabel('Expression/1e4')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','multiple_degradation_sweep_examples.pdf'))
        
    def xest_validation_at_low_transcription_rates(self):
        # pick three parameter values with low mrna and plot example mrna, example protein, and power spectrum
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 200
        repetition_factor = 5
        
#         number_of_traces = 4
#         repetition_factor = 1

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                    model_results[:,1]>0.05)))) #standard deviation
#                                                     model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<2))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        my_results = model_results[accepted_indices]
        lowest_indices = np.argsort(my_posterior_samples[:,0])
#         import pdb; pdb.set_trace()

        ##
        # first_samples
        ##
        first_parameter = my_posterior_samples[lowest_indices[0]]
        print first_parameter
        first_mRNA_trajectories, first_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = first_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = first_parameter[3],
                                                         basal_transcription_rate = first_parameter[0],
                                                         translation_rate = first_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = first_parameter[2],
                                                         equilibration_time = 1000,
                                                         synchronize = False )

        first_langevin_mRNA_trajectories, first_langevin_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = first_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = first_parameter[3],
                                                         basal_transcription_rate = first_parameter[0],
                                                         translation_rate = first_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = first_parameter[2],
                                                         equilibration_time = 1000)

        first_theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                         repression_threshold = first_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = first_parameter[3],
                                                         basal_transcription_rate = first_parameter[0],
                                                         translation_rate = first_parameter[1]
                                                         )

        first_theoretical_coherence, first_theoretical_period = hes5.calculate_coherence_and_period_of_power_spectrum(first_theoretical_power_spectrum)

        first_power_spectrum, first_coherence, first_period = hes5.calculate_power_spectrum_of_trajectories(first_protein_trajectories)
        first_langevin_power_spectrum, first_langevin_coherence, first_langevin_period = hes5.calculate_power_spectrum_of_trajectories(first_langevin_protein_trajectories)
        
        figuresize = (6,10)
        my_figure = plt.figure(figsize = figuresize)
        my_figure.add_subplot(521)
        plt.plot( first_mRNA_trajectories[:,0],
                  first_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*10', color = 'black',
                  lw = 0.5 )
        plt.plot( first_protein_trajectories[:,0],
                  first_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.plot( first_langevin_mRNA_trajectories[:,0],
                  first_langevin_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*10', color = 'green',
                  lw = 0.5 )
        plt.plot( first_langevin_protein_trajectories[:,0],
                  first_langevin_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'green', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.text(0.1, 0.3, r'$\alpha_m =$ ' + '{:.2f}'.format(first_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(first_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(first_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(first_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time [min]')
        plt.ylabel('Copy number [1e4]')
        plt.ylim(0,9)
        plt.xlim(0,1500)
#         plt.legend()
       
        my_figure.add_subplot(522)
        for trajectory in first_protein_trajectories[:,1:].transpose():
            compound_trajectory = np.vstack((first_protein_trajectories[:,0],trajectory)).transpose()
            this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(compound_trajectory)
            plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1], color = 'black', alpha = 0.01)
        plt.plot(first_power_spectrum[:,0],
                 first_power_spectrum[:,1], color = 'black')
        plt.plot(first_langevin_power_spectrum[:,0],
                 first_langevin_power_spectrum[:,1], color = 'green')
        plt.plot(first_theoretical_power_spectrum[:,0],
                 first_theoretical_power_spectrum[:,1], color = 'blue')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence:\n' + "{:.2f}".format(first_coherence) + ' '
                 "{:.2f}".format(first_langevin_coherence) + ' ' +
                 "{:.2f}".format(first_theoretical_coherence) +
                 '\nPeriod:\n' +  "{:.2f}".format(first_period)  + ' '
                 "{:.2f}".format(first_langevin_period)  + ' ' +
                 "{:.2f}".format(first_theoretical_period), 
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes,
                 fontsize = 5)

        ##
        # Hes5 samples
        ##
        second_parameter = my_posterior_samples[lowest_indices[1]]
        second_mRNA_trajectories, second_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = second_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = second_parameter[3],
                                                         basal_transcription_rate = second_parameter[0],
                                                         translation_rate = second_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = second_parameter[2],
                                                         equilibration_time = 1000,
                                                         synchronize = False )

        second_langevin_mRNA_trajectories, second_langevin_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = second_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = second_parameter[3],
                                                         basal_transcription_rate = second_parameter[0],
                                                         translation_rate = second_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = second_parameter[2],
                                                         equilibration_time = 1000)

        second_theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                         repression_threshold = second_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = second_parameter[3],
                                                         basal_transcription_rate = second_parameter[0],
                                                         translation_rate = second_parameter[1]
                                                         )

        second_theoretical_coherence, second_theoretical_period = hes5.calculate_coherence_and_period_of_power_spectrum(second_theoretical_power_spectrum)

        second_power_spectrum, second_coherence, second_period = hes5.calculate_power_spectrum_of_trajectories(second_protein_trajectories)
        second_langevin_power_spectrum, second_langevin_coherence, second_langevin_period = hes5.calculate_power_spectrum_of_trajectories(second_langevin_protein_trajectories)
        
        my_figure.add_subplot(523)
        mrna_example, = plt.plot( second_mRNA_trajectories[:,0],
                  second_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'black',
                  lw = 0.5 )
        protein_example, = plt.plot( second_protein_trajectories[:,0],
                  second_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.plot( second_langevin_mRNA_trajectories[:,0],
                  second_langevin_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'green',
                  lw = 0.5 )
        plt.plot( second_langevin_protein_trajectories[:,0],
                  second_langevin_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'green', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.text(0.1, 0.25, r'$\alpha_m =$ ' + '{:.2f}'.format(second_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(second_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(second_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(second_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
        plt.xlabel('Time [min]')
        plt.ylabel('Copy number [1e4]')
        plt.ylim(0,9)
        plt.xlim(0,1500)
#         plt.legend()

        my_figure.add_subplot(524)
        for trajectory in second_protein_trajectories[:,1:].transpose():
            compound_trajectory = np.vstack((second_protein_trajectories[:,0],trajectory)).transpose()
            this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(compound_trajectory)
            plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1], color = 'black', alpha = 0.01)
        plt.plot(second_power_spectrum[:,0],
                 second_power_spectrum[:,1], color = 'black')
        plt.plot(second_langevin_power_spectrum[:,0],
                 second_langevin_power_spectrum[:,1], color = 'green')
        plt.plot(second_theoretical_power_spectrum[:,0],
                 second_theoretical_power_spectrum[:,1], color = 'blue')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence:\n' + "{:.2f}".format(second_coherence) + ' '
                 "{:.2f}".format(second_langevin_coherence) + ' ' +
                 "{:.2f}".format(second_theoretical_coherence) +
                 '\nPeriod:\n' +  "{:.2f}".format(second_period)  + ' '
                 "{:.2f}".format(second_langevin_period)  + ' ' +
                 "{:.2f}".format(second_theoretical_period), 
                 verticalalignment='top', 
                 fontsize = 5, 
                 horizontalalignment='right',
                 transform=plt.gca().transAxes)
        
        ##
        # third example
        ##
        # generate the random samples:
        third_parameter = my_posterior_samples[lowest_indices[2]]
        third_mRNA_trajectories, third_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = third_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = third_parameter[3],
                                                         basal_transcription_rate = third_parameter[0],
                                                         translation_rate = third_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = third_parameter[2],
                                                         equilibration_time = 1000,
                                                         synchronize = False )

        third_langevin_mRNA_trajectories, third_langevin_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = third_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = third_parameter[3],
                                                         basal_transcription_rate = third_parameter[0],
                                                         translation_rate = third_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = third_parameter[2],
                                                         equilibration_time = 1000)

        third_theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                         repression_threshold = third_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = third_parameter[3],
                                                         basal_transcription_rate = third_parameter[0],
                                                         translation_rate = third_parameter[1]
                                                         )

        third_theoretical_coherence, third_theoretical_period = hes5.calculate_coherence_and_period_of_power_spectrum(third_theoretical_power_spectrum)
        third_power_spectrum, third_coherence, third_period = hes5.calculate_power_spectrum_of_trajectories(third_protein_trajectories)
        third_langevin_power_spectrum, third_langevin_coherence, third_langevin_period = hes5.calculate_power_spectrum_of_trajectories(third_langevin_protein_trajectories)
 
        my_figure.add_subplot(525)
        mrna_example, = plt.plot( third_mRNA_trajectories[:,0],
                  third_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'black',
                  lw = 0.5 )
        protein_example, = plt.plot( third_protein_trajectories[:,0],
                  third_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.plot( third_langevin_mRNA_trajectories[:,0],
                  third_langevin_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'green',
                  lw = 0.5 )
        plt.plot( third_langevin_protein_trajectories[:,0],
                  third_langevin_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'green', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time [min]')
        plt.ylabel('Copy number [1e4]')
        plt.xlim(0,1500)
        plt.ylim(0,9)
        plt.text(0.1, 0.22, r'$\alpha_m =$ ' + '{:.2f}'.format(third_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(third_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(third_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(third_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
#         plt.legend()

        my_figure.add_subplot(526)
        for trajectory in third_protein_trajectories[:,1:].transpose():
            compound_trajectory = np.vstack((third_protein_trajectories[:,0],trajectory)).transpose()
            this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(compound_trajectory)
            plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1], color = 'black', alpha = 0.01)
        plt.plot(third_power_spectrum[:,0],
                 third_power_spectrum[:,1], color = 'black')
        plt.plot(third_langevin_power_spectrum[:,0],
                 third_langevin_power_spectrum[:,1], color = 'green')
        plt.plot(third_theoretical_power_spectrum[:,0],
                 third_theoretical_power_spectrum[:,1], color = 'blue')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence:\n' + "{:.2f}".format(third_coherence) + ' '
                 "{:.2f}".format(third_langevin_coherence) + ' ' +
                 "{:.2f}".format(third_theoretical_coherence) +
                 '\nPeriod:\n' +  "{:.2f}".format(third_period)  + ' '
                 "{:.2f}".format(third_langevin_period)  + ' ' +
                 "{:.2f}".format(third_theoretical_period), 
                 fontsize = 5, 
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes)
 
        ##
        # fourth example
        ##
        # generate the random samples:
#         accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
#                                     np.logical_and(model_results[:,0]<65000, #cell_number
#                                     np.logical_and(model_results[:,1]<0.15, #standard deviation
#                                                     model_results[:,1]>0.05)))) #standard deviation
#                                                     model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
#                                                     prior_samples[:,0]<0.5))))) #time_delay

#         my_posterior_samples = prior_samples[accepted_indices]

        fourth_parameter = my_posterior_samples[lowest_indices[3]]

        fourth_mRNA_trajectories, fourth_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = fourth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = fourth_parameter[3],
                                                         basal_transcription_rate = fourth_parameter[0],
                                                         translation_rate = fourth_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = fourth_parameter[2],
                                                         equilibration_time = 1000,
                                                         synchronize = False )

        fourth_langevin_mRNA_trajectories, fourth_langevin_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = fourth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = fourth_parameter[3],
                                                         basal_transcription_rate = fourth_parameter[0],
                                                         translation_rate = fourth_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = fourth_parameter[2],
                                                         equilibration_time = 1000)

        fourth_theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                         repression_threshold = fourth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = fourth_parameter[3],
                                                         basal_transcription_rate = fourth_parameter[0],
                                                         translation_rate = fourth_parameter[1]
                                                         )

        fourth_theoretical_coherence, fourth_theoretical_period = hes5.calculate_coherence_and_period_of_power_spectrum(fourth_theoretical_power_spectrum)
        fourth_power_spectrum, fourth_coherence, fourth_period = hes5.calculate_power_spectrum_of_trajectories(fourth_protein_trajectories)
        fourth_langevin_power_spectrum, fourth_langevin_coherence, fourth_langevin_period = hes5.calculate_power_spectrum_of_trajectories(fourth_langevin_protein_trajectories)
 
        my_figure.add_subplot(527)
        mrna_example, = plt.plot( fourth_mRNA_trajectories[:,0],
                  fourth_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'black',
                  lw = 0.5 )
        protein_example, = plt.plot( fourth_protein_trajectories[:,0],
                  fourth_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.plot( fourth_langevin_mRNA_trajectories[:,0],
                  fourth_langevin_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'green',
                  lw = 0.5 )
        plt.plot( fourth_langevin_protein_trajectories[:,0],
                  fourth_langevin_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'green', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time [min]')
        plt.ylabel('Copy number [1e4]')
        plt.xlim(0,1500)
        plt.ylim(0,9)
        plt.text(0.1, 0.34, r'$\alpha_m =$ ' + '{:.2f}'.format(fourth_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(fourth_parameter[1]) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(fourth_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(fourth_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
#         plt.legend()

        my_figure.add_subplot(528)
        for trajectory in fourth_protein_trajectories[:,1:].transpose():
            compound_trajectory = np.vstack((fourth_protein_trajectories[:,0],trajectory)).transpose()
            this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(compound_trajectory)
            plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1], color = 'black', alpha = 0.01)
        plt.plot(fourth_power_spectrum[:,0],
                 fourth_power_spectrum[:,1], color = 'black')
        plt.plot(fourth_langevin_power_spectrum[:,0],
                 fourth_langevin_power_spectrum[:,1], color = 'green')
        plt.plot(fourth_theoretical_power_spectrum[:,0],
                 fourth_theoretical_power_spectrum[:,1], color = 'blue')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence:\n' + "{:.2f}".format(fourth_coherence) + ' '
                 "{:.2f}".format(fourth_langevin_coherence) + ' ' +
                 "{:.2f}".format(fourth_theoretical_coherence) +
                 '\nPeriod:\n' +  "{:.2f}".format(fourth_period)  + ' '
                 "{:.2f}".format(fourth_langevin_period)  + ' ' +
                 "{:.2f}".format(fourth_theoretical_period), 
                 fontsize = 5, 
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes)

        ##
        # fifth example
        ##
        # generate the random samples:
        fifth_parameter = np.array([11.0,29.0,31400.0,29.0])
#         fifth_parameter = my_posterior_samples[2]
        fifth_mRNA_trajectories, fifth_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = fifth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = fifth_parameter[3],
                                                         basal_transcription_rate = fifth_parameter[0],
                                                         translation_rate = fifth_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = fifth_parameter[2],
                                                         equilibration_time = 1000,
                                                         synchronize = False )

        fifth_langevin_mRNA_trajectories, fifth_langevin_protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = number_of_traces,
                                                                                        duration = 1500*repetition_factor,
                                                         repression_threshold = fifth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = fifth_parameter[3],
                                                         basal_transcription_rate = fifth_parameter[0],
                                                         translation_rate = fifth_parameter[1],
                                                         initial_mRNA = 10,
                                                         initial_protein = fifth_parameter[2],
                                                         equilibration_time = 1000)

        fifth_theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                         repression_threshold = fifth_parameter[2],
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = fifth_parameter[3],
                                                         basal_transcription_rate = fifth_parameter[0],
                                                         translation_rate = fifth_parameter[1]
                                                         )

        fifth_theoretical_coherence, fifth_theoretical_period = hes5.calculate_coherence_and_period_of_power_spectrum(fifth_theoretical_power_spectrum)

        fifth_power_spectrum, fifth_coherence, fifth_period = hes5.calculate_power_spectrum_of_trajectories(fifth_protein_trajectories)
        fifth_langevin_power_spectrum, fifth_langevin_coherence, fifth_langevin_period = hes5.calculate_power_spectrum_of_trajectories(fifth_langevin_protein_trajectories)
 
        my_figure.add_subplot(529)
        mrna_example, = plt.plot( fifth_mRNA_trajectories[:,0],
                  fifth_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'black',
                  lw = 0.5 )
        protein_example, = plt.plot( fifth_protein_trajectories[:,0],
                  fifth_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.plot( fifth_langevin_mRNA_trajectories[:,0],
                  fifth_langevin_mRNA_trajectories[:,1]*0.1, label = 'mRNA example*1000', color = 'green',
                  lw = 0.5 )
        plt.plot( fifth_langevin_protein_trajectories[:,0],
                  fifth_langevin_protein_trajectories[:,1]/10000, label = 'Protein example', color = 'green', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time [min]')
        plt.ylabel('Copy number [1e4]')
        plt.xlim(0,1500)
        plt.ylim(0,15)
        plt.text(0.1, 0.01, r'$\alpha_m =$ ' + '{:.2f}'.format(fifth_parameter[0]) +
                           r', $\alpha_p =$ ' + '{:.2f}'.format(fifth_parameter[1]) + 
                           r', $\mu_p =$ ' + '{:.2f}'.format(0.03) + 
                           '\n' + r'$p_0 = $ ' + '{:.2f}'.format(fifth_parameter[2]) + 
                           r', $\tau = $ ' + '{:.2f}'.format(fifth_parameter[3]),
                           fontsize = 5,
                           transform=plt.gca().transAxes)
#         plt.legend()

        my_figure.add_subplot(5,2,10)
        for trajectory in fifth_protein_trajectories[:,1:].transpose():
            compound_trajectory = np.vstack((fifth_protein_trajectories[:,0],trajectory)).transpose()
            this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(compound_trajectory)
            plt.plot(this_power_spectrum[:,0],this_power_spectrum[:,1], color = 'black', alpha = 0.01)
        plt.plot(fifth_power_spectrum[:,0],
                 fifth_power_spectrum[:,1], color = 'black')
        plt.plot(fifth_langevin_power_spectrum[:,0],
                 fifth_langevin_power_spectrum[:,1], color = 'green')
        plt.plot(fifth_theoretical_power_spectrum[:,0],
                 fifth_theoretical_power_spectrum[:,1], color = 'blue')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         import pdb; pdb.set_trace()
        plt.text(0.05, 0.95, 'Coherence:\n' + "{:.2f}".format(fifth_coherence) + ' '
                 "{:.2f}".format(fifth_langevin_coherence) + ' ' +
                 "{:.2f}".format(fifth_theoretical_coherence) +
                 '\nPeriod:\n' +  "{:.2f}".format(fifth_period)  + ' '
                 "{:.2f}".format(fifth_langevin_period)  + ' ' +
                 "{:.2f}".format(fifth_theoretical_period), 
                 fontsize = 5, 
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes)

        plt.tight_layout()
        my_figure.legend((mrna_example, protein_example), 
                        ('mRNA*1000', 'Protein'), 
                        loc = 'upper left', ncol = 2, fontsize = 10 )
        plt.figtext(0.5, 0.975, 'Langevin', horizontalalignment='left', color = 'green')
        plt.figtext(0.65, 0.975, 'Gillespie', horizontalalignment='left', color = 'black')
        plt.figtext(0.8, 0.975, 'LNA', horizontalalignment='left', color = 'blue')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)

#         plt.subplots_adjust(top = 0.9, hspace = 0.7)
        plt.subplots_adjust(top = 0.95)
       
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','logarithmic_low_transcription_rate_langevin_validation.pdf'))

    def xest_visualise_model_regimes(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 10
        figuresize = (6,2.5)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(1, 3 )

        coherence_bands = [[0,0.1],
                           [0.2,0.3],
                           [0.5,0.6]]
        
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
            
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','model_visualisation.pdf'))

    def xest_visualise_different_coherences(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_logarithmic')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        figure_coherence_bands = np.array([[[0.0,0.1],        
                                            [0.1,0.2],        
                                            [0.2,0.3]],        
                                           [[0.3,0.4],        
                                            [0.5,0.6],        
                                            [0.6,0.7]]])
        number_of_traces = 10
        for figure_index in range(2):
            figuresize = (6,9)
            my_figure = plt.figure(figsize = figuresize)
            outer_grid = matplotlib.gridspec.GridSpec(3, 3 )

            coherence_bands = figure_coherence_bands[figure_index]
#                                [0.2,0.3],
#                                [0.3,0.4]]

            for coherence_index, coherence_band in enumerate(coherence_bands):
                accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                            np.logical_and(model_results[:,0]<65000, #cell_number
                                            np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                            np.logical_and(model_results[:,1]>0.05,
                                            np.logical_and(model_results[:,3]>coherence_band[0],
                                                           model_results[:,3]<coherence_band[1]))))))

                my_posterior_results = model_results[accepted_indices]
                my_posterior_samples = prior_samples[accepted_indices]
       
                for parameter_index in range(3):
                    this_double_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec = outer_grid[coherence_index*3 + parameter_index],
                            height_ratios= [number_of_traces, 1])
                    this_inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                            subplot_spec=this_double_grid[0], hspace=0.0)
                    try:
                        this_parameter = my_posterior_samples[parameter_index]
                        this_results = my_posterior_results[parameter_index]
                    except IndexError:
                        this_parameter = my_posterior_samples[0]
                        this_results = my_posterior_results[0]

                    for subplot_index in range(number_of_traces):
                        this_axis = plt.Subplot(my_figure, this_inner_grid[subplot_index])
                        my_figure.add_subplot(this_axis)
                        this_trace = hes5.generate_langevin_trajectory(
                                                                 duration = 1500,
                                                                 repression_threshold = this_parameter[2],
                                                                 hill_coefficient = this_parameter[4],
                                                                 mRNA_degradation_rate = np.log(2)/30.0,
                                                                 protein_degradation_rate = np.log(2)/90,
                                                                 transcription_delay = this_parameter[3],
                                                                 basal_transcription_rate = this_parameter[0],
                                                                 translation_rate = this_parameter[1],
                                                                 initial_mRNA = 10,
                                                                 initial_protein = this_parameter[2],
                                                                 equilibration_time = 1000)
                        plt.plot(this_trace[:,0], this_trace[:,2]/1e4)
                        plt.ylim(4,8)
#                         this_axis.locator_params(axis='y', tight = True, nbins=1)
#                         this_axis.locator_params(axis='y', nbins=2)
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
                        if subplot_index < number_of_traces - 1:
                            this_axis.xaxis.set_ticklabels([])
                        if parameter_index !=0: 
                            this_axis.yaxis.set_ticklabels([])
                        if parameter_index == 0 and subplot_index == 5:
                            plt.ylabel('Expression/1e4', labelpad = 15)
                    plt.xlabel('Time [min]', labelpad = 2)
                    plt.yticks([4,8])
                    this_axis = plt.Subplot(my_figure, this_double_grid[1])
                    my_figure.add_subplot(this_axis)
                    plt.xlabel('Frequency [1/min]', labelpad = 2)
                    _, these_traces = hes5.generate_multiple_langevin_trajectories(number_of_trajectories = 200,
                                                                 duration = 1500*5,
                                                                 repression_threshold = this_parameter[2],
                                                                 hill_coefficient = this_parameter[4],
                                                                 mRNA_degradation_rate = np.log(2)/30.0,
                                                                 protein_degradation_rate = np.log(2)/90,
                                                                 transcription_delay = this_parameter[3],
                                                                 basal_transcription_rate = this_parameter[0],
                                                                 translation_rate = this_parameter[1],
                                                                 initial_mRNA = 10,
                                                                 initial_protein = this_parameter[2],
                                                                 equilibration_time = 1000)
                    this_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_traces)
                    smoothened_power_spectrum = hes5.smoothen_power_spectrum(this_power_spectrum)
                    plt.plot(this_power_spectrum[:,0], this_power_spectrum[:,1])
                    plt.plot(smoothened_power_spectrum[:,0], smoothened_power_spectrum[:,1], 
                             color = 'grey' )
                    this_axis.locator_params(axis='x', tight = True, nbins=3)
                    this_axis.tick_params(axis='both', length = 1)
                    if parameter_index == 0:
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

            plt.figtext(0.5,0.98,r'Coherence $\in$ ' 
                        + np.array_str(coherence_bands[0,:], precision=1), fontsize = 10, 
                    rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                    horizontalalignment = 'center')
            plt.figtext(0.5,0.65,r'Coherence $\in$ ' 
                        + np.array_str(coherence_bands[1,:], precision=1), fontsize = 10, 
                    rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                    horizontalalignment = 'center')
            plt.figtext(0.5,0.32,r'Coherence $\in$ ' 
                        + np.array_str(coherence_bands[2,:], precision=1), fontsize = 10, 
                    rotation = 'horizontal', verticalalignment = 'bottom', multialignment = 'center',
                    horizontalalignment = 'center')
            plt.tight_layout()
            my_figure.subplots_adjust(hspace = 0.5)
                
            my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','coherence_visualisation_' + 
                                           str(figure_index) + '.pdf'))

    def xest_visualise_heterozygous_model(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_hill')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05))))

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        
        figuresize = (7,9)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(2, 3 )
        
        for parameter_index in range(6):
            this_parameter_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec = outer_grid[parameter_index], height_ratios=[3,1],
                            hspace=0.3)

            this_parameter = my_posterior_samples[parameter_index]
            this_model_results = my_model_results[parameter_index]

            _, these_traces_1, _, these_traces_2 = hes5.generate_multiple_heterozygous_langevin_trajectories(
                                                                 number_of_trajectories = 200,
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
            these_compound_traces = np.zeros_like(these_traces_1)
            these_compound_traces[:,0] = these_traces_1[:,0]
            these_compound_traces[:,1:] = these_traces_1[:,1:]+these_traces_2[:,1:]
            this_compound_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_compound_traces)
            this_allele_1_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_traces_1)
            this_allele_2_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_traces_2)
           
            this_upper_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1,
                            subplot_spec = this_parameter_grid[0])
            for realisation_index in range(3):
                this_realisation_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1,
                            subplot_spec = this_upper_grid[realisation_index], hspace = 0.0) 

                this_homozygous_axis = plt.Subplot(my_figure, this_realisation_grid[0])
                my_figure.add_subplot(this_homozygous_axis)
                this_homozygous_axis.plot(these_compound_traces[:,0], 
                                         these_compound_traces[:,realisation_index + 1]/10000,
                                         color = 'black')
                plt.xlim(0,1500)
                plt.ylim(4.5,8.5)
                this_homozygous_axis.locator_params(axis='x', tight = True, nbins=3)
                this_homozygous_axis.locator_params(axis='y', tight = True, nbins=3)
                plt.gca().xaxis.set_ticklabels([])
                if realisation_index == 0:
                    plt.title('Coherence: ' + '{:.2f}'.format(this_model_results[3]) + 
                              r', $\alpha_m =$ ' + '{:.2f}'.format(this_parameter[0]) +
                              r', $n =$ ' + '{:.2f}'.format(this_parameter[4]) +
                              '\n' + r'$\alpha_p =$ ' + '{:.2f}'.format(this_parameter[1]) + 
                              r', $p_0 = $ ' + '{:.2f}'.format(this_parameter[2]) + 
                              r', $\tau = $ ' + '{:.2f}'.format(this_parameter[3]),
                              fontsize = 7)
                
                this_allele_axis_1 = plt.Subplot(my_figure, this_realisation_grid[1])
                my_figure.add_subplot(this_allele_axis_1)
                this_allele_axis_1.plot(these_traces_1[:,0], 
                                       these_traces_1[:,realisation_index + 1]/10000,
                                       color = 'green')
                plt.ylim(1.5,5.5)
                plt.xlim(0,1500)
                this_allele_axis_1.locator_params(axis='x', tight = True, nbins=3)
                this_allele_axis_1.locator_params(axis='y', tight = True, nbins=3)
                plt.gca().xaxis.set_ticklabels([])
                if parameter_index in [0,3] and realisation_index == 1:
                    plt.ylabel("Expression/1e4")
 

                this_allele_axis_2 = plt.Subplot(my_figure, this_realisation_grid[2])
                my_figure.add_subplot(this_allele_axis_2)
                this_allele_axis_2.plot(these_traces_2[:,0], 
                                       these_traces_2[:,realisation_index + 1]/10000,
                                       color = 'blue')
                plt.ylim(1.5,5.5)
                plt.xlim(0,1500)
                this_allele_axis_2.locator_params(axis='x', tight = True, nbins=3)
                this_allele_axis_2.locator_params(axis='y', tight = True, nbins=3)
              
                if realisation_index != 2: 
                    plt.gca().xaxis.set_ticklabels([])
                else:
                    plt.xlabel("Time [min]")    

            this_power_spectrum_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1,
                            subplot_spec = this_parameter_grid[1], hspace = 0.0)
            
            homozygous_power_spectrum_axis = plt.Subplot(my_figure,this_power_spectrum_grid[0])
            my_figure.add_subplot(homozygous_power_spectrum_axis)
            homozygous_power_spectrum_axis.plot(this_compound_power_spectrum[:,0],
                                                this_compound_power_spectrum[:,1],
                                                color = 'black')
            plt.xlim(0,0.01)
            homozygous_power_spectrum_axis.locator_params(axis='x', tight = True, nbins=3)
            homozygous_power_spectrum_axis.locator_params(axis='y', tight = True, nbins=3)
            plt.gca().xaxis.set_ticklabels([])
            plt.gca().yaxis.set_ticklabels([])
            if parameter_index == 0:
                plt.text(0.03, 0.6, 'Homozygous',
                           fontsize = 7,
                           color = 'black',
                           transform=plt.gca().transAxes)
           
            allele_1_power_spectrum_axis = plt.Subplot(my_figure, this_power_spectrum_grid[1])
                                                       
            my_figure.add_subplot(allele_1_power_spectrum_axis)
            allele_1_power_spectrum_axis.plot(this_allele_1_power_spectrum[:,0],
                                               this_allele_1_power_spectrum[:,1],
                                               color = 'green')
            plt.xlim(0,0.01)
            allele_1_power_spectrum_axis.locator_params(axis='x', tight = True, nbins=3)
            allele_1_power_spectrum_axis.locator_params(axis='y', tight = True, nbins=3)
            plt.gca().xaxis.set_ticklabels([])
            plt.gca().yaxis.set_ticklabels([])
            if parameter_index in [0, 3]:
                plt.ylabel("Power")
            if parameter_index == 0:
                plt.text(0.03, 0.6, 'Allele 1',
                           fontsize = 7,
                           color = 'green',
                           transform=plt.gca().transAxes)

            allele_2_power_spectrum_axis = plt.Subplot(my_figure, this_power_spectrum_grid[2])
            my_figure.add_subplot(allele_2_power_spectrum_axis)
            allele_2_power_spectrum_axis.plot(this_allele_2_power_spectrum[:,0],
                                               this_allele_2_power_spectrum[:,1],
                                               color = 'blue')
            plt.xlim(0,0.01)
            allele_2_power_spectrum_axis.locator_params(axis='x', tight = True, nbins=3)
            allele_2_power_spectrum_axis.locator_params(axis='y', tight = True, nbins=3)
            plt.gca().yaxis.set_ticklabels([])
            plt.xlabel("Frequency [1/min]")
            if parameter_index == 0:
                plt.text(0.03, 0.6, 'Allele 2',
                           fontsize = 7,
                           color = 'blue',
                           transform=plt.gca().transAxes)
 
        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','heterozygous_visualisation.pdf'))

    def xest_plot_ngn(self):
        excel_path = os.path.join(os.path.dirname(__file__),'data','Ngn2Staining_VS_VH5intensity_E10_5.xlsx')

        complete_excel_file = xlrd.open_workbook(excel_path)
        excel_sheet = complete_excel_file.sheet_by_index(3)
        hes5_column_values = excel_sheet.col_values(8)
        ngn2_column_values = excel_sheet.col_values(9)
        hes5_values = hes5_column_values[4:]
        ngn2_values = ngn2_column_values[4:]
        
        sns.set()
        font = {'size'   : 28}
        plt.rc('font', **font)

        plt.figure( figsize = (5,3) )
#         plt.scatter(hes5_values, ngn2_values, color = 'grey', lw = 0)
#         plt.scatter(hes5_values, ngn2_values, color = 'grey', lw = 0)
        sns.regplot(x=np.array(hes5_values), y=np.array(ngn2_values), fit_reg=False)
        plt.xlabel('Hes5 (a. u.)')
        plt.ylabel('Ngn2 (a. u.)')
        plt.xlim(0,)
        plt.ylim(0,)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','ngn_visualisation.pdf'))
        
    def xest_plot_distributions_for_poster(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
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
        transcription_rate_bins = np.linspace(-1,2,20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,2)
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,0.8)
        plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,2.3,20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,2.3)
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1,2], [r'$10^0$',r'$10^1$',r'$10^2$'])
        plt.xlabel("Translation rate \n [1/min]")
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                     rug = False)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.22)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                     bins = time_delay_bins)
        plt.gca().set_xlim(0,45)
        plt.gca().set_ylim(0,0.035)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel("Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                     rug = False)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(1,7)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_poster.pdf'))
 
    def xest_plot_prior_for_poster(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_logarithmic')
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
        transcription_rate_bins = np.linspace(-1,2,20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,2)
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,0.8)
        plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0,2.3,20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(0,2.3)
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1,2], [r'$10^0$',r'$10^1$',r'$10^2$'])
        plt.xlabel("Translation rate \n [1/min]")
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                     rug = False)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.22)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                     bins = time_delay_bins)
        plt.gca().set_xlim(0,45)
        plt.gca().set_ylim(0,0.035)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel("Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                     rug = False)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(1,7)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','prior_for_poster.pdf'))
 