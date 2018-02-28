import unittest
import os.path
import sys
import matplotlib as mpl
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_long_delay_different_prior.pdf'))
 
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        # plot distribution of accepted parameter samples
#         data_frame = pd.DataFrame( data = my_posterior_samples,
#                                columns= ['Transcription rate', 
#                                          'Translation rate', 
#                                          'Repression threshold', 
#                                          'Transcription delay'])
# #         pairplot = sns.pairplot(data_frame)
#         pairplot = sns.PairGrid(data_frame)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True )
# #         cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
# #         sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
# #         pairplot.map_offdiag(sns.kdeplot, cmap = 'Blues', n_levels = 60, shade = True) 
# #         pairplot.map_offdiag(sns.regplot, marker = '+', color = 'black',
# #                              scatter_kws = {'alpha' : 0.4}, fit_reg=False) 
#         pairplot.map_offdiag(sns.regplot, scatter_kws = {'alpha' : 0.4}, fit_reg=False) 
#         pairplot.set(xlim = (0,None), ylim = (0,None))
# #         pairplot.xlim(0, None)
#         pairplot.map_diag(sns.distplot, kde = False, rug = True)
#         pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)
#         pairplot.map_offdiag(sns.jointplot )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_bands.pdf'))
 
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
        pairplot = hes5.plot_posterior_distributions(my_posterior_samples)
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_not_oscillating_bands_different_prior.pdf'))
 
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
        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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

        print 'number of accepted samples is ' + str(len(my_posterior_samples))
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
        print 'number of accepted samples is'
        print len(my_posterior_samples)

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','all_parameter_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

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
        print 'number of accepted samples is'
        print len(my_posterior_samples)

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
 
    def test_plot_relative_parameter_variation_coherence_increase(self):

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
                                                [1000,10], 
                                                [1000,20]])
        
        power_spectra = []
        coherences = np.zeros(trace_and_repetition_numbers.shape[0])
        periods = np.zeros(trace_and_repetition_numbers.shape[0])
        index = 0
        for number_of_traces, repetition_number in trace_and_repetition_numbers:
            print number_of_traces
            print repetition_number
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

            power_spectra.append(this_power_spectrum)
            coherences[index] = this_coherence
            periods[index] = this_period
            index += 1 

        theoretical_power_spectrum = hes5.calculate_theoretical_power_spectrum_at_parameter_point( 
                                                     basal_transcription_rate = 11,
                                                     translation_rate = 29,
                                                     repression_threshold = 31400,
                                                     transcription_delay = 29,
                                                     mRNA_degradation_rate = np.log(2)/30,
                                                     hill_coefficient = 5,
                                                     protein_degradation_rate = np.log(2)/90)
        
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

        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output','multiple_degradation_sweep_results.npy'))
        
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
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 200
        repetition_factor = 5
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
#                                                     model_results[:,1]>0.05)))) #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
#                                                    model_results[:,3]>0.3))))) #coherence
#                                     np.logical_and(model_results[:,3]>0.3, #coherence
#                                                     prior_samples[:,3]>20))))) #time_delay
                                                    prior_samples[:,0]<2))))) #time_delay

        my_posterior_samples = prior_samples[accepted_indices]
        ##
        # first_samples
        ##
        first_parameter = my_posterior_samples[0]
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
        second_parameter = my_posterior_samples[1]
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
        third_parameter = my_posterior_samples[2]
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

        fourth_parameter = my_posterior_samples[3]

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
                                       'output','low_transcription_rate_langevin_validation.pdf'))
