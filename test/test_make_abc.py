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
        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 100,
                                                                saving_name = 'sampling_results_langevin_100reps' )
        
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
 
    def test_make_abc_all_parameters(self):
        ## generate posterior samples
        total_number_of_samples = 20000
        acceptance_ratio = 0.02

        prior_bounds = {'basal_transcription_rate' : (0,100),
                        'translation_rate' : (0,200),
                        'repression_threshold' : (0,150000),
                        'time_delay' : (5,40),
                        'mRNA_degradation_rate': (np.log(2)/500, np.log(2/5)),
                        'protein_degradation_rate': (np.log(2)/500, np.log(2/5))}

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 100,
                                                                saving_name = 'sampling_results_langevin_small_prior',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full' )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 4))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_langevin_different_prior' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_langevin_abc_differently(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
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
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
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
                                      'output','pairplot_langevin_bands_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
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
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
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
                                      'output','pairplot_langevin_oscillating_bands_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
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
                                      'output','pairplot_langevin_not_oscillating_bands_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_langevin_abc_in_band_not_oscillating_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
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
                                      'output','pairplot_langevin_not_oscillating_bands_long_delay' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def xest_plot_langevin_abc_in_band_long_delay(self):
        ## generate posterior samples
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_langevin_100reps')
        acceptance_ratio = 0.02
        total_number_of_samples = 20000
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
