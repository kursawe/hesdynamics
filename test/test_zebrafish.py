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
import scipy.optimize
import pandas as pd
import seaborn as sns

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestZebrafish(unittest.TestCase):

    def xest_generate_single_oscillatory_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_deterministic_goodfellow_trajectory( duration = 720,
                                                                           protein_repression_threshold = 100,
                                                                           miRNA_repression_threshold = 10,
                                                                           upper_mRNA_degradation_rate = 0.03,
                                                                           lower_mRNA_degradation_rate = 0.03,
                                                                           protein_degradation_rate = 0.03,
                                                                           hill_coefficient_protein_on_protein = 5,
                                                                           hill_coefficient_miRNA_on_protein = 5,
                                                                           hill_coefficient_protein_on_miRNA = 5,
                                                                           miRNA_degradation_rate = 0.00001,
                                                                           transcription_delay = 19,
                                                                           initial_mRNA = 3,
                                                                           initial_protein = 100,
                                                                           initial_miRNA = 1)
#                                                          integrator = 'PyDDE',
#                                                          for_negative_times = 'no_negative' )

        #Second, plot the model

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein', color = 'black', ls = '--')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,3]*0.03, label = 'miRNA', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillating_trajectory.pdf'))
        

    def protein_difference_upon_degradation_increase(self, parameter_array):

        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array
        
        _, mean_protein_before_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                          hill_coefficient,
                                                                          RNA_degradation,
                                                                          protein_degradation,
                                                                          1.0,
                                                                          1.0)

        _, mean_protein_after_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                         hill_coefficient,
                                                                         RNA_degradation*2.0,
                                                                         protein_degradation,
                                                                         1.0,
                                                                         1.0)
        
        difference = -(mean_protein_before_increase - mean_protein_after_increase)/mean_protein_before_increase
        
        print 'these parameters are'
        print parameter_array
        print 'this difference is'
        print difference
        
        return difference
    
    def degradation_constraint_function(self,parameter_array):
        
        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array
        _, steady_protein = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                               hill_coefficient,
                                                               RNA_degradation,
                                                               protein_degradation,
                                                               1.0,
                                                               1.0)

        hill_derivative = ( hill_coefficient/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )
        
#         print 'these parameters are'
#         print parameter_array
#         print 'the degradation constraint is'
#         print hill_derivative - RNA_degradation*protein_degradation
        return hill_derivative - RNA_degradation*protein_degradation  
    
    def delay_constraint_function(self,parameter_array):
        
        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array

        _, steady_protein = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                               hill_coefficient,
                                                               RNA_degradation,
                                                               protein_degradation,
                                                               1.0,
                                                               1.0)

        hill_derivative = ( hill_coefficient/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )

        squared_degradation_difference = protein_degradation*protein_degradation - RNA_degradation*RNA_degradation
        squared_degradation_sum = protein_degradation*protein_degradation + RNA_degradation*RNA_degradation

        try:
            omega = np.sqrt(0.5*(np.sqrt(squared_degradation_difference*squared_degradation_difference
                               + 4*hill_derivative*hill_derivative) - 
                               squared_degradation_sum))
        except RuntimeWarning:
            return -10.0
        arccos_value = np.arccos( ( omega*omega - protein_degradation*RNA_degradation)/
                                    hill_derivative )
        
#         print 'these parameters are'
#         print parameter_array
#         print 'the degradation constraint is'
#         print hill_derivative - RNA_degradation*protein_degradation
        return omega - arccos_value
    
    def xest_maximise_protein_reduction_by_degradation_increase(self):

        degradation_constraint = { 'type' : 'ineq',
                                   'fun' : self.degradation_constraint_function }

        delay_constraint = { 'type' : 'ineq',
                             'fun' : self.delay_constraint_function }
        
        optimize_result = scipy.optimize.minimize(fun = self.protein_difference_upon_degradation_increase, 
                                                  x0 = np.array([1.0,6.0,0.01,0.01]),
                                                  constraints = [degradation_constraint, delay_constraint],
                                                  bounds = [[0.0001,np.inf],
                                                            [2.0,6.0],
                                                            [0.0,3.0],
                                                            [0.0,3.0]])
        
        print('the maximal difference we can get is')
        print(optimize_result.x)
        
    def xest_make_abc_samples(self):
        ## generate posterior samples
        total_number_of_samples = 200000
#         total_number_of_samples = 100
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.01,60),
                        'translation_rate' : (0.01,40),
                        'repression_threshold' : (0,16000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/15.0, np.log(2)/15.0 ),
                        'mRNA_half_life' : ( 1, 15) }

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_zebrafish',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 7))
        
    def test_plot_zebrafish_inference(self):
        option = 'prior'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                       model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                                       model_results[:,0]<8000))  #standard deviation
#                                                        model_results[:,1]>0.05)))  #standard deviation
        elif option == 'prior':
            accepted_indices = range(len(prior_samples))
        elif option == 'coherence':
            accepted_indices = np.where( model_results[:,3]>0.3 )  #standard deviation
        elif option == 'period':
            accepted_indices = np.where( model_results[:,2]<100 )  #standard deviation
        elif option == 'period_and_coherence':
            accepted_indices = np.where( np.logical_and( model_results[:,2]<100,
                                                         model_results[:,3]>0.3 ))  
        elif option == 'mean_period_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<100,
                                                       model_results[:,3]>0.3))))  
        elif option == 'amplitude_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000, #protein_number
#                                         np.logical_and(model_results[:,4]>40,
#                                         np.logical_and(model_results[:,4]>60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,3]>0.15)))) #standard deviation
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
        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        my_posterior_samples[:,2]/=1000

        data_frame = pd.DataFrame( data = my_posterior_samples[:,:6],
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation'])

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
        plt.gca().set_ylim(0,6)
#         plt.gca().set_ylim(0,1)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(162)
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
 
        my_figure.add_subplot(163)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        plt.gca().set_ylim(0,0.3)
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

        my_figure.add_subplot(166)
#         translation_rate_bins = np.logspace(0,2.3,20)
#         degradation_rate_bins = np.linspace(np.log(2.0)/15.0,np.log(2)/1.0,20)
#         histogram, bin_edges = np.histogram(data_frame['mRNA degradation'], degradation_rate_bins, 
#                                             density = True)
#         plt.hist(histogram[::-1], np.log(2)/bin_edges[::-1] )

        half_lifes = np.log(2)/data_frame['mRNA degradation']
        print half_lifes
        half_life_bins = np.linspace(1,15,20)
#         half_life_histogram, _ = np.histogram(half_lifes, half_life_bins, density = True)
#         print(half_life_histogram)
#         prior_histogram, _ = np.histogram( np.log(2)/prior_samples[:,5], half_life_bins, density = True )
#         corrected_histogram = half_life_histogram/prior_histogram
#         corrected_histogram = half_life_histogram
#         print(corrected_histogram)
#         bin_centres = (half_life_bins[:-1] + half_life_bins[1:])/2
#         width = 0.7*(half_life_bins[1] - half_life_bins[0])
         
#         plt.bar(bin_centres, corrected_histogram, align = 'center' , width = width )
        sns.distplot(half_lifes,
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                    bins = half_life_bins)
#
#         sns.distplot(data_frame['mRNA degradation'],
#                      kde = False,
#                      rug = False,
#                      norm_hist = True,
#                      hist_kws = {'edgecolor' : 'black'},
#                      bins = degradation_rate_bins)
# #         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("mRNA half-life \n [min]")
#         plt.gca().set_ylim(0,4.0)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_' + option + '.pdf'))

    def xest_plot_zebrafish_period_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where( model_results[:,3]>0.2)  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
#         my_posterior_samples = prior_samples
#         my_model_results = model_results

        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]
        print('mean is')
#         print(np.mean(all_periods[all_periods<10]))
        print(np.mean(all_periods))
        print('median is')
#         print(np.median(all_periods[all_periods<10]))
        print(np.median(all_periods))
        print('minimum is')
        print(np.min(all_periods))
        period_histogram, bins = np.histogram(all_periods[all_periods<300], bins = 400) 
#         period_histogram, bins = np.histogram(all_periods, bins = 400) 
        maximum_index = np.argmax(period_histogram)
        print('max bin is')
# # # #         print bins[maximum_index]
# # #         print bins[maximum_index+1]
# #         print bins[maximum_index+2]
#         print bins[maximum_index-1]
# #         sns.distplot(all_periods[np.logical_and(all_periods<1000,
#                                                 all_periods>100)],
        sns.distplot(all_periods[all_periods<300],
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
        plt.xlim(0,300)
#         plt.ylim(0,0.2)
#         plt.ylim(0,0.0003)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_period_distribution.pdf'))
 
    def xest_plot_zebrafish_coherence_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        my_posterior_samples = prior_samples
        my_model_results = model_results

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
                                 'output','zebrafish_coherence_distribution.pdf'))
        
    def xest_increase_mRNA_degradation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_increased_degradation = np.copy(my_posterior_samples)
        my_posterior_samples_increased_degradation[:,5]*=1.5
        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_increased_degradation, 
                                                                        number_of_traces_per_sample=200 )

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_increased_degradation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
        
    def xest_plot_mrna_increase_results(self):
        
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
    
        this_figure, axes = plt.subplots(2,3,figsize = (6.5,4.5))

        ## DEGRADATION
        this_data_frame = pd.DataFrame(np.column_stack((parameters_before_change[:,5],
                                                       parameters_after_change[:,5])),
                                        columns = ['before','after'])
        this_axes = axes[0,0]
        this_data_frame.boxplot(ax = axes[0,0])
        this_axes.set_ylabel('mRNA degradation')

        ## EXPRESSION
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,0],
                                                        results_after_change[:,0])),
                                        columns = ['before','after'])
        this_axes = axes[0,1]
        this_data_frame.boxplot(ax = axes[0,1])
        this_axes.set_ylabel('Hes expression')

        ## STANDARD DEVIATION
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,1],
                                                        results_after_change[:,1])),
                                        columns = ['before','after'])
        this_axes = axes[0,2]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Hes std')

        ## PERIOD
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,2],
                                                        results_after_change[:,2])),
                                        columns = ['before','after'])
        this_axes = axes[1,0]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Period')

        ## COHERENCE
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,3],
                                                        results_after_change[:,3])),
                                        columns = ['before','after'])
        this_axes = axes[1,1]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Coherence')

        ## MRNA
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,4],
                                                        results_after_change[:,4])),
                                        columns = ['before','after'])
        this_axes = axes[1,2]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('mRNA number')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation.pdf'))

    def xest_plot_mRNA_change_examples(self):
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
    
        example_parameter_index = 20
        example_parameter_before = parameters_before_change[example_parameter_index]
        example_parameter_after = parameters_after_change[example_parameter_index]
        
        example_trace_before = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                  2000)

        example_trace_after = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_after[2], #initial_protein,
                                                                  2000)

        plt.figure(figsize = (6.5, 2.5))
        plt.subplot(121)
        plt.plot(example_trace_before[:,0],
                 example_trace_before[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.plot(example_trace_after[:,0],
                 example_trace_after[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation_examples.pdf'))
