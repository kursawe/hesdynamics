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
                                                                saving_name = 'sampling_results_narrowed',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill',
                                                                logarithmic = True)
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_narrow_abc_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
        
    def xest_plot_posterior_distributions(self):
        
        option = 'full'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
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
        elif option == 'not oscillating': 
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
        plt.gca().set_ylim(0,1.0)
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
        plt.gca().set_ylim(0,2.0)
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
        plt.gca().set_ylim(0,0.04)
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
                                    'output','inference_for_paper.pdf'))
 
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

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_narrowed')
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
                                    'output','prior_for_paper.pdf'))

    def test_plot_period_distribution_for_paper(self):
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
        
        print('mean period is')
        print(np.mode(all_periods[all_periods<600]))
#         import pdb; pdb.set_trace()
        my_figure = plt.figure(figsize= (5,3))
        sns.boxplot(data = [all_periods[all_periods<600], np.array(real_data)*60])
        plt.xticks([0,1], ['Model', 'Experiment']) 
        plt.ylabel('Period [min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','abc_period_distribution_for_paper.pdf'))
 
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
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','narrowed_degradation_sweep.npy'),
                    my_sweep_results)


    def xest_make_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
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
            np.save(os.path.join(os.path.dirname(__file__), 'output','narrowed_relative_sweeps_' + parameter_name + '.npy'),
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
 