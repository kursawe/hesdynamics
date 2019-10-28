import unittest
import os
os.environ["OMP_NUM_THREADS"] = "1"
import os.path
import sys
import matplotlib as mpl
import matplotlib.gridspec 
mpl.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
# mpl.rcParams['ps.fonttype'] = 3
# mpl.rcParams['pdf.fonttype'] = 3
# mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)
import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestMakeFinalFigures(unittest.TestCase):
                                 
    def xest_make_period_distribution_plot(self):
        hilbert_periods = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                'shortened_posterior_hilbert_periods_per_cell_one_sample.npy'))
#                                   'shortened_smoothened_posterior_hilbert_periods_per_cell_one_sample.npy'))

        plt.figure(figsize = (2.5,1.9))
        plt.hist(hilbert_periods/60, density = True, bins =20, range = (0,10), edgecolor = 'black')
        plt.axvline(3.2, color = 'black')
#         plt.axvline(0.5, color = 'black')
        print('mean observed period is')
        print(np.mean(hilbert_periods/60))
        print('median observed period is')
        print(np.median(hilbert_periods/60))
        print('standard deviation of periods is')
        print(np.std(hilbert_periods/60))
#         plt.axvline(this_period/60)
        plt.xlabel('Period [hrs]')
#         plt.ylim(0,1)
        plt.ylabel('Likelihood')
        
        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'hilbert_period_distribution_for_paper')
        plt.savefig(file_name + '.pdf')
        plt.savefig(file_name + '.png', dpi = 600)

    def xest_make_standard_deviation_distribution_plot(self):
        
#         string_option = 'divided1'
#         string_option = 'divided2'
#         string_option = 'empty'
        string_option = 'relative'
#         string_option = 'standard'
#         string_option = 'square_bracket'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                   model_results[:,1]>0.05)))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('total number of accepted samples')
        print(len(my_posterior_samples))
        my_model_results = model_results[accepted_indices]

        my_figure = plt.figure(figsize= (2.5,1.9))

        all_standard_deviations = my_model_results[:,1]

        print('mean observed standard deviation is')
        print(np.mean(all_standard_deviations))
        print('median observed standard deviation is')
        print(np.median(all_standard_deviations))
        print('standard deviation of standard deviation is')
        print(np.std(all_standard_deviations))
#       
        plt.hist(all_standard_deviations,bins = 20, edgecolor = 'black')
        plt.ylabel("Likelihood")
        if string_option == 'standard':
            plt.xlabel("Standard deviation/mean HES5")
        elif string_option == 'square_bracket':
            plt.xlabel("Standard deviation [mean HES5]")
        elif string_option == 'divided1':
            plt.xlabel(r'$\frac{\mathrm{Standard}\ \mathrm{deviation}}{\mathrm{mean}\ \mathrm{HES5}}$')
        elif string_option == 'divided2':
            plt.xlabel(r'Standard deviation$\div$' + '\nmean HES5')
        elif string_option == 'empty':
            plt.xlabel("   ")
        elif string_option == 'relative':
            plt.xlabel('Relative standard deviation')
        plt.xlim(0.03,0.2)
#         plt.gca().locator_params(axis='y', tight = True, nbins=3)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)

        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'standard_deviation_predicted_distribution_for_paper_' + string_option)
        plt.savefig(file_name + '.pdf')
        plt.savefig(file_name + '.png', dpi = 600)
        
    def xest_make_model_visualisation(self):
        
                # sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        number_of_traces = 10
        figuresize = (7.3,3)
        my_figure = plt.figure(figsize = figuresize)
        outer_grid = matplotlib.gridspec.GridSpec(1, 3 )

        coherence_bands = [[0,0.05],
                           [0.45,0.47],
                           [0.85,0.9]]
        
        panel_labels = {0: 'i', 1: 'ii', 2: 'iii'}

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
#                     height_ratios= [number_of_traces, 1])
                    height_ratios= [number_of_traces, 1],
                    hspace = 0.6)
            this_inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                    subplot_spec=this_double_grid[0], hspace=0.0)
            this_parameter = my_posterior_samples[0]
            this_results = my_posterior_results[0]
            
            print(this_parameter)

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
                    plt.title('Coherence: ' + '{:.2f}'.format(this_results[3]), 
                              fontsize = 10)
#                     plt.title('Coherence: ' + '{:.2f}'.format(this_results[3]),
#                               fontsize = 5)
                    plt.gca().text(-0.12, 2.1, panel_labels[coherence_index], transform=plt.gca().transAxes)
                if subplot_index < number_of_traces - 1:
                    this_axis.xaxis.set_ticklabels([])
                if subplot_index !=9 or coherence_index != 0: 
                    this_axis.yaxis.set_ticklabels([])
                else:
                    plt.yticks([4,8], fontsize = 8)
                    plt.gca().tick_params('y', length = 5, direction = 'inout')
                if coherence_index == 0 and subplot_index == 4:
                    plt.ylabel('Expression/1e4 ', labelpad = 15)
                plt.xlim(0,1500)
            plt.xlabel('Time [min]', labelpad = 2)

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
            
        file_name = os.path.join(os.path.dirname(__file__),'output',
                                 'final_model_visualisation_for_paper')

        plt.savefig(file_name + '.pdf')
        plt.savefig(file_name + '.png', dpi = 600)
        
    def xest_plot_coherence_curves(self):

        my_figure = plt.figure( figsize = (2.5, 1.9) )

        my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                                        'extended_degradation_sweep.npy'))
#                                                           'repeated_degradation_sweep.npy'))
#         print(my_degradation_sweep_results[0,:,0])
#         print(np.log(2)/90)
#         my_filtered_indices = np.where(np.logical_and(my_degradation_sweep_results[:,9,4] - 
#                                                       my_degradation_sweep_results[:,3,4]>
#                                                       my_degradation_sweep_results[:,3,4]*1.0,
#                                                       my_degradation_sweep_results[:,3,4]>0.1))
#         print(len(my_filtered_indices[0]))
#         print(len(my_degradation_sweep_results))
#         my_degradation_sweep_results = my_degradation_sweep_results[my_filtered_indices]
        x_coord = -0.3
        y_coord = 1.05
        for results_table in my_degradation_sweep_results:
            plt.plot(results_table[:,0],
                    results_table[:,4], color = 'C0', alpha = 0.02, zorder = 0)
        plt.axvline( np.log(2)/90, color = 'black' )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
        plt.gca().set_rasterization_zorder(1)
        plt.xlabel(r'HES5 degradation [min$^{-1}$]')
        plt.ylabel('Coherence')
        plt.ylim(0,1)
        plt.xlim(0,np.log(2)/15.)
#         plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)

        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__),
                                 'output','coherence_curves_for_paper')
 
        plt.savefig(file_name + '.pdf', dpi = 600)
        plt.savefig(file_name + '.png', dpi = 600)
        
    def xest_plot_bifurcation_analysis(self):
#         option = 'stochastic'
#         option = 'stochastic'
        option = 'deterministic'

        X = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_protein_degradation_values_' + option + '.npy'))
        Y = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_mrna_degradation_values_' + option + '.npy'))
        expected_coherence = np.load(os.path.join(os.path.dirname(__file__),
                                       'data','oscillation_coherence_values_' + option + '.npy'))

        print('maximal coherence is')
        print(np.max(expected_coherence))
        this_figure = plt.figure(figsize = (2.5,1.9))
#         colormesh = plt.pcolormesh(X,Y,expected_coherence, rasterized = True, vmax = 0.22909221814462538)
        colormesh = plt.pcolormesh(X,Y,expected_coherence, rasterized = True)
#         plt.pcolor(X,Y,expected_coherence)
        plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.xlabel("Protein degradation [min$^{-1}$]", labelpad = 1.3)
        plt.ylabel("mRNA degradation\n[min$^{-1}$]", y=0.4)
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.new_vertical(size=0.07, pad=0.5, pack_start=True)
        this_figure.add_axes(cax)

        tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        this_colorbar = this_figure.colorbar(colormesh, cax = cax, orientation = 'horizontal')
        this_colorbar.locator = tick_locator
        this_colorbar.update_ticks()
        for ticklabel in this_colorbar.ax.get_xticklabels():
            ticklabel.set_horizontalalignment('left') 
        this_colorbar.ax.set_ylabel('Expected\ncoherence', rotation = 0, verticalalignment = 'top', labelpad = 30)
        plt.tight_layout(pad = 0.05)
#         plt.tight_layout()

        file_name = os.path.join(os.path.dirname(__file__),
                                 'output','oscillation_coherence_for_paper_' + option)
 
        plt.savefig(file_name + '.pdf', dpi = 600)
        plt.savefig(file_name + '.eps', dpi = 600)
        plt.savefig(file_name + '.png', dpi = 600)
        
    def test_make_likelihood_plot(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_narrowed')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_repeated')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_massive')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                                   model_results[:,1]>0.05)))

        label_fontsize = 10
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
#                                                           'output',
#                                                           'narrowed_relative_sweeps_' + 
                                                        'repeated_relative_sweeps_' + 
#                                                           'extended_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print('these accepted base samples are')
#             number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:,9,3] > 600,
#                                                                     my_parameter_sweep_results[:,9,4] < 0.1))[0])
            number_of_absolute_samples = len(np.where(np.logical_or(accepted_model_results[:,2] > 600,
                                                                    accepted_model_results[:,3] < 0.1))[0])
            print(number_of_absolute_samples)
            
            decrease_indices = np.where(np.logical_and(np.logical_or(accepted_model_results[:,3] < 0.1,
                                                                    accepted_model_results[:,2] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,0,3] < 300,
                                                        my_parameter_sweep_results[:,0,4] > 0.1)))
            # decrease_indices = np.where(np.logical_and(accepted_model_results[:,3] > 0.2,
                                                        # my_parameter_sweep_results[:,0,4] < accepted_model_results[:,3]*0.8))
#             decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                                     my_parameter_sweep_results[:,9,3] > 600),
#                                         np.logical_and(my_parameter_sweep_results[:,4,3] < 300,
#                                                         my_parameter_sweep_results[:,4,4] > 0.1)))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)
            print('these decrease samples are')
            number_of_decrease_samples = len(decrease_indices[0])
            print(number_of_decrease_samples)

            increase_indices = np.where(np.logical_and(np.logical_or(accepted_model_results[:,3] < 0.1,
                                                                    accepted_model_results[:,2] > 600),
                                        np.logical_and(my_parameter_sweep_results[:,1,3] < 300,
                                                        my_parameter_sweep_results[:,1,4] > 0.1)))
            # increase_indices = np.where(np.logical_and(accepted_model_results[:,3] > 0.2,
                                                        # my_parameter_sweep_results[:,1,4] < accepted_model_results[:,3]*0.8))
#             increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:,9,4] < 0.1,
#                                                                     my_parameter_sweep_results[:,9,3] > 600),
#                                         np.logical_and(my_parameter_sweep_results[:,14,3] < 300,
#                                                         my_parameter_sweep_results[:,14,4] > 0.1)))

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

        my_figure = plt.figure( figsize = (6.25, 1.9) )
        plt.bar(all_positions, sorted_bars[::-1], edgecolor = 'black')
        sorted_labels.reverse()
#         plt.xticks( all_positions + 0.4 , 
        plt.xticks( all_positions, 
                    sorted_labels,
                    rotation = 30,
                    fontsize = label_fontsize,
                    horizontalalignment = 'right')
        plt.xlim(all_positions[0] - 0.5, all_positions[-1] + 0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=4)
        plt.ylim(0,sorted_bars[-1]*1.2)
        plt.ylabel('Likelihood')

        my_figure.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'likelihood_plot_for_paper_with_fontsize_' + str(label_fontsize))
        plt.savefig(file_name + '.pdf')
        plt.savefig(file_name + '.eps')
        plt.savefig(file_name + '.png', dpi = 600)
 