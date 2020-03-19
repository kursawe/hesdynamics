import os.path
import os
os.environ["OMP_NUM_THREADS"] = "1"
import unittest
import sys
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import matplotlib.gridspec 
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import scipy.optimize
import pandas as pd
import seaborn as sns
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
try:
    import gpflow
except ImportError:
    print('Could not import gpflow. This may affect GP regression tests.')
from numba import jit, autojit

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..','src'))
import hes5

import socket
import multiprocessing as mp
domain_name = socket.getfqdn()
if domain_name == 'jochen-ThinkPad-S1-Yoga-12':
    number_of_available_cores = 2
else:
#     number_of_available_cores = 1
    number_of_available_cores = mp.cpu_count()

class TestMakeFinalFigures(unittest.TestCase):

    def xest_switching_vs_fluctuation_rate_paper_figure_draft(self):
        
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
#         fluctuation_rates = np.array([0.05,2.0])
#         fluctuation_rates = np.logspace(0,3,10)
        fluctuation_rates = np.linspace(2,100,20)
        number_of_traces = 1000
#         number_of_traces = 10
        percentages = np.zeros_like(fluctuation_rates)
        activation_times = np.zeros_like(fluctuation_rates)
        activation_time_deviations = np.zeros_like(fluctuation_rates) 

        for fluctuation_index, fluctuation_rate in enumerate(fluctuation_rates):
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            turned_on_targets = x[-1,:]>2
            percentages[fluctuation_index] = np.sum(turned_on_targets)/number_of_traces
            active_level_bools = x>2
            these_activation_times = np.zeros(number_of_traces)
            for column_index, column in enumerate(active_level_bools.transpose()):
                entries = np.nonzero(column)
                if len(entries[0]) > 0:
                    minimum_entry = np.min(entries)
                    time = times[minimum_entry]
                    these_activation_times[column_index] = time
                else:
                    these_activation_times[column_index] = times[-1]
            activation_times[fluctuation_index] = np.mean(these_activation_times)
            activation_time_deviations[fluctuation_index] = np.std(these_activation_times)

        plt.figure(figsize = (2.25,2.25))
        plt.plot(fluctuation_rates, percentages)
        plt.xlabel('Y aperiodic lengthscale')
        plt.ylabel('Switching probability')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'stochastic_fluctuation_rate_probability_draft_figure.pdf'))

    def test_stochastic_flucutation_rate_dependant_activation_figure_draft(self):
        
        number_of_traces = 4
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        color_list = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        fluctuation_rates = [2,15,100]
        for fluctuation_rate in fluctuation_rates:
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            this_figure = plt.figure(figsize = (4.5,2.25))
            outer_grid = matplotlib.gridspec.GridSpec(1, 2 )
            this_left_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                    subplot_spec=outer_grid[0], hspace=0.0)
            for subplot_index in range(number_of_traces):
                this_axis = plt.Subplot(this_figure, this_left_grid[subplot_index])
                this_figure.add_subplot(this_axis)
                plt.plot(times,y.transpose()[subplot_index], lw = 0.5, color = color_list[subplot_index] )
                plt.yticks([2,7], fontsize = 8)
                plt.ylim(0,10)
            plt.ylabel('Input Signal Y')
            plt.xlabel('Time')
            this_axis.yaxis.set_label_coords(-0.15,2.0)
            plt.subplot(122)
            this_axis = plt.Subplot(this_figure, outer_grid[0])
            for trace_index, x_trace in enumerate(x.transpose()):
                plt.plot(times, x_trace, color = color_list[trace_index] )
            plt.ylabel('Downstream Response X')
            plt.xlabel('Time')
            plt.ylim(0,4)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'stochastic_multiple_fluctuation_rate_dependent_activation_' + 
                                     '{:.2f}'.format(fluctuation_rate) + '.pdf'))

    def xest_plot_dual_parameter_change(self):
        #This will plot two-dimensional and one-dimensional posterior distributions for the rate change ratios between MBS and CTRL
        
#         model = 'standard_extra'
        model = 'extrinsic_noise_extra'
#         model = 'transcription_amplification'
        
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay_large_extra')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay_large')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay_large_extra')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay')

        if model == 'standard_extra':
            saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_zebrafish_delay_large_extra')
            dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'data','zebrafish_dual_sweeps_standard_extra_complete_matrix.npy'))
        if model == 'extrinsic_noise_extra':
            saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_zebrafish_extrinsic_noise_delay_large_extra')
            dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'data','zebrafish_dual_sweeps_extrinsic_noise_extra_complete_matrix.npy'))
        if model == 'transcription_amplification':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_transcription_amplification')
            dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_transcription_amplification_complete_matrix.npy'))
        else:
            ValueError('do not recognise model name ' + model)

        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.1,
                                                   model_results[:,2]<150))))))
#                                     np.logical_and(model_results[:,3]>0.1,
#                                                     model_results[:,2]<150))))))
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
        print('number of accepted samples')
        print(len(my_posterior_results))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_final.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted_final.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_large_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_large_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_extra_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_more.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted_more.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_all.npy'))
#         fluctuation_rate_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_fluctuation_rates.npy'))
#         fluctuation_rate_results = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                                         'zebrafish_dual_sweeps_fluctuation_rates_full.npy'))

        translation_changes = dual_sweep_results[0,0,:,1]
        degradation_changes = dual_sweep_results[0,:,0,0]
        X, Y = np.meshgrid(translation_changes, degradation_changes)

        # need to replace this with something
        likelihoods = np.zeros((degradation_changes.shape[0],
                                translation_changes.shape[0]))
        
#         fluctuation_rates_before = fluctuation_rate_results[:,9,9,2]
#         fluctuation_rates_before = dual_sweep_results[:,9,9,-1]
        fluctuation_rates_before = my_posterior_results[:,-1]
#         print(fluctuation_rates_before)
        list_of_indices = []
        corresponding_proportions = []
        total_condition_mask = np.zeros(len(dual_sweep_results))
        for translation_index, translation_change in enumerate(translation_changes):
            for degradation_index, degradation_change in enumerate(degradation_changes):
                these_results_after = dual_sweep_results[:, 
                                                         degradation_index, 
                                                         translation_index, 
                                                         :]
#                 these_fluctuation_rates_after = fluctuation_rate_results[:, 
#                                                          degradation_index, 
#                                                          translation_index, 
#                                                          :]

                relative_noise_after = ( these_results_after[:,-2]/np.power(these_results_after[:,3]*
                                        these_results_after[:,2],2))
                relative_noise_before = ( my_posterior_results[:,-2]/np.power(my_posterior_results[:,1]*
                                        my_posterior_results[:,0],2))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]>0.1,
#                                                 these_results_after[:,5]<my_posterior_results[:,3])))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 these_results_after[:,4] <150))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                                 relative_noise_after > relative_noise_before))
#                 condition_mask = relative_noise_after > relative_noise_before
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 relative_noise_after>relative_noise_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(relative_noise_after>1.2*relative_noise_before,
#                                                 these_results_after[:,4]<150))))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.5,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.5,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before)))
                condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
                                np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
                                np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
                                np.logical_and(my_posterior_results[:,3]>0.1,
                                np.logical_and(my_posterior_samples[:,1]*translation_change<40,
#                                 np.logical_and(these_results_after[:,4]<150,
#                                 np.logical_and(these_results_after[:,3]<0.25,
                                                these_results_after[:,-1]>1.1*fluctuation_rates_before)))))
#                                                 these_results_after[:,-1]>1.1*fluctuation_rates_before)))))
#                                                 these_results_after[:,-1]>fluctuation_rates_before))))
#                 condition_mask = np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]>fluctuation_rates_before)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(these_results_after[:,4]<150,
#                                                 these_results_after[:,-1]>fluctuation_rates_before))))
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before))))
#                 condition_mask = these_fluctuation_rates_after[:,2]>fluctuation_rates_before
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<5000,
#                                 np.logical_and(these_results_after[:,2]>2000,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
                
                total_condition_mask += condition_mask

                likelihoods[degradation_index, translation_index] = np.sum(condition_mask)

                these_indices = np.where(condition_mask)[0]
                if len(these_indices>0):
                    for item in these_indices:
                        list_of_indices.append(item)
                        corresponding_proportions.append((degradation_change, translation_change))

        print('total accepted samples (of prior and in total)')
        print(np.sum(total_condition_mask))
        print(len(list_of_indices))
                
        list_of_indices = np.array(list_of_indices)
        corresponding_proportions = np.array(corresponding_proportions)
        print(likelihoods)

        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_likelihoods_' + model + '.npy'),
                total_condition_mask)

        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_indices_' + model + '.npy'),
                list_of_indices)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_change_proportions_' + model + '.npy'),
                corresponding_proportions)

        print(translation_changes)
        translation_step = translation_changes[1] - translation_changes[0]
        left_translation_boundary = translation_changes[0] - 0.5*translation_step
        right_translation_boundary = translation_changes[-1] + 0.5*translation_step
        translation_bin_edges = np.linspace(left_translation_boundary,right_translation_boundary, len(translation_changes) +1)
        print(translation_bin_edges)

        print(degradation_changes)
        degradation_step = degradation_changes[1] - degradation_changes[0]
        left_degradation_boundary = degradation_changes[0] - 0.5*degradation_step
        right_degradation_boundary = degradation_changes[-1] + 0.5*degradation_step
        degradation_bin_edges = np.linspace(left_degradation_boundary,right_degradation_boundary, len(degradation_changes) +1)
        print(degradation_bin_edges)
        
#         print('likelihood of dec. coh. and incr. fluct. at 1.5 transl.')
#         print(likelihoods[9,14])

        this_figure = plt.figure(figsize = (2.5,1.9))
        colormesh = plt.pcolormesh(translation_bin_edges,degradation_bin_edges,likelihoods, rasterized = True)
#         colormesh = plt.pcolormesh(translation_bin_edges,degradation_bin_edges,likelihoods, rasterized = True,
#                                    vmin = 0, vmax = 10)
#         colormesh = plt.pcolormesh(likelihoods, rasterized = True)
#         colormesh = plt.pcolormesh(degradation_bin_edges,translation_bin_edges,likelihoods, rasterized = True,
#                                     vmin = 0, vmax = 100)
#         plt.pcolor(X,Y,expected_coherence)
#         plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.xlabel("Translation ratio MBS/CTRL", labelpad = 1.3, x = 0.45)
#         plt.xlabel("Translation ratio MBS/CTRL")
        plt.xlim(4,)
        plt.ylim(0.8,)
#         plt.xlim(0.95,2.05)
#         plt.ylim(0.25,1.05)
#         plt.ylim(0.05,1.05)
        plt.ylabel("mRNA degradation\nratio MBS/CTRL", y=0.4)
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.new_vertical(size=0.07, pad=0.5, pack_start=True)
        this_figure.add_axes(cax)

        tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        this_colorbar = this_figure.colorbar(colormesh, cax = cax, orientation = 'horizontal')
        this_colorbar.locator = tick_locator
        this_colorbar.update_ticks()
#         for ticklabel in this_colorbar.ax.get_xticklabels():
#             ticklabel.set_horizontalalignment('left') 
        this_colorbar.ax.set_ylabel('Likelihood\nof ratios', rotation = 0, verticalalignment = 'top', labelpad = 30)
        plt.tight_layout(pad = 0.05)
#         plt.tight_layout()

        file_name = os.path.join(os.path.dirname(__file__),
#                                  'output','zebrafish_likelihood_plot_extrinsic_noise')
#                                  'output','zebrafish_likelihood_plot_shifted_final')
                                 'output','zebrafish_likelihood_plot_' + model)
 
        plt.savefig(file_name + '.pdf', dpi = 600)
        plt.savefig(file_name + '.eps', dpi = 600)
        plt.savefig(file_name + '.png', dpi = 600)


        sns.set(font_scale = 1.1, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)

        my_figure = plt.figure(figsize= (3.5,3))

        translation_change_distribution = np.sum(likelihoods, axis = 0)
        my_figure.add_subplot(121)
        translation_bar_width = translation_changes[1] - translation_changes[0]
        translation_integral = np.trapz(translation_change_distribution, translation_changes)
        plt.bar(translation_changes, translation_change_distribution/translation_integral, align = 'center', 
                width = translation_bar_width, edgecolor = 'black')
        # plt.ylabel("Probability", labelpad = 20)
        plt.ylabel("Probability")
        plt.xlabel("Translation\nratio\nMBS/CTRL")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(122)
#         translation_rate_bins = np.logspace(0,2.3,20)
        degradation_bar_width = degradation_changes[1] - degradation_changes[0]
        degradation_change_distribution = np.sum(likelihoods, axis = 1)
        degradation_integral = np.trapz(degradation_change_distribution, degradation_changes)
        plt.bar(degradation_changes, degradation_change_distribution/degradation_integral, align = 'center',
                width = degradation_bar_width, edgecolor = 'black')
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,1)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        # plt.xticks([-1,0,1], [r'$10^{\mathrm{-}1}$',r'$10^0$',r'$10^1$'])
        plt.xlabel("mRNA degradation\nratio\nMBS/CTRL")
        # plt.gca().set_ylim(0,1)
        # plt.gca().set_xlim(-1,1)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
        # plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_ratio_changes_' + model + '.pdf'))
        
        my_figure = plt.figure(figsize= (3.5,3))
        my_figure.add_subplot(121)
        translation_range = translation_changes[-1] - translation_changes[0]
        plt.bar(translation_changes, 1/translation_range, align = 'center', 
                width = translation_bar_width, edgecolor = 'black')
        # plt.ylabel("Probability", labelpad = 20)
        plt.ylabel("Probability")
        plt.xlabel("Translation\nratio\nMBS/CTRL")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(122)
#         translation_rate_bins = np.logspace(0,2.3,20)
        degradation_range = degradation_changes[-1] - degradation_changes[0]
        plt.bar(degradation_changes, 1/degradation_range, align = 'center',
                width = degradation_bar_width, edgecolor = 'black')
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,1)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        # plt.xticks([-1,0,1], [r'$10^{\mathrm{-}1}$',r'$10^0$',r'$10^1$'])
        plt.xlabel("mRNA degradation\nratio\nMBS/CTRL")
        # plt.gca().set_ylim(0,1)
        # plt.gca().set_xlim(-1,1)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
        # plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_ratio_changes_' + model + '_prior.pdf'))
 
    def xest_plot_zebrafish_inference_extrinsic_noise(self):
        # the options relevant to the paper are 'prior', 'mean_std_period' (=posterior from CTRL only)
        # dual_coherence_and_lengthscale_decrease
        option = 'prior'
#         option = 'mean_period_and_coherence'
#         option = 'mean_longer_periods_and_coherence'
#         option = 'mean_and_std'
#         option = 'mean_std_period'
#         option = 'mean_std_period_coherence'
#         option = 'mean_std_period_coherence_noise'
#         option = 'coherence_decrease_translation'
#         option = 'coherence_decrease_degradation'
#         option = 'dual_coherence_decrease'
#         option = 'mean'
        # option = 'dual_coherence_and_lengthscale_decrease'
#         option = 'mean_std_period_fewer_samples'
#         option = 'mean_std_period_coherence'
#         option = 'weird_decrease'

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
#                                     'sampling_results_zebrafish_extrinsic_noise_delay_large')
                                    'sampling_results_zebrafish_extrinsic_noise_delay_large_extra')
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
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                                       model_results[:,0]<1500))  #standard deviation
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
        elif option == 'mean_longer_periods_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<150,
                                        np.logical_and(model_results[:,3]>0.25,
                                                       model_results[:,3]<0.4)))))
        elif option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period_fewer_samples':
            accepted_indices = np.where(np.logical_and(model_results[:4000,0]>1000, #protein number
                                        np.logical_and(model_results[:4000,0]<2500,
                                        np.logical_and(model_results[:4000,1]<0.15,
                                        np.logical_and(model_results[:4000,1]>0.05,
                                                       model_results[:4000,2]<150)))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'mean_std_period_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.1,
                                                       model_results[:,2]<150))))))
        elif option == 'mean_std_period_coherence_noise':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.4,
                                        np.logical_and(prior_samples[:,-1]>10, #noise
                                                       model_results[:,2]<150)))))))
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
        elif option == 'weird_decrease':
            change = 'decreased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradationtest')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
#             weird_indices = np.where(results_before_change[:,0]>results_after_change[:,0])
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'dual_coherence_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_likelihoods_extrinsic_noise_extra.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        elif option == 'dual_coherence_and_lengthscale_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.1,
                                                       model_results[:,2]<150))))))
            saving_path = os.path.join(os.path.dirname(__file__),'data','zebrafish_dual_sweeps_likelihoods_extrinsic_noise_extra.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        else:
            ValueError('could not identify posterior option')
#       
        if option not in ['weird_decrease', 'coherence_decrease_degradation',
                          'coherence_decrease_translation']:
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))
        print('minimal transcription is')
        print(np.min(my_posterior_samples[:,0]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,0])))
        print('minimal translation is')
        print(np.min(my_posterior_samples[:,1]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,1])))

        my_posterior_samples[:,2]/=1000

        print(my_posterior_samples.shape)
#         my_pairplot = hes5.plot_posterior_distributions(my_posterior_samples)

        data_frame = pd.DataFrame( data = my_posterior_samples[:,(0,1,2,3,4,5,7)],
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e3', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation',
                                             'Extrinsic noise rate'])

        ### PAIRGRID
#         my_adjusted_posterior_samples = np.copy(my_posterior_samples)
#         my_adjusted_posterior_samples[:,5] = np.log(2)/my_adjusted_posterior_samples[:,5]
#         my_adjusted_posterior_samples[:,0] = np.log10(my_adjusted_posterior_samples[:,0])
#         my_adjusted_posterior_samples[:,1] = np.log10(my_adjusted_posterior_samples[:,1])
#         new_data_frame = pd.DataFrame( data = my_adjusted_posterior_samples[:,:6],
#                                    columns= ['log10(Transcription rate)', 
#                                              'log10(Translation rate)', 
#                                              'Repression threshold/1e3', 
#                                              'Transcription delay',
#                                              'Hill coefficient',
#                                              'mRNA half life'])
#         my_pairplot = sns.PairGrid(new_data_frame)
# #         my_pairplot = sns.pairplot(new_data_frame)
#         my_pairplot.map_upper(plt.scatter, alpha = 0.02, color = 'black', rasterized = True)
# #         my_pairplot.map_upper(sns.kdeplot,rasterized = True)
#         my_pairplot.map_diag(plt.hist)
#         my_pairplot.map_lower(sns.kdeplot, cmap = 'Reds', rasterized = True)
# #         my_pairplot.axes[-1,0].set_xscale("log")
# #         my_pairplot.axes[-1,1].set_xscale("log")
#         my_pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                          'output',
#                                          'pairplot_zebrafish_abc_' +  option + '.pdf'))
#         ### END PAIRGRID

        sns.set(font_scale = 1.1, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(171)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(np.log10(1.0),np.log10(120.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-0.5,np.log10(120.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription\nrate [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1.2)
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
        plt.xticks([0,1,2], [r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        my_figure.add_subplot(172)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(np.log10(0.1),np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : None},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,1)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([-1,0,1], [r'$10^{\mathrm{-}1}$',r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation\nrate [1/min]")
        plt.gca().set_ylim(0,1)
        plt.gca().set_xlim(-1,1)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        my_figure.add_subplot(173)
        sns.distplot(data_frame['Repression threshold/1e3'],
                     kde = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression\nthreshold [1e3]")
        plt.gca().set_ylim(0,0.8)
        plt.gca().set_xlim(0,4)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(174))
        time_delay_bins = np.linspace(1,12,13)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     bins = time_delay_bins)
        plt.gca().set_xlim(1,10)
#         plt.gca().set_ylim(0,0.07)
#         plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription\ndelay [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(175))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.5)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        my_figure.add_subplot(176)
#         translation_rate_bins = np.logspace(0,2.3,20)
#         degradation_rate_bins = np.linspace(np.log(2.0)/15.0,np.log(2)/1.0,20)
#         histogram, bin_edges = np.histogram(data_frame['mRNA degradation'], degradation_rate_bins, 
#                                             density = True)
#         plt.hist(histogram[::-1], np.log(2)/bin_edges[::-1] )

        half_lifes = np.log(2)/data_frame['mRNA degradation']
        print(half_lifes)
        half_life_bins = np.linspace(1,11,20)
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
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
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

        ## EXTRINSIC NOISE
        my_figure.add_subplot(177)
#         transcription_rate_bins = np.logspace(-1,2,20)
#         transcription_rate_bins = np.linspace(np.log10(0.1),np.log10(1000),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Extrinsic noise rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = 20)
#                     bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
#         plt.gca().set_xlim(-0.5,np.log10(60.0))
        plt.xlabel("Transcription\nnoise rate [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,0.5)
#         plt.gca().set_ylim(0,1)
        plt.xticks([0,2], [r'$10^0$',r'$10^2$'])
#         plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_extrinsic_noise_' + option + '.pdf'))
        
        sns.reset_orig()
        # one for the transcription delay
        plt.figure(figsize = (2.5, 1.9))
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     bins = time_delay_bins)
        plt.gca().set_xlim(1,10)
#         plt.gca().set_ylim(0,0.07)
#         plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel("Transcription delay [min]")
        plt.ylabel("Probability")
        plt.tight_layout()
# 
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inferred_delay_distribution_' + option + '.pdf'))

        # one for the extrinsic noise
        plt.figure(figsize = (2.5, 1.9))
        sns.distplot(np.log10(data_frame['Extrinsic noise rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = 20)
#                     bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
#         plt.gca().set_xlim(-0.5,np.log10(60.0))
        plt.xlabel("Transcription noise rate [1/min]", x = 0.4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,0.5)
        plt.ylabel("Probability")
#         plt.gca().set_ylim(0,1)
        plt.xticks([0,2], [r'$10^0$',r'$10^2$'])
#         plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        plt.tight_layout()
# 
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inferred_extrinsic_noise_distribution_' + option + '.pdf'))

    def xest_plot_posterior_predictions_without_noise(self):
        my_no_noise_results_before = np.load(os.path.join(os.path.dirname(__file__),
                                                          'data','zebrafish_noise_comparison_no_noise_before.npy'))

        my_no_noise_results_after = np.load(os.path.join(os.path.dirname(__file__),
                                                         'data','zebrafish_noise_comparison_no_noise_after.npy'))
        
        my_selected_results_before = np.load(os.path.join(os.path.dirname(__file__),
                                                          'data', 'zebrafish_noise_comparison_actual_before.npy'))
        
        my_selected_results_after = np.load(os.path.join(os.path.dirname(__file__),
                                                         'data','zebrafish_noise_comparison_actual_after.npy'))
 
        my_selected_parameters_before = np.load(os.path.join(os.path.dirname(__file__),
                                                         'data','zebrafish_noise_comparison_real_parameters_before.npy'))
        
        my_selected_parameters_after = np.load(os.path.join(os.path.dirname(__file__),
                                                         'data','zebrafish_noise_comparison_real_parameters_after.npy'))

        dictionary_of_indices = { 'Coherence' : 3,
                                  'Period' : 2,
                                  'Mean expression' : 0,
                                  'COV' : 1,
                                  'Aperiodic lengthscale' : 11}
        
        for stats_name in dictionary_of_indices.keys():
            plt.figure(figsize = (5.0,1.9))
            axes1 = plt.subplot(121)
            this_data_frame = pd.DataFrame(np.column_stack((my_no_noise_results_before[:,dictionary_of_indices[stats_name]],
                                                            my_no_noise_results_after[:,dictionary_of_indices[stats_name]])),
                                           columns = ['CTRL','MBSm'])
            this_data_frame.boxplot()
            plt.title('Without transcriptional noise', fontsize = 10)
            if stats_name == 'Period':
                plt.ylim(0,150)
            plt.ylabel(stats_name)

            plt.subplot(122, sharey = axes1)
            this_data_frame = pd.DataFrame(np.column_stack((my_selected_results_before[:,dictionary_of_indices[stats_name]],
                                                            my_selected_results_after[:,dictionary_of_indices[stats_name]])),
                                           columns = ['CTRL','MBSm'])
            this_data_frame.boxplot()
            plt.ylabel(stats_name)
            plt.title('With transcriptional noise', fontsize = 10)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), 'output','no_noise_comparison_' + stats_name.replace(' ','_') + '.pdf'))
        
        real_fluctuation_rate_increases = ((my_selected_results_after[:,11]-my_selected_results_before[:,11])/
                                             my_selected_results_before[:,11])

        maximal_reference_index = np.argmax(real_fluctuation_rate_increases)
#         maximal_reference_index = np.argmax(relative_noise_increases)
        example_parameter_before = my_selected_parameters_before[maximal_reference_index]
        example_parameter_after = my_selected_parameters_after[maximal_reference_index]

        print('parameters before and after')
        print(example_parameter_before)
        print(example_parameter_after)
        
        print('summary statistics before and after with noise')
        print(my_selected_results_before[maximal_reference_index])
        print(my_selected_results_after[maximal_reference_index])
        print('summary statistics before and after without noise')
        print(my_no_noise_results_before[maximal_reference_index])
        print(my_no_noise_results_after[maximal_reference_index])
#         print(periods_before)
#         print(periods_after)
        example_trace_before = hes5.generate_langevin_trajectory( 8*60, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                2000,
                                                                example_parameter_before[7])
#                                                                 2000)

        example_trace_after = hes5.generate_langevin_trajectory( 8*60, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                example_parameter_after[2], #initial_protein,
                                                                2000,
                                                                example_parameter_after[7])
#                                                                 2000)

        example_no_noise_trace_before = hes5.generate_langevin_trajectory( 8*60, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                2000,
                                                                0.0)
#                                                                 2000)

        example_no_noise_trace_after = hes5.generate_langevin_trajectory( 8*60, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                example_parameter_after[2], #initial_protein,
                                                                2000,
                                                                0.0)
#                                                                 2000)


        traces_dict = { 'with noise' : [example_no_noise_trace_before, example_no_noise_trace_after],
                        'without noise' : [example_trace_before, example_trace_after]}
        for trace_name in traces_dict.keys():
            plt.figure(figsize = (2.5, 3.8))
            plt.subplot(211)
            plt.title('Control', fontsize = 10)
            plt.plot(traces_dict[trace_name][0][::6,0],
                     traces_dict[trace_name][0][::6,2])
            plt.ylabel('Her6 expression')
            plt.xlabel('Time [min]')
            plt.subplot(212)
            plt.title('MBSm', fontsize = 10)
            plt.plot(traces_dict[trace_name][1][::6,0],
                     traces_dict[trace_name][1][::6,2])
            plt.ylabel('Her6 expression')
            plt.xlabel('Time [min]')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'dual_change_example_' + trace_name.replace(' ','_') + '.pdf'))
 