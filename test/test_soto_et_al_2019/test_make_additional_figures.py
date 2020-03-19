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
from numba import jit

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

class TestMakeAdditionalFigures(unittest.TestCase):

    def test_include_delay_in_switching_examples(self):

        number_of_traces = 4
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        color_list = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        fluctuation_rates = [2,7,100]
        for fluctuation_rate in fluctuation_rates:
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces,
                                                                                include_upstream_feedback = True,
                                                                                feedback_delay = 5.0)
            this_figure = plt.figure(figsize = (4.5,4.5))
            outer_grid = matplotlib.gridspec.GridSpec(nrows= 2, ncols = 1 )
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
            #this_axis.yaxis.set_label_coords(-0.15,2.0)
            plt.subplot(212)
            this_axis = plt.Subplot(this_figure, outer_grid[1])
            for trace_index, x_trace in enumerate(x.transpose()):
                plt.plot(times, x_trace, color = color_list[trace_index] )
            plt.ylabel('Downstream Response X')
            plt.xlabel('Time')
            plt.ylim(0,4)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'delay_stochastic_multiple_fluctuation_rate_dependent_activation_updated_' + 
                                     '{:.2f}'.format(fluctuation_rate) + '.pdf'))

    def test_switching_vs_hes_levels_paper_figure_draft(self):
        
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
#         fluctuation_rates = np.array([0.05,2.0])
#         fluctuation_rates = np.logspace(0,3,10)
#         fluctuation_rates = np.linspace(2,60,20)
        upstream_levels = np.linspace(4,10,20)
        fluctuation_rate = 2
        number_of_traces = 1000
#         number_of_traces = 10
        percentages = np.zeros_like(upstream_levels)
        activation_times = np.zeros_like(upstream_levels)
        activation_time_deviations = np.zeros_like(upstream_levels) 

        for level_index, upstream_level in enumerate(upstream_levels):
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces,
                                                                                include_upstream_feedback = True,
                                                                                upstream_initial_level = upstream_level)
            turned_on_targets = x[-1,:]>2
            percentages[level_index] = np.sum(turned_on_targets)/number_of_traces
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
            activation_times[level_index] = np.mean(these_activation_times)
            activation_time_deviations[level_index] = np.std(these_activation_times)

        plt.figure(figsize = (2.25,2.25))
        plt.plot(upstream_levels, percentages)
        plt.xlabel('Y initial level')
        plt.ylabel('Switching probability')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'levels_probability_draft_figure.pdf'))

    def test_switching_dynamics_at_higher_levels_illustration_figure(self):
        number_of_traces = 4
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        color_list = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        fluctuation_rate = 2
        times, upstream, downstream = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces,
                                                                                            include_upstream_feedback = True,
                                                                                            upstream_initial_level = 8.5)
        this_figure = plt.figure(figsize = (4.5,4.5))
        plt.subplot(211)
        for trace_index, y_trace in enumerate(upstream.transpose()):
            plt.plot(times,y_trace, lw = 0.5, color = color_list[trace_index] )
#             plt.yticks([2,7], fontsize = 8)
        plt.ylim(0,10)
        plt.ylabel('Upstream Signal Y')
        plt.xlabel('Time')
        plt.subplot(212)
        for trace_index, x_trace in enumerate(downstream.transpose()):
            plt.plot(times, x_trace, color = color_list[trace_index] )
        plt.ylabel('Downstream Response X')
        plt.xlabel('Time')
        plt.ylim(0,4)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'new_stochastic_multiple_fluctuation_rate_dependent_activation_different_level.pdf'))


 