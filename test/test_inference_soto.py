import unittest
import os.path
import sys
import argparse
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
# mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
# mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
font = {'size'   : 20}
plt.rc('font', **font)
import numpy as np
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from jitcdde import jitcdde,y,t
import time
from scipy.spatial.distance import euclidean
from scipy import stats
import pymc3 as pm
import arviz as az
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

number_of_cpus = mp.cpu_count()
font_size = 25
class TestInference(unittest.TestCase):
    def xest_time_dependent_langevin_trajectory_with_step_and_slope(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        duration = 6000
        hill_coefficient = 5
        protein_repression = 3900
        mRNA_deg_repression = 1000
        translation_repression = 3000
        max_mir9 = 4000
        # define slope function of mir-9
        mir9_slope = np.linspace(0,max_mir9,6000)
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/10
        mRNA_half_life_upper_bound = np.log(2)/20
        protein_degradation_rate = np.log(2)/15
        basal_transcription_rate = 1
        translation_rate = 10
        transcription_delay = 10
        mRNA_degradation_slope = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_slope/mRNA_deg_repression),hill_coefficient)))
        # define translation rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        translation_slope = translation_rate / (1 + np.power((mir9_slope/translation_repression),hill_coefficient))

        trace_slope = hes5.generate_time_dependent_langevin_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([protein_repression]*duration),
                                                                       all_hill_coefficients = np.array([hill_coefficient]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_slope,
                                                                       all_protein_degradation_rates = np.array([protein_degradation_rate]*duration),
                                                                       all_basal_transcription_rates = np.array([basal_transcription_rate]*duration),
                                                                       all_translation_rates = translation_slope,
                                                                       all_transcription_delays = np.array([transcription_delay]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)

        # define step function of mir-9
        # mir9_step = np.array([0]*(duration//4) +
        #                      [max_mir9/3]*(duration//4) +
        #                      [2*max_mir9/3]*(duration//4) +
        #                      [max_mir9]*(duration//4))
        mir9_step = np.array([0]*(duration//2) +
                             [2*max_mir9/3]*(duration//2))

        mRNA_degradation_step = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_step/mRNA_deg_repression),hill_coefficient)))
        # define translation rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        translation_step = translation_rate / (1 + np.power((mir9_step/translation_repression),hill_coefficient))

        trace_step = hes5.generate_time_dependent_langevin_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([protein_repression]*duration),
                                                                       all_hill_coefficients = np.array([hill_coefficient]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_step,
                                                                       all_protein_degradation_rates = np.array([protein_degradation_rate]*duration),
                                                                       all_basal_transcription_rates = np.array([basal_transcription_rate]*duration),
                                                                       all_translation_rates = translation_step,
                                                                       all_transcription_delays = np.array([transcription_delay]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)


        fig, ax1 = plt.subplots(2,2,figsize=(18,9),constrained_layout=True)
        fig.suptitle('Hill coefficient = ' + str(hill_coefficient))

        # slope
        ax1[0,0].plot(trace_slope[:,0],mir9_slope,c='#86AC41',label='mRNA')
        ax1[0,0].hlines(translation_repression,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(0,1.1*translation_repression,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,0].hlines(mRNA_deg_repression,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(4000,1.1*mRNA_deg_repression,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,0].set_xlabel("Time (mins)")
        ax1[0,0].set_ylabel("mir9 (copy number)")
        ax1[0,0].tick_params(axis='y', labelcolor='#86AC41')
        # ax1[1,0].plot(trace_slope[:,0],trace_slope[:,1],c='#20948B',label='mRNA')
        ax1[1,0].plot(trace_slope[:,0],trace_slope[:,2],c='#F18D9E',label='Protein')
        ax1[1,0].set_xlabel("Time (mins)")
        ax1[1,0].set_ylabel("protei n (copy number)")
        # # ax1[1,0].set_ylim(0,40)
        # ax2 = ax1[1,0].twinx()
        # ax2.set_ylabel("Protein (copy number)")
        # ax1[1,0].tick_params(axis='y', labelcolor='#20948B')
        # ax2.tick_params(axis='y', labelcolor='#F18D9E')
        # # ax2.set_ylim(0,1000)
        # step
        ax1[0,1].plot(trace_step[:,0],mir9_step,c='#86AC41',label='mRNA')
        ax1[0,1].hlines(translation_repression,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(0,1.1*translation_repression,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,1].hlines(mRNA_deg_repression,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(4000,1.1*mRNA_deg_repression,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,1].set_xlabel("Time (mins)")
        ax1[0,1].set_ylabel("mir9 (copy number)")
        ax1[0,1].tick_params(axis='y', labelcolor='#86AC41')
        # ax1[1,1].plot(trace_step[:,0],trace_step[:,1],c='#20948B',label='mRNA')
        ax1[1,1].plot(trace_step[:,0],trace_step[:,2],c='#F18D9E',label='Protein')
        ax1[1,1].set_xlabel("Time (mins)")
        ax1[1,1].set_ylabel("protein (copy number)")
        # ax1[1,1].set_ylim(0,40)
        # ax3 = ax1[1,1].twinx()
        # ax3.set_ylabel("Protein (copy number)")
        # ax1[1,1].tick_params(axis='y', labelcolor='#20948B')
        # ax3.tick_params(axis='y', labelcolor='#F18D9E')
        # ax3.set_ylim(0,1000)

        # plt.tight_layout()
        plt.savefig(saving_path + 'mir9_langevin_trace_step_and_slope_' + str(hill_coefficient) + '.png')

    def test_generate_time_dependent_deterministic_feed_forward_loop_trajectory(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        duration = 400
        delta_t = 0.1
        k1 = 2
        k2 = 2
        k3 = 1
        k4 = 1
        initial_R = 1
        initial_X = 0
        S_min = 0
        S_max = 4
        # slope = [i/(duration//2)*S_max for i in range(duration//2)]#np.linspace(0,1,duration//2)
        # linear_S = np.array([S_min]*(duration//4) +
        #                     slope +
        #                     [S_max]*(duration//4))
        linear_S = np.array([0*S_max]*(duration//4) +
                            [S_max/4]*(duration//4) +
                            [2*S_max/4]*(duration//4) +
                            [3*S_max/4]*(duration//4))
        all_k1_rates = np.array([k1]*duration)*linear_S
        all_k2_rates = np.array([k2]*duration)
        all_k3_rates = np.array([k3]*duration)*linear_S
        all_k4_rates = np.array([k4]*duration)

        linear_trace = hes5.generate_time_dependent_deterministic_feed_forward_loop_trajectory(duration = duration,
                                                                                        delta_t = delta_t,
                                                                                        all_k1_rates = all_k1_rates,
                                                                                        all_k2_rates = all_k2_rates,
                                                                                        all_k3_rates = all_k3_rates,
                                                                                        all_k4_rates = all_k4_rates,
                                                                                        initial_R = initial_R,
                                                                                        initial_X = initial_X)


        step_S = np.array([0*S_max]*(duration//4) +
                          [S_max/8]*(duration//4) +
                          [S_max/4]*(duration//4) +
                          [S_max]*(duration//4))
        all_k1_rates = np.array([k1]*duration)*step_S
        all_k2_rates = np.array([k2]*duration)
        all_k3_rates = np.array([k3]*duration)*step_S
        all_k4_rates = np.array([k4]*duration)
        step_trace = hes5.generate_time_dependent_deterministic_feed_forward_loop_trajectory(duration = duration,
                                                                                        delta_t = delta_t,
                                                                                        all_k1_rates = all_k1_rates,
                                                                                        all_k2_rates = all_k2_rates,
                                                                                        all_k3_rates = all_k3_rates,
                                                                                        all_k4_rates = all_k4_rates,
                                                                                        initial_R = initial_R,
                                                                                        initial_X = initial_X)

        # import pdb; pdb.set_trace()
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        ax[0,0].plot(np.arange(duration),linear_S)
        ax[0,1].plot(np.arange(duration),step_S)

        ax[1,0].plot(linear_trace[:,0]*delta_t,linear_trace[:,1],label='R')
        ax[1,0].plot(linear_trace[:,0]*delta_t,linear_trace[:,2],label='X')
        ax[1,1].plot(step_trace[:,0]*delta_t,step_trace[:,1],label='R')
        ax[1,1].plot(step_trace[:,0]*delta_t,step_trace[:,2],label='X')
        plt.legend()
        plt.tight_layout()
        plt.savefig(saving_path + 'ffl_step.png')
