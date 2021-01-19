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

    def test_time_dependent_langevin_trajectory(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        duration = 6000
        # define step wise function of mir-9
        mir9 = np.array([0]*duration)
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/20
        mRNA_half_life_upper_bound = np.log(2)/35
        mRNA_degradation = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9/100),5)))
        # define transcription rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        transcription = 1 / (1 + np.power((mir9/300),5))

        trace = hes5.generate_time_dependent_langevin_trajectory(duration = duration,
                                                                 all_repression_thresholds = np.array([390]*duration),
                                                                 all_hill_coefficients = np.array([5]*duration),
                                                                 all_mRNA_degradation_rates = mRNA_degradation,
                                                                 all_protein_degradation_rates = np.array([np.log(2)/22]*duration),
                                                                 all_basal_transcription_rates = transcription,
                                                                 all_translation_rates = np.array([1]*duration),
                                                                 all_transcription_delays = np.array([29]*duration),
                                                                 initial_mRNA = 0,
                                                                 initial_protein = 0,
                                                                 equilibration_time = 5000.0)

        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.plot(trace[:,0],trace[:,1],c='#20948B',label='mRNA')
        ax1.set_xlabel("Time (mins)")
        ax1.set_ylabel("mRNA (copy number)")
        ax1.tick_params(axis='y', labelcolor='#20948B')
        ax2 = ax1.twinx()
        ax2.plot(trace[:,0],trace[:,2],c='#F18D9E',label='Protein')
        ax2.set_ylabel("Protein (copy number)")
        ax2.tick_params(axis='y', labelcolor='#F18D9E')
        plt.title('Protein and mRNA Observations')
        plt.tight_layout()
        plt.savefig(saving_path + 'mir9_langevin_trace.png')

    def test_time_dependent_langevin_trajectory_with_step_and_slope(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        duration = 6000
        hill_coefficient = 5
        # define slope function of mir-9
        mir9_slope = np.linspace(0,600,6000)
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/20
        mRNA_half_life_upper_bound = np.log(2)/35
        mRNA_degradation_slope = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_slope/100),hill_coefficient)))
        # define transcription rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        transcription_slope = 1 / (1 + np.power((mir9_slope/300),hill_coefficient))

        trace_slope = hes5.generate_time_dependent_langevin_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([390]*duration),
                                                                       all_hill_coefficients = np.array([5]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_slope,
                                                                       all_protein_degradation_rates = np.array([np.log(2)/22]*duration),
                                                                       all_basal_transcription_rates = transcription_slope,
                                                                       all_translation_rates = np.array([1]*duration),
                                                                       all_transcription_delays = np.array([29]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)

        # define step function of mir-9
        mir9_step = np.array([0]*(duration//4) +
                             [200]*(duration//4) +
                             [400]*(duration//4) +
                             [600]*(duration//4))
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/20
        mRNA_half_life_upper_bound = np.log(2)/35
        mRNA_degradation_step = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_step/100),hill_coefficient)))
        # define transcription rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        transcription_step = 1 / (1 + np.power((mir9_step/300),hill_coefficient))

        trace_step = hes5.generate_time_dependent_langevin_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([390]*duration),
                                                                       all_hill_coefficients = np.array([5]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_step,
                                                                       all_protein_degradation_rates = np.array([np.log(2)/22]*duration),
                                                                       all_basal_transcription_rates = transcription_step,
                                                                       all_translation_rates = np.array([1]*duration),
                                                                       all_transcription_delays = np.array([29]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)


        fig, ax1 = plt.subplots(2,2,figsize=(18,9),constrained_layout=True)
        fig.suptitle('Hill coefficient = ' + str(hill_coefficient))

        # slope
        ax1[0,0].plot(trace_slope[:,0],mir9_slope,c='#86AC41',label='mRNA')
        ax1[0,0].hlines(300,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(0,320,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,0].hlines(100,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(4000,120,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,0].set_xlabel("Time (mins)")
        ax1[0,0].set_ylabel("mir9 (copy number)")
        ax1[0,0].tick_params(axis='y', labelcolor='#86AC41')
        ax1[1,0].plot(trace_slope[:,0],trace_slope[:,1],c='#20948B',label='mRNA')
        ax1[1,0].set_xlabel("Time (mins)")
        ax1[1,0].set_ylabel("mRNA (copy number)")
        ax1[1,0].set_ylim(0,35)
        ax2 = ax1[1,0].twinx()
        ax2.plot(trace_slope[:,0],trace_slope[:,2],c='#F18D9E',label='Protein')
        ax2.set_ylabel("Protein (copy number)")
        ax1[1,0].tick_params(axis='y', labelcolor='#20948B')
        ax2.tick_params(axis='y', labelcolor='#F18D9E')
        ax2.set_ylim(0,1000)
        # step
        ax1[0,1].plot(trace_step[:,0],mir9_step,c='#86AC41',label='mRNA')
        ax1[0,1].hlines(300,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(0,320,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,1].hlines(100,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(4000,120,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,1].set_xlabel("Time (mins)")
        ax1[0,1].set_ylabel("mir9 (copy number)")
        ax1[0,1].tick_params(axis='y', labelcolor='#86AC41')
        ax1[1,1].plot(trace_step[:,0],trace_step[:,1],c='#20948B',label='mRNA')
        ax1[1,1].set_xlabel("Time (mins)")
        ax1[1,1].set_ylabel("mRNA (copy number)")
        ax1[1,1].set_ylim(0,35)
        ax3 = ax1[1,1].twinx()
        ax3.plot(trace_step[:,0],trace_step[:,2],c='#F18D9E',label='Protein')
        ax3.set_ylabel("Protein (copy number)")
        ax1[1,1].tick_params(axis='y', labelcolor='#20948B')
        ax3.tick_params(axis='y', labelcolor='#F18D9E')
        ax3.set_ylim(0,1000)

        # plt.tight_layout()
        plt.savefig(saving_path + 'mir9_langevin_trace_step_and_slope_' + str(hill_coefficient) + '.png')

    def test_time_dependent_deterministic_trajectory_with_step_and_slope(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        duration = 6000
        hill_coefficient = 2
        # define slope function of mir-9
        mir9_slope = np.linspace(0,600,6000)
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/20
        mRNA_half_life_upper_bound = np.log(2)/35
        mRNA_degradation_slope = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_slope/100),hill_coefficient)))
        # define transcription rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        transcription_slope = 1 / (1 + np.power((mir9_slope/300),hill_coefficient))

        trace_slope = hes5.generate_time_dependent_deterministic_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([390]*duration),
                                                                       all_hill_coefficients = np.array([5]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_slope,
                                                                       all_protein_degradation_rates = np.array([np.log(2)/22]*duration),
                                                                       all_basal_transcription_rates = transcription_slope,
                                                                       all_translation_rates = np.array([1]*duration),
                                                                       all_transcription_delays = np.array([29]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)

        # define step function of mir-9
        mir9_step = np.array([0]*(duration//4) +
                             [200]*(duration//4) +
                             [400]*(duration//4) +
                             [600]*(duration//4))
        # define mRNA degradation rate based on action of mir-9 (function S(r) in Goodfellow et al (2014))
        mRNA_half_life_lower_bound = np.log(2)/20
        mRNA_half_life_upper_bound = np.log(2)/35
        mRNA_degradation_step = mRNA_half_life_lower_bound + ((mRNA_half_life_upper_bound - mRNA_half_life_lower_bound) /
                                                         (1 + np.power((mir9_step/100),hill_coefficient)))
        # define transcription rate based on action of mir-9 (function F(r) in Goodfellow et al (2014))
        transcription_step = 1 / (1 + np.power((mir9_step/300),hill_coefficient))

        trace_step = hes5.generate_time_dependent_deterministic_trajectory(duration = duration,
                                                                       all_repression_thresholds = np.array([390]*duration),
                                                                       all_hill_coefficients = np.array([5]*duration),
                                                                       all_mRNA_degradation_rates = mRNA_degradation_step,
                                                                       all_protein_degradation_rates = np.array([np.log(2)/22]*duration),
                                                                       all_basal_transcription_rates = transcription_step,
                                                                       all_translation_rates = np.array([1]*duration),
                                                                       all_transcription_delays = np.array([29]*duration),
                                                                       initial_mRNA = 0,
                                                                       initial_protein = 0,
                                                                       equilibration_time = 5000.0)


        fig, ax1 = plt.subplots(2,2,figsize=(18,9),constrained_layout=True)
        fig.suptitle('Hill coefficient = ' + str(hill_coefficient))

        # slope
        ax1[0,0].plot(trace_slope[:,0],mir9_slope,c='#86AC41',label='mRNA')
        ax1[0,0].hlines(300,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(0,320,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,0].hlines(100,0,6000,linestyle='--',color='grey')
        ax1[0,0].text(4000,120,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,0].set_xlabel("Time (mins)")
        ax1[0,0].set_ylabel("mir9 (copy number)")
        ax1[0,0].tick_params(axis='y', labelcolor='#86AC41')
        ax1[1,0].plot(trace_slope[:,0],trace_slope[:,1],c='#20948B',label='mRNA')
        ax1[1,0].set_xlabel("Time (mins)")
        ax1[1,0].set_ylabel("mRNA (copy number)")
        ax1[1,0].set_ylim(0,35)
        ax2 = ax1[1,0].twinx()
        ax2.plot(trace_slope[:,0],trace_slope[:,2],c='#F18D9E',label='Protein')
        ax2.set_ylabel("Protein (copy number)")
        ax1[1,0].tick_params(axis='y', labelcolor='#20948B')
        ax2.tick_params(axis='y', labelcolor='#F18D9E')
        ax2.set_ylim(0,1000)
        # step
        ax1[0,1].plot(trace_step[:,0],mir9_step,c='#86AC41',label='mRNA')
        ax1[0,1].hlines(300,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(0,320,'Protein translation\nrate reduced by half',fontsize=12)
        ax1[0,1].hlines(100,0,6000,linestyle='--',color='grey')
        ax1[0,1].text(4000,120,'mRNA degradation \nrate reduced by half',fontsize=12)
        ax1[0,1].set_xlabel("Time (mins)")
        ax1[0,1].set_ylabel("mir9 (copy number)")
        ax1[0,1].tick_params(axis='y', labelcolor='#86AC41')
        ax1[1,1].plot(trace_step[:,0],trace_step[:,1],c='#20948B',label='mRNA')
        ax1[1,1].set_xlabel("Time (mins)")
        ax1[1,1].set_ylabel("mRNA (copy number)")
        ax1[1,1].set_ylim(0,35)
        ax3 = ax1[1,1].twinx()
        ax3.plot(trace_step[:,0],trace_step[:,2],c='#F18D9E',label='Protein')
        ax3.set_ylabel("Protein (copy number)")
        ax1[1,1].tick_params(axis='y', labelcolor='#20948B')
        ax3.tick_params(axis='y', labelcolor='#F18D9E')
        ax3.set_ylim(0,1000)

        # plt.tight_layout()
        plt.savefig(saving_path + 'mir9_deterministic_trace_step_and_slope_' + str(hill_coefficient) + '.png')
