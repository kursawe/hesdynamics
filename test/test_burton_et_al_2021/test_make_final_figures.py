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
cm_to_inches = 0.3937008
class TestInference(unittest.TestCase):

    def xest_plot_protein_observations(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        # data = np.load(saving_path + 'protein_observations_ps6_fig5_1_cells_5_minutes.npy')
        data = np.load(saving_path + 'true_data_ps3.npy')
        protein = np.maximum(0,data[:,2] + 100*np.random.randn(data.shape[0]))

        my_figure, ax1 = plt.subplots(figsize=(12.47*0.7,8.32*0.7))
        ax1.scatter(data[:,0],data[:,2],marker='o',s=3,color='#20948B',alpha=0.75,label='protein')
        ax1.scatter(data[0:-1:10,0],protein[0:-1:10],marker='o',s=10,color='#F18D9E')
        # ax1.scatter(protein[:,0],protein[:,1],marker='o',s=10,c='#F18D9E')
        ax2 = ax1.twinx()
        ax2.scatter(data[0:-1:10,0],data[0:-1:10,1],marker='o',s=3,color='#86AC41',alpha=0.75,label='mRNA',zorder=1)
        ax1.tick_params(axis='y', labelcolor='#F18D9E')
        ax2.tick_params(axis='y', labelcolor='#86AC41')
        ax1.set_xlabel('Time (mins)')
        ax1.set_ylabel('Number of protein molecules')
        ax2.set_ylabel('Number of mRNA molecules')
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        plt.tight_layout()
        my_figure.savefig(os.path.join(saving_path,'ps3_data.pdf'))

    def xest_plot_mala_and_abc_posterior(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/single_cell_mala','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')

        # first single cell
        experiment_string = '040417_cell_53_cluster_1'
        output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])
        output[:,0]/=10000

        # second single cell
        experiment_string = '280317p1_cell_43_cluster_1'
        output1 = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output1 = output1.reshape(output1.shape[0]*output1.shape[1],output1.shape[2])
        output1[:,0]/=10000

        # third single cell
        experiment_string = '280317p1_cell_23_cluster_2'
        output2 = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output2 = output2.reshape(output2.shape[0]*output2.shape[1],output2.shape[2])
        output2[:,0]/=10000

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                    'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_samples[:,2]/=10000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay',
                                             'Hill coefficient'])

        fig, ax = plt.subplots(4,5,figsize= (13*1.4,12*1.4))

        # transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[0,0].hist(np.log10(np.array(data_frame['Transcription rate'])),
                      ec = 'black',
                      density = True,
                      alpha=0.6)
        ax[0,0].set_xlim(-1,np.log10(120.0))
        ax[0,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[0,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[0,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        # translation_rate_bins = np.linspace(0,40,20)
        # import pdb; pdb.set_trace()
        ax[0,1].hist(data_frame['Translation rate'],
                     ec = 'black',
                     density = True,
                     bins = 20,
                     alpha=0.6)
        ax[0,1].set_xlim(9,40+1)
        ax[0,1].set_xticks([10,20,40])
        ax[0,1].set_xticklabels([10,20,40])
        ax[0,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[0,2].hist(data_frame['Repression threshold/1e4'],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     alpha=0.6)
        ax[0,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[0,2].set_ylim(0,0.22)
        ax[0,2].set_xlim(0,12)
        ax[0,2].locator_params(axis='x', tight = True, nbins=4)
        ax[0,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,10)
        ax[0,3].hist(data_frame['Transcription delay'],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     alpha=0.6)
        ax[0,3].set_xlim(5,40)
        # ax[0,3].set_ylim(0,0.04)
        ax[0,3].set_xlabel(" $\\tau$ (min)",fontsize=font_size*1.2)

        ax[0,4].hist(data_frame['Hill coefficient'],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     alpha=0.6)
        # ax[0,4].set_ylim(0,0.35)
        ax[0,4].set_xlim(2,6)
        ax[0,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ## MALA single cell 1
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[1,0].hist(output[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[1,0].set_xlim(-1,np.log10(120.0))
        ax[1,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[1,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[1,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[1,1].hist(output[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[1,1].set_xlim(np.log(10),np.log(40))
        ax[1,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[1,1].set_xticklabels([10,20,40])
        ax[1,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[1,2].hist(output[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[1,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[1,2].set_ylim(0,0.22)
        ax[1,2].set_xlim(0,12)
        ax[1,2].locator_params(axis='x', tight = True, nbins=4)
        ax[1,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[1,3].hist(output[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[1,3].set_xlim(5,40)
        # ax[1,3].set_ylim(0,0.04)
        ax[1,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[1,4].hist(output[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[1,4].set_ylim(0,0.35)
        ax[1,4].set_xlim(2,6)
        ax[1,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ## MALA single cell 2
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[2,0].hist(output1[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[2,0].set_xlim(-1,np.log10(120.0))
        ax[2,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[2,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[2,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[2,1].hist(output1[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[2,1].set_xlim(np.log(10),np.log(40))
        ax[2,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[2,1].set_xticklabels([10,20,40])
        ax[2,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[2,2].hist(output1[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[2,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[2,2].set_ylim(0,0.22)
        ax[2,2].set_xlim(0,12)
        ax[2,2].locator_params(axis='x', tight = True, nbins=4)
        ax[2,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[2,3].hist(output1[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[2,3].set_xlim(5,40)
        # ax[2,3].set_ylim(0,0.04)
        ax[2,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[2,4].hist(output1[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[2,4].set_ylim(0,0.35)
        ax[2,4].set_xlim(2,6)
        ax[2,4].set_xlabel("$h$",fontsize=font_size*1.2)

        ## MALA single cell 3
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[3,0].hist(output1[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[3,0].set_xlim(-1,np.log10(120.0))
        ax[3,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[3,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[3,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[3,1].hist(output2[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[3,1].set_xlim(np.log(10),np.log(40))
        ax[3,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[3,1].set_xticklabels([10,20,40])
        ax[3,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[3,2].hist(output2[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[3,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[3,2].set_ylim(0,0.22)
        ax[3,2].set_xlim(0,12)
        ax[3,2].locator_params(axis='x', tight = True, nbins=4)
        ax[3,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[3,3].hist(output2[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[3,3].set_xlim(5,40)
        # ax[3,3].set_ylim(0,0.04)
        ax[3,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[3,4].hist(output2[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[3,4].set_ylim(0,0.35)
        ax[3,4].set_xlim(2,6)
        ax[3,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ax[0,2].text(-4,.17,'ABC (population)',fontsize=font_size*1.2)
        ax[1,2].text(-4,.8,'Single cell (cluster 1)',fontsize=font_size*1.2)
        ax[2,2].text(-4,.6,'Single cell (cluster 1)',fontsize=font_size*1.2)
        ax[3,2].text(-4,.32,'Single cell (cluster 2)',fontsize=font_size*1.2)

        plt.tight_layout(w_pad = 0.0001)

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','mala_vs_abc_test' + '.pdf'))

    def xest_visualise_cluster_posterior_variance(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/single_cell_mala','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        clusters = ['cluster_1',
                    'cluster_2',
                    'cluster_3',
                    'cluster_4']
        for cluster in clusters:
            strings = [i for i in os.listdir(saving_path) if '.npy' in i
                                                          if cluster in i]
            alpha = 0.35
            fig, ax = plt.subplots(1,5,figsize=(1.4*13,1.4*3.02))
            parameters = [2,3,0,4,1]
            parameter_names = np.array(["$\log(\\alpha_m)$ (1/min)",
                                        "$\log(\\alpha_p)$ (1/min)",
                                        "$P_0$",
                                        "$\\tau$ (mins)",
                                        "$h$",])
            for string in strings:
                ps_string = string[string.find('ions_')+5:string.find('_cluster')]
                output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_'+ps_string +'_' + cluster + '_detrended.npy')
                output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

                for index, parameter in enumerate(parameters):
                    if parameter == 2:
                        hist, bins = np.histogram(np.exp(output[:,parameter]),density=True,bins=20)
                        logbins = np.geomspace(bins[0],bins[-1],20)
                        # import pdb; pdb.set_trace()
                        ax[index].hist(output[:,parameter],density=True,bins=logbins,alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                        ax[index].set_xlim(0,6)
                        ax[index].set_ylim(0,1.0)
                    if parameter == 3:
                        ax[index].hist(output[:,parameter],density=True,bins=np.geomspace(np.min(output[:,parameter]),np.max(output[:,parameter]),20),alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                    else:
                        ax[index].hist(output[:,parameter],density=True,bins=20,alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                ax[1].set_ylim(0,5)
                ax[0].set_ylabel("Probability",fontsize=font_size*1.2)


                def format_tick_labels(x, pos):
                    return '{:.0e}'.format(x)
                from matplotlib.ticker import FuncFormatter
                ax[2].yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

                # ax[2].set_yticklabels()
            plt.tight_layout()
            fig.suptitle("Cluster " + cluster[-1],y=1.001,fontsize=font_size*1.2)
            plt.savefig(saving_path + cluster + "_posteriors.png")

    def xest_plot_mala_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/output_jan_25','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        strings = [i for i in os.listdir(saving_path) if '.npy' in i
                                                      if 'ps11_fig5_1_cells_5_minutes' in i]
        for string in strings:
            ps_string = string[string.find('ps'):string.find('.npy')]
            true_parameters = np.load(loading_path + ps_string[:ps_string.find('_fi')] + '_parameter_values.npy')

            output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_'+ps_string +'.npy')
            output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

            mean_repression = round(np.mean(output[:,0]),-4) # round to nearest 10000

            hist_transcription, bins_transcription, _ = plt.hist(np.exp(output[:,2]),bins=20,density=True)
            logbins_transcription = np.geomspace(bins_transcription[0],
                                                 bins_transcription[-1],
                                                 20)
            plt.clf()

            hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,3]),bins=20,density=True)
            logbins_translation = np.geomspace(bins_translation[0],
                                               bins_translation[-1],
                                               20)
            plt.clf()

            my_figure = plt.figure(figsize=(18.87,5.66))
            # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
            plt.subplot(1,5,1)
            # sns.kdeplot(output[:,0])
            heights, bins, _ = plt.hist(output[:,0]/10000,bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[0]/10000,0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=0.8,xmax=8)
            # plt.xticks([0.5*mean_repression,1.5*mean_repression],labels=[int(0.5*mean_repression),int(1.5*mean_repression)])
            plt.xlabel('$P_0$ (10e4)',fontsize=font_size)
            plt.ylabel('Probability',fontsize=font_size)

            plt.subplot(1,5,2)
            # sns.kdeplot(output[:,1],bw=0.4)
            heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[1],0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=2,xmax=6)
            plt.xlabel('$h$',fontsize=font_size)

            plt.subplot(1,5,3)
            plt.xscale('log')
            # sns.kdeplot(output[:,2])
            heights, bins, _ = plt.hist(output[:,2],bins=logbins_transcription,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(np.log(true_parameters[4]),0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=0.2,xmax=7)
            plt.xticks([1,6],labels=[1,6])
            plt.xlabel('log($\\alpha_m$) (1/min)',fontsize=font_size)

            plt.subplot(1,5,4)
            plt.xscale('log')
            # sns.kdeplot(output[:,3])
            heights, bins, _ = plt.hist(output[:,3],bins=np.geomspace(1.6,3.75,20),density=True,ec='black',color='#20948B',alpha=0.3)
            # import pdb; pdb.set_trace()
            plt.vlines(np.log(true_parameters[5]),0,1.1*max(heights),color='k',lw=2)
            # plt.xlim(xmin=2,xmax=4)
            plt.xlabel('log($\\alpha_p$) (1/min)',fontsize=font_size)

            plt.subplot(1,5,5)
            # sns.kdeplot(output[:,4],bw=0.4)
            heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[6],0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=5,xmax=40)
            plt.xlabel('$\\tau$ (mins)',fontsize=font_size)

            plt.tight_layout()
            # plt.show()
            # saving_path = os.path.join(os.path.dirname(__file__), 'output','')
            my_figure.savefig(os.path.join(saving_path,'final_' + ps_string + '_posteriors_mala.png'))
            plt.clf()
            parameter_names = np.array(["$P_0$",
                                        "$h$",
                                        "$\\alpha_m$",
                                        "$\\alpha_p$",
                                        "$\\tau$"])

            output[:,[2,3]] = np.exp(output[:,[2,3]])
            df = pd.DataFrame(output[::10,:],columns=parameter_names)

            from scipy.stats import pearsonr

            def corrfunc(x, y, ax=None, **kws):
                """Plot the correlation coefficient in the top left hand corner of a plot."""
                # import pdb; pdb.set_trace()
                r, _ = pearsonr(x, y)
                ax = ax or plt.gca()
                ax.annotate(f'$\\nu$ = {r:.2f}', xy=(.1, .5), xycoords=ax.transAxes)
                # ax.set_axis_off()

            from matplotlib.colors import LinearSegmentedColormap
            colors = ['#000000','#20948B','#FFFFFF']  # Black -> color -> White
            n_bins = 200  # Discretizes the interpolation into bins
            cmap_name = 'my_list'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

            correlation_matrix = df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool),k=1)
            fig, ax = plt.subplots(figsize=(7.92*0.85,5.94*0.85))
            # sns.set(font_scale=font_size*0.1)
            sns.heatmap(correlation_matrix,mask=mask,annot=True,cmap=cm,cbar_kws={'label': 'Correlation coefficient, $\\nu$'},ax=ax)
            plt.savefig(saving_path + 'correlations_' + ps_string + '.png'); plt.close()

            # Create a pair grid instance
            # import pdb; pdb.set_trace()
            grid = sns.PairGrid(data= df[parameter_names[[0,3]]])
            # Map the plots to the locations
            grid = grid.map_upper(corrfunc)
            grid = grid.map_lower(sns.scatterplot, alpha=0.002,color='#20948B')
            grid = grid.map_lower(sns.kdeplot,color='k')
            grid = grid.map_diag(sns.histplot, bins = 20,color='#20948B');
            plt.savefig(saving_path + 'low_corr_pairplot_' + ps_string + '.png'); plt.close()

    def xest_plot_protein_observations(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')

        data = np.load(loading_path + 'protein_observations_ps11_fig5_1_cells_5_minutes.npy')
        plt.figure(figsize=(9.13,5.71))
        plt.scatter(data[:,0],data[:,1],marker='o',s=12,c='#F18D9E')
        plt.xlabel('Time (mins)',fontsize=font_size)
        plt.ylabel('Protein molecules',fontsize=font_size)
        plt.tight_layout()
        plt.savefig(saving_path + 'best_protein.png')

    def xest_accuracy_of_chains_by_coherence(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','output_jan_17/')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps')]

        coherence_values = np.zeros(len([i for i in chain_path_strings]))
        mean_error_values = np.zeros(len([i for i in chain_path_strings]))
        mode_error_values = np.zeros(len([i for i in chain_path_strings]))
        cov_error_values = np.zeros(len([i for i in chain_path_strings]))

        for chain_path_string in chain_path_strings:
            # import pdb; pdb.set_trace()
            mala = np.load(loading_path + chain_path_string)
            samples = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
            samples[:,[2,3]] = np.exp(samples[:,[2,3]])
            parameter_set_string = chain_path_string[chain_path_string.find('ps'):chain_path_string.find('_fig5')]
            true_values = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            sample_mean = np.mean(samples,axis=0)
            sample_std = np.std(samples,axis=0)
            # posterior mode
            sample_mode = np.zeros(5)
            for i in range(5):
                heights, bins, _ = plt.hist(samples[:,i],bins=35)
                sample_mode[i] = bins[heights.argmax()]

            coherence_values[np.where(coherence_values==0)[0][0]] = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[-2]

            mean_error_values[np.where(mean_error_values==0)[0][0]] = np.sum((np.abs(true_values-sample_mode))/true_values) # mean difference
            # cov_error_values[np.where(cov_error_values==0)[0][0]] = np.sum(sample_std/true_values) # std error

        plt.figure(figsize=(8.32,5.54))
        # mean error
        mean_mean = np.zeros(len(np.unique(coherence_values)))
        mean_std= np.zeros(len(np.unique(coherence_values)))
        for index, coherence in enumerate(np.unique(coherence_values)):
            mean_error_per_coherence = mean_error_values[coherence_values==coherence]
            plt.scatter(np.array([np.unique(coherence_values)[index]]*len(mean_error_per_coherence)),
                        mean_error_per_coherence,alpha=0.6,s=50, color='#b5aeb0')
            mean_mean[index] = np.mean(mean_error_values[coherence_values==coherence])
            mean_std[index] = np.std(mean_error_values[coherence_values==coherence])
        # # cov
        # cov_mean = np.zeros(len(np.unique(coherence_values)))
        # cov_std= np.zeros(len(np.unique(coherence_values)))
        # for index, coherence in enumerate(np.unique(coherence_values)):
        #     cov_error_per_coherence = cov_error_values[coherence_values==coherence]
        #     plt.scatter(np.array([np.unique(coherence_values)[index]]*len(cov_error_per_coherence)),
        #                 cov_error_per_coherence,alpha=0.6,s=50, color='#b5aeb0')
        #     cov_mean[index] = np.mean(cov_error_per_coherence)
        #     # import pdb; pdb.set_trace()
        #     cov_std[index] = np.std(cov_error_values[coherence_values==coherence])
        # # plt.fill_between([np.unique(coherence_values)[0],0.25],0, 25, alpha=0.1,color='#188C19',label="Biologically observed values",zorder=1)

        plt.fill_between(np.unique(coherence_values),np.maximum(0,mean_mean-mean_std), mean_mean+mean_std, alpha=0.2,color='#b5aeb0',zorder=2)
        plt.plot(np.unique(coherence_values),mean_mean,c='#b5aeb0',alpha=0.5,zorder=3)
        plt.scatter(np.unique(coherence_values)[0],0*mean_mean[0]-0.5,s=150,marker="v",color='#F18D9E',zorder=4,label="Low Coherence")
        plt.scatter(np.unique(coherence_values)[3],0*mean_mean[3]-0.5,s=150,marker="v",color='#8d9ef1',zorder=5,label="High Coherence")
        # plt.errorbar(np.unique(coherence_values),variance_mean/5,variance_std/5,linestyle=None,fmt='o',label='Relative SD',alpha=0.7)
        plt.xlabel('Coherence',fontsize=font_size)
        # plt.xlim(0,0.25)
        plt.ylabel('Relative mean error',fontsize=font_size)
        # plt.ylim(0,100)
        plt.tight_layout()
        plt.legend()
        plt.savefig(loading_path + 'coherence_mean_error_values.png')

    def xest_accuracy_of_chains_by_sampling_and_cells(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        ps6_true_parameter_values = np.load(os.path.join(saving_path,'ps6_parameter_values.npy'))[[0,1,4,5,6]]
        # ps6_true_parameter_values[[2,3]] = np.exp(ps6_true_parameter_values[[2,3]])
        ps6_measurement_variance = np.power(ps6_true_parameter_values[-1],2)
        ps9_true_parameter_values = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))[[0,1,4,5,6]]
        # ps9_true_parameter_values[[2,3]] = np.exp(ps9_true_parameter_values[[2,3]])
        ps9_measurement_variance = np.power(ps9_true_parameter_values[-1],2)

        ps6_chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps6')]
        ps9_chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps9')]
        ps6_datasets = {}
        ps9_datasets = {}

        for string in ps6_chain_path_strings:
            ps6_datasets[string] = np.load(loading_path + string)

        for string in ps9_chain_path_strings:
            ps9_datasets[string] = np.load(loading_path + string)
        ps6_mean_error_dict = {}
        ps6_cov_dict = {}
        ps9_mean_error_dict = {}
        ps9_cov_dict = {}

        for key in ps6_datasets.keys():
            ps6_mean_error_dict[key] = 0
            ps6_cov_dict[key] = 0
            short_chains = ps6_datasets[key].reshape(ps6_datasets[key].shape[0]*ps6_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            short_chain_cov = np.sum(short_chains_std/ps6_true_parameter_values)
            # relative mean
            relative_mean = np.sum(np.abs(ps6_true_parameter_values - short_chains_mean)/ps6_true_parameter_values)
            ps6_mean_error_dict[key] = relative_mean
            ps6_cov_dict[key] = short_chain_cov

        for key in ps9_datasets.keys():
            ps9_mean_error_dict[key] = 0
            ps9_cov_dict[key] = 0
            short_chains = ps9_datasets[key].reshape(ps9_datasets[key].shape[0]*ps9_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            short_chain_cov = np.sum(short_chains_std/ps9_true_parameter_values)
            # relative mean
            relative_mean = np.sum(np.abs(ps9_true_parameter_values - short_chains_mean)/ps9_true_parameter_values)
            ps9_mean_error_dict[key] = relative_mean
            ps9_cov_dict[key] = short_chain_cov

        plotting_strings = ['1_cells_5_minutes',
                            '1_cells_8_minutes',
                            '1_cells_12_minutes',
                            '1_cells_15_minutes']

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        mean_and_sd_covs = np.zeros((4,3))
        mean_and_sd_means = np.zeros((4,3))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps6_cov_dict.items() if 'ps6_' + string in key.lower()]
            xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[0].set_ylabel("Coefficient of Variation",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps6_mean_error_dict.items() if 'ps6_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[3,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[3,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[3,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[3,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],color='#F18D9E',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#F18D9E')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],color='#F18D9E',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#F18D9E')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps6_cov_and_mean_error_values_frequency.png')

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps9_cov_dict.items() if 'ps9_' + string in key.lower()]
            xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[0].set_ylabel("Coefficient of Variation",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps9_mean_error_dict.items() if 'ps9_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[3,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[3,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[3,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[3,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],c='#8d9ef1',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#8d9ef1')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],c='#8d9ef1',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#8d9ef1')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps9_cov_and_mean_error_values_frequency.png')

        plotting_strings = ['1_cells_15_minutes',
                            '2_cells_15_minutes',
                            '3_cells_15_minutes',
                            '4_cells_15_minutes',
                            '5_cells_15_minutes']

        mean_and_sd_covs = np.zeros((5,3))
        mean_and_sd_means = np.zeros((5,3))
        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        # ax[0].set_yscale('log')
        # ax[1].set_yscale('log')
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps6_cov_dict.items() if 'ps6_' + string in key.lower()]
            xcoords = [np.int(string[0])*12]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xticks(np.arange(6)*12)
            ax[0].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[0].set_ylabel("Coefficient of Variation",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps6_mean_error_dict.items() if 'ps6_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xticks(np.arange(6)*12)
            ax[1].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[0,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[0,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[0,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[0,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],color='#F18D9E',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#F18D9E')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],color='#F18D9E',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0],np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#F18D9E')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps6_cov_and_mean_error_values_length.png')

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps9_cov_dict.items() if 'ps9_' + string in key.lower()]
            xcoords = [np.int(string[0])*12]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xticks(np.arange(6)*12)
            # ax[0].set_xticklabels([10,20,40])
            ax[0].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[0].set_ylabel("Coefficient of Variation",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps9_mean_error_dict.items() if 'ps9_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xticks(np.arange(6)*12)
            ax[1].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[0,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[0,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[0,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[0,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],c='#8d9ef1',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#8d9ef1')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],c='#8d9ef1',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#8d9ef1')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps9_cov_and_mean_error_values_length.png')

    def xest_compare_mala_random_walk_histograms(self):
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','')
        mala = np.load(saving_path + 'mala_output_repression.npy')
        random_walk = np.load(saving_path + 'random_walk_repression.npy')
        true_values = np.load("data/ps3_parameter_values.npy")
        # import pdb; pdb.set_trace()
        # mh_mean_error = [np.std(random_walk[:i])/3407.99 for i in range(2000)]
        # mala_mean_error = [np.std(mala[:i])/3407.99 for i in range(2000)]
        # plt.figure(figsize=(7,5))
        # plt.plot(mh_mean_error,label='MH',color='#F18D9E')
        # plt.plot(mala_mean_error,label='MALA',color='#20948B')
        # plt.xlabel('Iterations')
        # plt.ylabel('Coefficient of Variation')
        # plt.title('Repression Threshold')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(saving_path + '2d_cov_repression.png')

        height = 0.012
        bw = 0.3
        bins=50
        lw = 3

        my_figure = plt.figure(figsize=(13.35*0.7,6.68*0.7))
        _,bins,_ = plt.hist(mala,density=True,bins=bins,alpha=0.8,color='#20948B',label='MALA')
        plt.vlines(np.mean(mala),0,height,color='#20948B',label='MALA mean',lw=lw)
        plt.hist(random_walk[:,0],density=True,bins=bins,alpha=0.6,color='#F18D9E',label='MH')
        plt.vlines(np.mean(random_walk[:,0]),0,height,color='#F18D9E',linestyle='dashed',label='MH Mean',lw=lw)
        plt.vlines(true_values[0],0,height,color='k',label='True Mean',lw=lw)
        plt.xlabel("$P_0$",fontsize=font_size)
        plt.legend()
        plt.xlim(xmin=2*bins[0]-bins[2],xmax=2*bins[-1]-bins[-3])
        plt.ylabel('Probability',fontsize=font_size)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','algo_comparison_repression.png'))
