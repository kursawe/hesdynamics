import unittest
import os
import os.path

import sys

os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rc

font = {'size': 10,
        'sans-serif': 'Arial'}
plt.rc('font', **font)

import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# make sure we find the right python module

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import hes5


class TestInfrastructure(unittest.TestCase):

    def xest_make_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
        # this is a test comment to see whether git push still works
        #         number_of_parameter_points = 2
        #         number_of_trajectories = 2

        #         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        #         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sampling_results_MiVe_expanded')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:, 0] > 5000,  # protein number
                                                   np.logical_and(model_results[:, 0] < 65000,  # protein_number
                                                                  #                                     np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                                  np.logical_and(model_results[:, 1] > 0.07,
                                                                                 model_results[:,
                                                                                 1] < 0.19))))  # standard deviation

        my_posterior_samples = prior_samples[accepted_indices]

        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative=True,
                                                                                     relative_range=(0.5, 1.5))

        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output',
                                 'repeated_relative_sweeps_MiVe_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])
            # print(parameter_name + ' ' + str(my_parameter_sweep_results[parameter_name]))

    def xest_plot_ss_curves(self):

        parameters = {'basal_transcription_rate',
                      'translation_rate',
                      'repression_threshold',
                      'time_delay',
                      'hill_coefficient'}
        # definde performances as an ordered list rather than an unordered set to be able to index later
        performances = ['Protein',
                        'Std',
                        'Period',
                        'Coherence',
                        'mRNA']
        layout = {'pad': 1.5,
                  'w_pad': 1.5,
                  'h_pad': 1.5,
                  'rect': (0, 0, 1, 1)}
        for parameter in parameters:
            # my_figure = plt.figure(figsize=(2.5, 1.9))

            this_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                                                'repeated_relative_sweeps_MiVe_' + parameter + '.npy'))
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
            print(parameter)
            print(this_parameter_sweep_results.shape)
            # lis = list(enumerate(this_parameter_sweep_results[:3,:,:]))
            fig, axs = plt.subplots(5, 1, sharex=True)
            fig.set_tight_layout(layout)
            for i, results_table in enumerate(this_parameter_sweep_results[:, :, :]):
                for j, performance in enumerate(performances, 1):
                    # plt.subplot(5, 1, j)
                    axs[j - 1].plot(results_table[:, 0],
                                    results_table[:, j])  # , color='C0', alpha=0.02, zorder=0)
                    # plt.axvline(np.log(2) / 90, color='black')
                    # plt.gca().locator_params(axis='x', tight=True, nbins=4)
                    # plt.gca().locator_params(axis='y', tight=True, nbins=3)
                    # plt.gca().set_rasterization_zorder(1)
                    axs[j - 1].set_ylabel(performance)
                    # plt.ylim(0, 1)
                    # plt.xlim(0, np.log(2) / 15.)
                    #         plt.gca().text(x_coord, y_coord, 'A', transform=plt.gca().transAxes)

                plt.xlabel(parameter)
                file_name = os.path.join(os.path.dirname(__file__),
                                         'output', 'performance_curves_allPP_' + parameter)
                # plt.savefig(file_name + '.pdf', dpi=600)
                fig.savefig(file_name + '.png', dpi=600)

    def test_ss_curves_cluster(self):
        ##Consider defining parameters and performance names globally in the future/in a separate file
        ##Consider gathering plotting parameters such as layout into a separate file
        #All model parameters as a list - order is important
        parameters = ['basal_transcription_rate',
                      'translation_rate',
                      'repression_threshold',
                      'time_delay',
                      'hill_coefficient']
        #Indexes of parameters we choose to look at - transcription rate & repression thd
        chosen_param = np.array([0])
        # define performances as an ordered list rather than an unordered set to be able to index later
        performances = ['Protein',
                        'Std',
                        'Period',
                        'Coherence',
                        'mRNA']

        eps = 1e-15 #Define the smallest detectable change
        #Define path to retrieve parameter points
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sampling_results_MiVe_expanded')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:, 0] > 5000,  # protein number
                                                   np.logical_and(model_results[:, 0] < 65000,  # protein_number
                                                                  #                                     np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                                  np.logical_and(model_results[:, 1] > 0.07,
                                                                                 model_results[:,
                                                                                 1] < 0.19))))  # standard deviation

        parameter_points = prior_samples[accepted_indices]
        parameter_points[:, 0:2] = np.log10(parameter_points[:, 0:2])
        print('number of accepted samples is')
        print(len(parameter_points))
        hzd_std = []
        # For each parameter
        for parameter in (parameters[idx] for idx in chosen_param):
            this_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                                                'repeated_relative_sweeps_MiVe_' + parameter + '.npy'))
            # For each initial parameter point
            for i, results_table in enumerate(this_parameter_sweep_results[:, :, 1:6]):
                if i==0:
                    #Define descriptors for the traces
                    dif = np.zeros((this_parameter_sweep_results.shape[0],results_table.shape[0]-1,results_table.shape[1]))
                    first_der = np.zeros((this_parameter_sweep_results.shape[0],dif.shape[1],dif.shape[2]))
                    second_der = np.zeros((this_parameter_sweep_results.shape[0],first_der.shape[1]-1,first_der.shape[2]))
                    monotony = np.zeros((this_parameter_sweep_results.shape[0],results_table.shape[1]))
                #dif[k] = trace[k]-trace[k-1]
                dif[i,:,:] = results_table[1:, :] - results_table[:-1, :]
                #first_der is actually sign:
                #           -1 Decrease
                #            0 Constant
                #           +1 Increase
                first_der[i,:,:] = np.where(dif[i,:,:] < -eps, -1, 1)
                #first_der = np.where(dif > eps, 1, first_der)
                first_der[i,:,:] = np.where(np.logical_and(dif[i,:,:] > -eps,dif[i,:,:] < eps), 0, first_der[i,:,:])
                #second_der is change of sign
                #           -2 Was increasing now decreasing
                #           -1 Was constant now decrease/Was increasing now constant
                #            0 Was any now the same
                #           +1 Was decreasing now constant/was constant now increasing
                #           +2 Was decreasing now increasing
                second_der[i,:,:] = first_der[i,1:, :] - first_der[i,:-1, :]
                #monotony: note that constant is not marked
                #           -1 Non-increasing
                #            0 Mixed
                #           +1 Non-decreasing
                #For all summary statistics
                for k,ss in enumerate(first_der[i,:,:].T):
                    monotony[i,k] = -1 if (ss != 1).all() & (ss==-1).any() else 0
                    if (ss != -1).all() & (ss == 1).any(): monotony[i,k] = +1
            #Gather all columns of monotony(std)==0
            hzd_std = first_der[np.where(monotony[:,1] == 0)][:,:, 1]
            hzd_std_df = pd.DataFrame(data=hzd_std)
            duplicates = hzd_std_df.duplicated()
            falses = duplicates.value_counts()
            #std = [first_der[ind] for ind in np.where(monotony.T[1] == 0)]
            for i,ind in enumerate(np.where(monotony[:,1] == 0)):
                if i==0:
                    hzd_std = first_der[ind][:, 1].T
                else:
                    np.concatenate(hzd_std,first_der[ind][:,1])
            #Produce clustering based on each SS monontony
            for i,perf_monotony,performance in zip(range(monotony.shape[1]+1),monotony.T,performances[0:2]):#enumerate(monotony.T):
                #Create dataframe for pairplot: all PP concatenated with corresponding column in monotony
                df_columns = parameters + [performance]
                indexes = np.where(perf_monotony == 0)
                #pps = parameter_points[indexes]
                this_performance_df = pd.DataFrame(data = np.concatenate((parameter_points[indexes],
                                                                          np.array([perf_monotony[indexes]]).T),
                                                                         axis = 1),
                                                   columns = df_columns)

                #Create pairplot
                fig = sns.pairplot(this_performance_df)#,hue=performance)
                #Save figure
                fig.savefig(os.path.join(os.path.dirname(__file__),'output',
                                         'effectsOf_'+parameter+'_sweep_on_'+performance+'_hzd'))
            #Save trace descriptors at parameter
            #file_name = os.path.join(os.path.dirname(__file__),'output','performance_curve_descriptors_'+parameter)
            #np.save(file_name)

                # file_name = os.path.join(os.path.dirname(__file__),
                #'output', 'performance_curves_allPP_' + parameter)
                # plt.savefig(file_name + '.pdf', dpi=600)
                # fig.savefig(file_name + '.png', dpi=600)

    def test_ss_curves_cluster_linearCorr(self):
        ##Consider defining parameters and performance names globally in the future/in a separate file
        ##Consider gathering plotting parameters such as layout into a separate file
        minWinSz = 3 #Minimum window size
        constants = [] #List of the summary statistics that remained constant for this parameter sweep: [p, pp, ss]
        rho = 0.1 #Penalising factor for window size -  the smaller the window the bigger the penalty
        #All model parameters as a list - order is important
        parameters = ['basal_transcription_rate',
                      'translation_rate',
                      'repression_threshold',
                      'time_delay',
                      'hill_coefficient']
        #Indexes of parameters we choose to look at - transcription rate & repression thd
        chosen_param = np.array([0])
        # define performances as an ordered list rather than an unordered set to be able to index later
        performances = ['Protein',
                        'Std',
                        'Period',
                        'Coherence',
                        'mRNA']
        #Define path to retrieve parameter points
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sampling_results_MiVe_expanded')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        mask1 = (model_results[:,0] > 5000) * (model_results[:,0] < 65000) * \
               (model_results[:,1] > 0.07) * (model_results[:,1] < 0.19)

        mask2 = (model_results[:,0] > 11000) * (model_results[:,0] < 30000) * \
               (model_results[:,1] > 0.07) * (model_results[:,1] < 0.19) * \
               (model_results[:,2] <2000)
        accepted_indices = np.where(mask1)[0] #indices of filter used for the sweeps
        accepted_indices2 = np.where(mask2)[0] #indices of desired filter
        index_dict = dict((value,idx) for idx,value in enumerate(list(accepted_indices)))
        reindex = [index_dict[x] for x in accepted_indices2] #indexes of desired filter in the sweep results

        parameter_points = prior_samples[accepted_indices2]
        parameter_points[:, 0:2] = np.log10(parameter_points[:, 0:2])
        print('number of accepted samples is')
        print(len(parameter_points))
        hzd_std = []

        # For each parameter
        for h,parameter in enumerate((parameters[idx] for idx in chosen_param)):
            this_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                                                'repeated_relative_sweeps_MiVe_' + parameter + '.npy'))
            if h == 0:
                # Define 5D matrix to store linear correlation indexes parameter x pp x ss x wSize x wStartIndex
                windowSweep = np.zeros((chosen_param.shape[0],
                                        accepted_indices2.shape[0], len(performances),
                                        this_parameter_sweep_results.shape[1] - minWinSz + 1,
                                        this_parameter_sweep_results.shape[1] - minWinSz + 1))
                # Define 4D matrix to store mean squares of the linear correlation indexes for each wSize
                windowMS = np.zeros((chosen_param.shape[0],accepted_indices2.shape[0], len(performances),
                                     this_parameter_sweep_results.shape[1] - minWinSz + 1))
            # For each initial parameter point
            for i, results_table in enumerate(this_parameter_sweep_results[reindex, :, 0:6]):
                # For each summary statistic
                for j, this_ss_sweep in enumerate(results_table[:,1:6].T):
                    this_p_sweep = results_table[:, j]
                    # For all window sizes
                    for k,wSize in enumerate(range(minWinSz, this_ss_sweep.shape[0]+1)):
                        wMask = np.concatenate((np.ones((1,wSize)),
                                                np.zeros((1,this_ss_sweep.shape[0]-wSize)))
                                               ,axis=1).astype(bool).flatten()
                        # Sweep window over this_ss_sweep and compute linear corr of this_ss_sweeps(wMask)
                        for l in range(0,this_ss_sweep.shape[0]-wSize+1): #while wMask[0,-1]!= 1:
                            #a=wMask[0:0,-1:-1]
                            aux = np.corrcoef(this_p_sweep[wMask],this_ss_sweep[wMask])[0,1]
                            aux2 = np.cov(this_p_sweep[wMask],this_ss_sweep[wMask])
                            if np.isnan(aux) & ([h,i,j] not in constants):
                                constants.append([h,i,j])
                            windowSweep[h,i,j,k,l]=np.corrcoef(this_p_sweep[wMask],this_ss_sweep[wMask])[0,1]
                            wMask = np.insert(wMask[:-1],0,wMask[-1])
                        # Compute mean sum of squares for the window size & penalize on small window size
                        windowMS[h, i, j, k] = sum(pow(windowSweep[h, i, j, k, :], 2)) / \
                                                   (this_ss_sweep.shape[0] - wSize + 1) - \
                                               rho*(this_ss_sweep.shape[0]-wSize)/(this_ss_sweep.shape[0]-minWinSz)




            #Gather all columns of monotony(std)==0
            hzd_std = first_der[np.where(monotony[:,1] == 0)][:,:, 1]
            hzd_std_df = pd.DataFrame(data=hzd_std)
            duplicates = hzd_std_df.duplicated()
            falses = duplicates.value_counts()
            #std = [first_der[ind] for ind in np.where(monotony.T[1] == 0)]
            for i,ind in enumerate(np.where(monotony[:,1] == 0)):
                if i==0:
                    hzd_std = first_der[ind][:, 1].T
                else:
                    np.concatenate(hzd_std,first_der[ind][:,1])
            #Produce clustering based on each SS monontony
            for i,perf_monotony,performance in zip(range(monotony.shape[1]+1),monotony.T,performances[0:2]):#enumerate(monotony.T):
                #Create dataframe for pairplot: all PP concatenated with corresponding column in monotony
                df_columns = parameters + [performance]
                indexes = np.where(perf_monotony == 0)
                #pps = parameter_points[indexes]
                this_performance_df = pd.DataFrame(data = np.concatenate((parameter_points[indexes],
                                                                          np.array([perf_monotony[indexes]]).T),
                                                                         axis = 1),
                                                   columns = df_columns)

                #Create pairplot
                fig = sns.pairplot(this_performance_df)#,hue=performance)
                #Save figure
                fig.savefig(os.path.join(os.path.dirname(__file__),'output',
                                         'effectsOf_'+parameter+'_sweep_on_'+performance+'_hzd'))
            #Save trace descriptors at parameter
            #file_name = os.path.join(os.path.dirname(__file__),'output','performance_curve_descriptors_'+parameter)
            #np.save(file_name)

                # file_name = os.path.join(os.path.dirname(__file__),
                #'output', 'performance_curves_allPP_' + parameter)
                # plt.savefig(file_name + '.pdf', dpi=600)
                # fig.savefig(file_name + '.png', dpi=600)

    def xest_open_sweeps(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'repeated_relative_sweeps_MiVe_')
        sweeper = np.load(saving_path + 'repression_threshold.npy')
        for key, value in sweeper.iteritems():
            print(key, value)

    def xest_a_make_abc_samples(self):
        print('making abc samples')
        ## generate posterior samples
        total_number_of_samples = 100000
        # total_number_of_samples = 10

        #         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate': (0.01, 120),
                        'translation_rate': (0.01, 60),
                        'repression_threshold': (0.01, 40000),
                        'time_delay': (5, 40),
                        'hill_coefficient': (2, 6)}

        my_prior_samples, my_results = hes5.generate_lookup_tables_for_abc(total_number_of_samples,
                                                                           number_of_traces_per_sample=200,
                                                                           saving_name='sampling_results_MiVe_expanded',
                                                                           prior_bounds=prior_bounds,
                                                                           prior_dimension='hill',
                                                                           logarithmic=True,
                                                                           simulation_timestep=1.0,
                                                                           simulation_duration=1500 * 5)

        self.assertEquals(my_prior_samples.shape,
                          (total_number_of_samples, 5))

    def xest_plot_posterior_distributions(self):
        # Load data
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_repeated')
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_massive')
                                   'sampling_results_MiVe_expanded')
        prior_performances = np.load(saving_path + '.npy')
        prior_parameters = np.load(saving_path + '_parameters.npy')

        protein_low = 11000
        protein_high = 29000
        std_low = 0.07
        std_high = 0.19

        option = 'priors'
        if option == 'full':
            accepted_indices = np.where(np.logical_and(prior_performances[:, 0] > protein_low,  # protein number
                                                       np.logical_and(prior_performances[:, 0] < protein_high,
                                                                      # protein_number
                                                                      np.logical_and(prior_performances[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     prior_performances[:,
                                                                                     1] > 0.05))))  # standard deviation
        #                                         np.logical_and(prior_performances[:,1]>0.05,  #standard deviation
        #                                                     prior_parameters[:,3]>20))))) #time_delay

        elif option == 'posteriors_2':
            accepted_indices = np.where(np.logical_and(prior_performances[:, 0] > protein_low,  # protein number
                                                       np.logical_and(prior_performances[:, 0] < protein_high,
                                                                      # protein_number
                                                                      np.logical_and(
                                                                          prior_performances[:, 1] < std_high,
                                                                          # standard deviation
                                                                          prior_performances[:,
                                                                          1] > std_low))))  # standard deviation
            #                                         np.logical_and(prior_performances[:,1]>0.05,  #standard deviation
            #                                                     prior_parameters[:,3]>20))))) #time_delay

        elif option == 'posteriors':
            accepted_indices = np.where(np.logical_and(prior_performances[:, 0] > protein_low,  # protein number
                                                       prior_performances[:, 0] < protein_high))
            # protein_number
            #                                          np.logical_and(prior_performances[:,6]<0.15,  #standard deviation
            # prior_performances[:, 1] > 0.05)))
        elif option == 'priors':
            accepted_indices = np.where(prior_performances[:, 0] > 0)  # protein number

        else:
            ValueError('could not identify posterior option')
        #
        my_posterior_parameters = prior_parameters[accepted_indices]
        performances = prior_performances[accepted_indices]

        # Organise data
        my_posterior_parameters[:, 2] /= 10000  # Repression threshold 1e-4
        parameter_frame = pd.DataFrame(data=my_posterior_parameters,
                                       columns=['Transcription rate',
                                                'Translation rate',
                                                'Repression threshold 1e-4',
                                                'Transcription delay',
                                                'Hill coefficient'])
        performance_frame = pd.DataFrame(data=performances[:, range(0, 5)],
                                         columns=['Mean protein stochastic',
                                                  'STD stochastic',
                                                  'Period stochastic',
                                                  'Coherence stochastic',
                                                  'Mean mRNA stochastic'])
        parameter_bounds = pd.DataFrame({'Transcription rate': (0.01, 120),
                                         'Translation rate': (0.01, 60),
                                         'Repression threshold 1e-4': (0.000001, 4),
                                         'Transcription delay': (5, 40),
                                         'Hill coefficient': (2, 6)})

        performance_bounds = pd.DataFrame({'Mean protein stochastic': (11000, 29000),
                                           'STD stochastic': (performances[:, 1].min(), performances[:, 1].max()),
                                           'Period stochastic': (performances[:, 2].min(), performances[:, 2].max()),
                                           'Coherence stochastic': (performances[:, 3].min(), performances[:, 3].max()),
                                           'Mean mRNA stochastic': (
                                               performances[:, 4].min(), performances[:, 4].max())})
        ##########################################
        #quantile_performance = performance_frame.quantile(0.75)
        #selection = performance_frame[performance_frame<quantile_performance]
        #selection2 = np.percentile(performance_frame.
        ############################################
        parameter_pairplot = hes5.plot_posterior_distributions_MiVe(parameter_frame, parameter_bounds)
        parameter_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                                'output', 'Parameter_pairplot_extended_abc_MiVe_' + option + '.png'))
        performance_pairplot = hes5.plot_posterior_distributions_MiVe(performance_frame, performance_bounds,
                                                                      logarithmic=False)
        performance_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                                  'output',
                                                  'Performances_pairplot_extended_abc_MiVe_' + option + '.png'))

        print('Number of accepted samples is ')
        print(len(my_posterior_parameters))
        # Save posterior parameter set and corresponding performances

        # performance_frame = pd.DataFrame(data=performances,
        #                                  columns=['Mean protein stochastic',
        #                                           'STD stochastic',
        #                                           'Period stochastic',
        #                                           'Coherence stochastic',
        #                                           'Mean mRNA stochastic',
        #                                           'Mean protein deterministic',
        #                                           'STD deterministic',
        #                                           'Period deterministic',
        #                                           'Coherence deterministic',
        #                                           'Mean mRNA deterministic',
        #                                           'High frequency weight',
        #                                           'Fluctuation weight'])

    def xest_plot_model_traces(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sweeping_results_MiVe_')
        # Plot design
        font = {'family': 'serif',
                'weight': 'light',
                'size': 8}
        mpl.rc('font', **font)  # pass in the font dict as kwargs
        # Generate langevin traces sweeping parameters
        parameter_bounds = {'basal_transcription_rate': (0.5, 3),
                            'translation_rate': (0.5, 10),
                            'repression_threshold': (0.01, 40000),
                            'transcription_delay': (5, 40),
                            'hill_coefficient': (2, 6)}
        parameter_bounds = pd.DataFrame(parameter_bounds)
        parameter_defaults = {'basal_transcription_rate': [1],
                              'translation_rate': [1],
                              'repression_threshold': [10000],
                              'transcription_delay': [29],
                              'hill_coefficient': [5]}
        parameter_defaults = pd.DataFrame(parameter_defaults)
        noPartitions = 3
        # Transcription rate
        l = []
        for parameter in parameter_bounds.columns:
            grid = np.linspace(parameter_bounds.loc[0, parameter], parameter_bounds.loc[1, parameter], noPartitions)
            parameters = parameter_defaults
            plt.figure(parameter)
            for i in range(0, noPartitions):
                parameterValue = grid[i]
                parameters.loc[0, parameter] = parameterValue
                trace = hes5.generate_langevin_trajectory(duration=2000,
                                                          repression_threshold=parameters.loc[
                                                              0, 'repression_threshold'],
                                                          hill_coefficient=parameters.loc[0, 'hill_coefficient'],
                                                          mRNA_degradation_rate=np.log(2) / 30,
                                                          protein_degradation_rate=np.log(2) / 90,
                                                          basal_transcription_rate=parameters.loc[
                                                              0, 'basal_transcription_rate'],
                                                          translation_rate=parameters.loc[0, 'translation_rate'],
                                                          transcription_delay=parameters.loc[0, 'transcription_delay'],
                                                          initial_mRNA=0,
                                                          initial_protein=0,
                                                          equilibration_time=0.0,
                                                          extrinsic_noise_rate=0.0,
                                                          transcription_noise_amplification=1.0,
                                                          timestep=0.5
                                                          )
                trace = pd.DataFrame({'time': trace[:, 0],
                                      'mRNA': trace[:, 1],
                                      'protein': trace[:, 2]})
                plt.subplot(2, 1, 1)
                line, = plt.plot(trace.loc[:, 'time'], trace.loc[:, 'mRNA'])
                l.append(line)
                # plt.title(str(round(parameterValue,2)))
                # plt.xlabel('time')
                if i == 0:
                    plt.ylabel('mRNA')
                plt.subplot(2, 1, 2)
                plt.plot(trace.loc[:, 'time'], trace.loc[:, 'protein'])
                # plt.title(parameter + ' = ' + str(parameterValue))
                plt.xlabel('time')
                if i == 0:
                    plt.ylabel('protein')
            plt.suptitle((parameter.replace('_', ' ')).upper())
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.8, top=0.8, wspace=0.6, hspace=0.4)
            plt.figlegend((l[0], l[1], l[2]), (str(grid[0]), str(grid[1]), str(grid[2])), loc='upper right')
            plt.savefig(saving_path + parameter)

        # Repression threshold
        # Transcription delay
        # Hill coefficient
