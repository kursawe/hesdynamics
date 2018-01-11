import unittest
import os.path
import sys
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestSimpleHes5Model(unittest.TestCase):
                                 
    def test_make_simple_parameter_sweep(self):
        # First, vary the rescaled repression threshold
        ########
        #
        # REPRESSION THRESHOLD
        #
        ########
        number_of_parameter_points = 100
        repression_threshold_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for p0 in np.linspace(0.01,1,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 60,
                                                         repression_threshold = p0,
                                                         mRNA_degradation_rate = np.log(2)/30*29.0,
                                                         protein_degradation_rate = np.log(2)/90*29.0,
                                                         repression_delay = 1.0,
                                                         initial_mRNA = 3.0/(29),
                                                         initial_protein = 100.0/np.power(29.0,2) )
            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            repression_threshold_results[index,0] = p0
            repression_threshold_results[index,1] = this_period
            repression_threshold_results[index,2] = this_amplitude
            repression_threshold_results[index,3] = this_variation
            index +=1

        my_figure = plt.figure( figsize = (4.5, 4.5) )
        my_figure.add_subplot(321)
        plt.plot(repression_threshold_results[:,0],
                 repression_threshold_results[:,1], color = 'black')
        plt.axvline(100.0/np.power(29.0,2) )
        plt.xlabel('Rescaled repression threshold')
        plt.ylabel('Rescaled period')

        my_figure.add_subplot(322)
        plt.plot(repression_threshold_results[:,0],
                 repression_threshold_results[:,2], color = 'black')
        plt.axvline( 100.0/np.power(29.0,2) )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('Rescaled repression threshold')
        plt.ylabel('Relative amplitude')

        ########
        #
        # MRNA DEGRADATION
        #
        ########       
        number_of_parameter_points = 100
        mrna_degradation_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for mu_m in np.linspace(0.01,1,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 60,
                                                         repression_threshold = 100.0/np.power(29.0,2),
                                                         mRNA_degradation_rate = mu_m,
                                                         protein_degradation_rate = np.log(2)/90*29.0,
                                                         repression_delay = 1.0,
                                                         initial_mRNA = 3.0/(29),
                                                         initial_protein = 100.0/np.power(29.0,2) )
            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            mrna_degradation_results[index,0] = mu_m
            mrna_degradation_results[index,1] = this_period
            mrna_degradation_results[index,2] = this_amplitude
            mrna_degradation_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(323)
        plt.plot(mrna_degradation_results[:,0],
                 mrna_degradation_results[:,1], color = 'black')
        plt.axvline( np.log(2)/30*29.0 )
        plt.xlabel('Rescaled mRNA degradation')
        plt.xlim(0,1)
        plt.ylabel('Rescaled period')

        my_figure.add_subplot(324)
        plt.plot(mrna_degradation_results[:,0],
                 mrna_degradation_results[:,2], color = 'black')
        plt.axvline( np.log(2)/30*29.0 )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlim(0,1)
        plt.xlabel('Rescaled mRNA degradation')
        plt.ylabel('Relative amplitude')       
        
        ########
        #
        # PROTEIN DEGRADATION
        #
        ########       
        number_of_parameter_points = 100
        protein_degradation_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for mu_p in np.linspace(0.01,1,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 60,
                                                         repression_threshold = 100.0/np.power(29.0,2),
                                                         mRNA_degradation_rate = np.log(2)/30*29.0,
                                                         protein_degradation_rate = mu_p,
                                                         repression_delay = 1.0,
                                                         initial_mRNA = 3.0/(29),
                                                         initial_protein = 100.0/np.power(29.0,2) )
            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            protein_degradation_results[index,0] = mu_p
            protein_degradation_results[index,1] = this_period
            protein_degradation_results[index,2] = this_amplitude
            protein_degradation_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(325)
        plt.plot(protein_degradation_results[:,0],
                 protein_degradation_results[:,1], color = 'black')
        plt.axvline( np.log(2)/90*29.0 )
        plt.xlabel('Rescaled protein degradation')
        plt.xlim(0,1)
        plt.ylabel('Rescaled period')

        my_figure.add_subplot(326)
        plt.plot(protein_degradation_results[:,0],
                 protein_degradation_results[:,2], color = 'black')
        plt.axvline( np.log(2)/90*29.0 )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlim(0,1)
        plt.xlabel('Rescaled protein degradation')
        plt.ylabel('Relative amplitude')
        
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','rescaled_parameter_sweep.pdf'))

    def test_investigate_discontinuities_in_parameter_sweep(self): 
        ########
        #
        # MRNA DEGRADATION REPEAT
        #
        ########       
        number_of_parameter_points = 100
        mrna_degradation_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for mu_m in np.linspace(0.01,1,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 60,
                                                         repression_threshold = 100.0/np.power(29.0,2),
                                                         mRNA_degradation_rate = mu_m,
                                                         protein_degradation_rate = np.log(2)/90*29.0,
                                                         repression_delay = 1.0,
                                                         initial_mRNA = 3.0/(29),
                                                         initial_protein = 100.0/np.power(29.0,2) )
            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            mrna_degradation_results[index,0] = mu_m
            mrna_degradation_results[index,1] = this_period
            mrna_degradation_results[index,2] = this_amplitude
            mrna_degradation_results[index,3] = this_variation
            index +=1

        my_figure = plt.figure( figsize = (4.5, 2.5) )
        my_figure.add_subplot(121)
        plt.plot(mrna_degradation_results[:,0],
                 mrna_degradation_results[:,2], color = 'black')
        plt.axvline( np.log(2)/30*29.0, color = 'blue' )
        plt.axvline( 0.75, color = 'green', ls = '--', dashes = [3,1] )
        plt.axvline( 0.2, color = 'orange', ls = '--', dashes = [2,0.5] )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlim(0,1)
        plt.xlabel('Rescaled mRNA degradation')
        plt.ylabel('Relative amplitude')  
        
        my_figure.add_subplot(122)
        first_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 100.0/np.power(29.0,2),
                                             mRNA_degradation_rate = np.log(2)/30*29.0,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        second_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 100.0/np.power(29.0,2),
                                             mRNA_degradation_rate = 0.75,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        third_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 100.0/np.power(29.0,2),
                                             mRNA_degradation_rate = 0.2,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )


        plt.plot(first_trajectory[:,0], first_trajectory[:,2], color = 'blue')
        plt.plot(second_trajectory[:,0], second_trajectory[:,2], color = 'green', linestyle = '--',
                 dashes = [3,1])
        plt.plot(third_trajectory[:,0], third_trajectory[:,2], color = 'orange', linestyle = '--',
                 dashes = [2,0.5])
#         plt.ylim(0,1.)
        plt.xlabel('Rescaled time')
        plt.ylabel('Rescaled protein')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','discontinuities_in_parameter_sweep_investigation.pdf'))
        
    def test_investigate_different_p0_values(self):
        
        my_figure = plt.figure( figsize = (4.5, 2.5) )
        first_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 100.0/np.power(29.0,2),
                                             mRNA_degradation_rate = np.log(2)/30*29.0,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        second_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 1.0,
                                             mRNA_degradation_rate = np.log(2)/30*29.0,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        third_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 0.8,
                                             mRNA_degradation_rate = np.log(2)/30*29.0,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        fourth_trajectory = hes5.generate_single_trajectory( duration = 60,
                                             repression_threshold = 0.05,
                                             mRNA_degradation_rate = np.log(2)/30*29.0,
                                             protein_degradation_rate = np.log(2)/90*29.0,
                                             repression_delay = 1.0,
                                             initial_mRNA = 3.0/(29),
                                             initial_protein = 100.0/np.power(29.0,2) )

        standard_p0 = 100.0/np.power(29.0,2),
        plt.plot(first_trajectory[:,0], first_trajectory[:,2], color = 'blue', 
                 label = r'$p_0=0.12$' )
        plt.plot(second_trajectory[:,0], second_trajectory[:,2], color = 'green', linestyle = '--',
                 dashes = [3,1], label = r'$p_0=1.0$')
        plt.plot(third_trajectory[:,0], third_trajectory[:,2], color = 'orange', linestyle = '--',
                 dashes = [2,0.5], label = r'$p_0=0.8$')
        plt.plot(fourth_trajectory[:,0], fourth_trajectory[:,2], color = 'purple', linestyle = '--',
                 dashes = [0.5,0.5], label = r'$p_0=0.05$')
#         plt.ylim(0,1.)
        plt.xlabel('Rescaled time')
        plt.ylabel('Rescaled protein')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','p0_investigation.pdf'))
 
    def test_make_full_parameter_sweep(self):
        ########
        #
        # REPRESSION THRESHOLD
        #
        ########
        number_of_parameter_points = 100
        repression_threshold_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for p0 in np.linspace(1,60000,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                         repression_threshold = p0,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            repression_threshold_results[index,0] = p0
            repression_threshold_results[index,1] = this_period
            repression_threshold_results[index,2] = this_amplitude
            repression_threshold_results[index,3] = this_variation
            index +=1

        my_figure = plt.figure( figsize = (4.5, 7.5) )
        my_figure.add_subplot(521)
        plt.plot(repression_threshold_results[:,0]/10000,
                 repression_threshold_results[:,1], color = 'black')
#         plt.axvline( 23000 )
        plt.axvline( 2.3 )
#         plt.gca().locator_params(axis='x', tight = True, nbins=3)
#         plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.0e'))
#         plt.gca().ticklabel_format(axis = 'x', style = 'sci')
        plt.ylim(0,700)
        plt.xlabel('Repression threshold/1e4')
        plt.ylabel('Period [min]')

        my_figure.add_subplot(522)
#         plt.plot(repression_threshold_results[:,0],
        plt.plot(repression_threshold_results[:,0]/10000,
                 repression_threshold_results[:,2], color = 'black')
#         plt.axvline( 23000 )
        plt.axvline( 2.3 )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('Repression threshold/1e4')
        plt.ylabel('Relative amplitude')
        plt.ylim(0,2)

        ########
        #
        # MRNA DEGRADATION
        #
        ########       
        number_of_parameter_points = 100
        mrna_degradation_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for mu_m in np.linspace(0.00,np.log(2)/15,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 720.0,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = mu_m,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            mrna_degradation_results[index,0] = mu_m
            mrna_degradation_results[index,1] = this_period
            mrna_degradation_results[index,2] = this_amplitude
            mrna_degradation_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(523)
        plt.plot(mrna_degradation_results[:,0],
                 mrna_degradation_results[:,1], color = 'black')
        plt.axvline( np.log(2)/30 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('mRNA degradation [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(524)
        plt.plot(mrna_degradation_results[:,0],
                 mrna_degradation_results[:,2], color = 'black')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.axvline( np.log(2)/30 )
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('mRNA degradation [1/min]')
        plt.ylabel('Relative amplitude')       
        plt.ylim(0,2)
        
        ########
        #
        # PROTEIN DEGRADATION
        #
        ########       
        number_of_parameter_points = 100
        protein_degradation_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for mu_p in np.linspace(0.00,np.log(2)/15,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 720.0,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = mu_p,
                                                         translation_rate = 230,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            protein_degradation_results[index,0] = mu_p
            protein_degradation_results[index,1] = this_period
            protein_degradation_results[index,2] = this_amplitude
            protein_degradation_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(525)
        plt.plot(protein_degradation_results[:,0],
                 protein_degradation_results[:,1], color = 'black')
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Protein degradation [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(526)
        plt.plot(protein_degradation_results[:,0],
                 protein_degradation_results[:,2], color = 'black')
        plt.axvline( np.log(2)/90 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('Protein degradation [1/min]')
        plt.ylabel('Relative amplitude')
        plt.ylim(0,2)
        
        ########
        #
        # TIME DELAY
        #
        ########       
        number_of_parameter_points = 100
        time_delay_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for tau in np.linspace(5.0,40.0,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 720.0,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = tau,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            time_delay_results[index,0] = tau
            time_delay_results[index,1] = this_period
            time_delay_results[index,2] = this_amplitude
            time_delay_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(527)
        plt.plot(time_delay_results[:,0],
                 time_delay_results[:,1], color = 'black')
        plt.axvline( 29.0 )
#         plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time delay [min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(528)
        plt.plot(time_delay_results[:,0],
                 time_delay_results[:,2], color = 'black')
        plt.axvline( 29.0 )
#         plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('Time delay [min]')
        plt.ylabel('Relative amplitude')
        plt.ylim(0,2)
 
        ########
        #
        # TRANSLATION RATE
        #
        ########       
        number_of_parameter_points = 100
        translation_rate_results = np.zeros((number_of_parameter_points,4))
        index = 0
        for alpha_p in np.linspace(1.0,400.0,number_of_parameter_points):
            this_trajectory = hes5.generate_single_trajectory( duration = 720.0,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = alpha_p,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

            this_period, this_amplitude, this_variation = hes5.measure_period_and_amplitude_of_signal(this_trajectory[:,0],
                                                         this_trajectory[:,1])

            translation_rate_results[index,0] = alpha_p
            translation_rate_results[index,1] = this_period
            translation_rate_results[index,2] = this_amplitude
            translation_rate_results[index,3] = this_variation
            index +=1

        my_figure.add_subplot(529)
        plt.plot(translation_rate_results[:,0],
                 translation_rate_results[:,1], color = 'black')
        plt.axvline( 230 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         plt.gca().locator_params(axis='y', tight = True, nbins=5)
        plt.xlabel('Translation rate [1/min]')
        plt.ylabel('Period [min]')
        plt.ylim(0,700)

        my_figure.add_subplot(5,2,10)
        plt.plot(translation_rate_results[:,0],
                 translation_rate_results[:,2], color = 'black')
        plt.axvline( 230 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         plt.gca().locator_params(axis='y', tight = True, nbins=5)
#         plt.fill_between(repression_threshold_results[:,0],
#                          repression_threshold_results[:,2] + repression_threshold_results[:,3],
#                          np.max(repression_threshold_results[:,2]- repression_threshold_results[:,3],0),
#                          lw = 0, color = 'grey')
        plt.xlabel('Translation rate [1/min]')
        plt.ylabel('Relative amplitude')
        plt.ylim(0,2)

        plt.tight_layout()

        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','full_parameter_sweep.pdf'))
        
    def test_different_tau_values(self):
        my_figure = plt.figure( figsize = (4.5, 2.5) )
        first_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        second_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 12.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        third_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 20.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        fourth_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 40.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        standard_p0 = 100.0/np.power(29.0,2),
        plt.plot(first_trajectory[:,0], first_trajectory[:,2], color = 'blue', 
                 label = r'$\tau=29$' )
        plt.plot(second_trajectory[:,0], second_trajectory[:,2], color = 'green', linestyle = '--',
                 dashes = [3,1], label = r'$\tau=12$')
        plt.plot(third_trajectory[:,0], third_trajectory[:,2], color = 'orange', linestyle = '--',
                 dashes = [2,0.5], label = r'$\tau=20$')
        plt.plot(fourth_trajectory[:,0], fourth_trajectory[:,2], color = 'purple', linestyle = '--',
                 dashes = [0.5,0.5], label = r'$\tau=40$')
#         plt.ylim(0,1.)
        plt.xlabel('Rescaled time')
        plt.ylabel('Rescaled protein')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','tau_investigation.pdf'))
        
    def test_different_protein_degradation_values(self):
        my_figure = plt.figure( figsize = (6.5, 2.5) )
        first_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 29.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        second_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30.0,
                                                         protein_degradation_rate = np.log(2)/30.0,
                                                         translation_rate = 230,
                                                         repression_delay = 12.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        third_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/90.0,
                                                         protein_degradation_rate = np.log(2)/90.0,
                                                         translation_rate = 230,
                                                         repression_delay = 20.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        fourth_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                          repression_threshold = 23000,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         translation_rate = 230,
                                                         repression_delay = 40.0,
                                                         initial_mRNA = 3.0,
                                                         initial_protein = 23000)

        standard_p0 = 100.0/np.power(29.0,2),
        plt.plot(first_trajectory[:,0], first_trajectory[:,2], color = 'blue', 
                 label = r'$t_{1/2,M}= 30, t_{1/2,P} = 90$' )
        plt.plot(second_trajectory[:,0], second_trajectory[:,2], color = 'green', linestyle = '--',
                 dashes = [3,1], label = r'$t_{1/2,M}= 30, t_{1/2,P} = 30$')
        plt.plot(third_trajectory[:,0], third_trajectory[:,2], color = 'orange', linestyle = '--',
                 dashes = [2,0.5], label = r'$t_{1/2,M}= 90, t_{1/2,P} = 90$')
        plt.plot(fourth_trajectory[:,0], fourth_trajectory[:,2], color = 'purple', linestyle = '--',
                 dashes = [0.5,0.5], label = r'$\mu_M=\mu_P=0.03$ (Monk)')
#         plt.ylim(0,1.)
        plt.xlabel('Time [min]')
        plt.ylabel('Protein')
        plt.tight_layout()
        # Shrink current axis by 20%
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.5, box.height])

        # Put a legend to the right of the current axis
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(os.path.join(os.path.dirname(__file__), 
                    'output','degradation_investigation.pdf'))    