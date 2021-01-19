# import PyDDE
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.interpolate
import multiprocessing as mp
from numba import jit
from numpy import ndarray, number
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn.apionly as sns
import pandas as pd
import socket
import jitcdde
import warnings
import seaborn as sns
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
# try:
#     import gpflow
# except ImportError:
#     print('Could not import gpflow. Gpflow will not be available for GP regression. This will not affect any functions used in our publications.')
import sklearn.gaussian_process as gp
import GPy
try:
    import george
except ImportError:
    print('Could not import george. George will not be available for GP regression. This will not affect any functions used in our publications.')

domain_name = socket.getfqdn()
if domain_name == 'jochen-ThinkPad-S1-Yoga-12':
    number_of_available_cores = 2
elif domain_name.endswith('csf3.alces.network'):
    number_of_available_cores = 24
else:
#     number_of_available_cores = 1
    number_of_available_cores = mp.cpu_count()

def generate_deterministic_trajectory( duration = 720,
                                       repression_threshold = 10000,
                                       hill_coefficient = 5,
                                       mRNA_degradation_rate = np.log(2)/30,
                                       protein_degradation_rate = np.log(2)/90,
                                       basal_transcription_rate = 1,
                                       translation_rate = 1,
                                       transcription_delay = 29,
                                       initial_mRNA = 0,
                                       initial_protein = 0,
                                       for_negative_times = 'initial',
                                       integrator = 'agnostic'):
    '''Generate one trace of the Hes5 model. This function implements the deterministic model in
    Monk, Current Biology (2003).

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    for_negative_times : string
        decides what protein and MRNA values are assumed for negative times. This
        is necessary since function values for t-tau are required for all t>0.
        The values 'initial', 'zero' and 'no_negative' are supported. The default 'initial' will assume that protein and
        mRNA numbers were constant at the values of the initial condition for all negative times.
        If 'zero' is chosen, then the protein and mRNA numbers are assumed to be 0 at negative times.
        If 'no_negative' is chosen, no assumptions are made for negative times, and transcription
        is blocked until transcription_delay has passed.

    integrator : string
        'agnostic' or 'PyDDE' are allowed integrators. If 'agnostic' is used, the langevin equation
        with noise_strength zero will be employed. In this case the argument for 'for_negative_times'
        will be ignored

    Returns
    -------

    trace : ndarray
        2 dimenstional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''

    if integrator == 'agnostic':
        trace = generate_agnostic_noise_trajectory(duration,
                                           repression_threshold,
                                           hill_coefficient,
                                           mRNA_degradation_rate,
                                           protein_degradation_rate,
                                           basal_transcription_rate,
                                           translation_rate,
                                           transcription_delay,
                                           mRNA_noise_strength = 0,
                                           protein_noise_strength = 0,
                                           initial_mRNA = initial_mRNA,
                                           initial_protein = initial_protein,
                                           equilibration_time = 0,
                                           time_step = 0.01,
                                           sampling_frequency = 1)

        return trace

    elif integrator == 'PyDDE':
        hes5_dde = PyDDE.dde()
        initial_condition = np.array([initial_mRNA,initial_protein])
        # The coefficients (constants) in the equations
        if for_negative_times == 'initial':
            negative_times_indicator = 0.0
        elif for_negative_times == 'zero':
            negative_times_indicator = 1.0
        elif for_negative_times == 'no_negative':
            negative_times_indicator = 2.0
        else:
            ValueError("The parameter set for for_negative_times could not be interpreted.")

        parameters = np.array([repression_threshold,
                               hill_coefficient,
                               mRNA_degradation_rate,
                               protein_degradation_rate,
                               basal_transcription_rate,
                               translation_rate,
                               transcription_delay,
                               negative_times_indicator])

        hes5_dde.dde(y=initial_condition, times=np.arange(0.0, duration, 1.0),
                     func=hes5_ddegrad, parms=parameters,
                     tol=0.000005, dt=0.01, hbsize=10000, nlag=1, ssc=[0.0, 0.0])
                     #hbsize is buffer size, I believe this is how many values in the past are stored
                     #nlag is the number of delay variables (tau_1, tau2, ... taun_lag)
                     #ssc means "statescale" and would somehow only matter for values close to 0

        this_data = hes5_dde.data
        return hes5_dde.data

def hes5_ddegrad(y, parameters, time):
    '''Gradient of the Hes5 delay differential equation for
    deterministic runs of the model.
    It evaluates the right hand side of DDE 1 in Monk(2003).

    Parameters
    ----------
    y : ndarray
        vector of the form [mRNA, protein] contain the concentration of these species at time t

    parameters : ndarray
        vector of the form [repression_threshold,  hill_coefficient, mRNA_degradation_rate,
                            protein_degradation_rate, basal_transcription_rate, translation_rate,
                            transcription_delay, negative_times_indicator]
        containing the value of these parameters.
        The value of negative_times_indicator corresponds to for_negative_times in generate_deterministic_trajectory().
        The value 0.0 corresponds to the option 'initial', whereas 1.0 corresponds to 'zero',
        and 2.0 corresponds to 'no_negative'.

    time : float
        time at which the gradient is calculated

    Returns
    -------

    gradient : ndarray
        vector of the form [dmRNA, dProtein] containing the evaluated right hand side of the
        delay differential equation for the species concentrations provided in y, the given
        parameters, and at time t.
    '''

    repression_threshold = float(parameters[0]); #P0
    hill_coefficient = parameters[1]; #NP
    mRNA_degradation_rate = parameters[2]; #MuM
    protein_degradation_rate = parameters[3]; #Mup
    basal_transcription_rate = parameters[4]; #alpha_m
    translation_rate = parameters[5]; #alpha_p
    time_delay = parameters[6]; #tau
    negative_times_indicator = parameters[7] #string for negative times

    if negative_times_indicator == 0.0:
        for_negative_times = 'initial'
    elif negative_times_indicator == 1.0:
        for_negative_times = 'zero'
    elif negative_times_indicator == 2.0:
        for_negative_times = 'no_negative'
    else:
        ValueError("Could not interpret the value of for_negative_times")

    mRNA = float(y[0])
    protein = float(y[1])

    if (time>time_delay):
        past_protein = PyDDE.pastvalue(1,time-time_delay,0)
    elif time>0.0:
        if for_negative_times == 'initial':
            past_protein = PyDDE.pastvalue(1,0.0,0)
        elif for_negative_times == 'zero':
            past_protein = 0.0
    else:
        past_protein = protein

    dprotein = translation_rate*mRNA - protein_degradation_rate*protein

    if for_negative_times != 'no_negative':
        hill_function_value = 1.0/(1.0+pow(past_protein/repression_threshold,hill_coefficient))
        dmRNA = basal_transcription_rate*hill_function_value-mRNA_degradation_rate*mRNA
    else:
        if time < time_delay:
            dmRNA = -mRNA_degradation_rate*mRNA
        else:
            hill_function_value = 1.0/(1.0+pow(past_protein/repression_threshold,hill_coefficient))
            dmRNA = basal_transcription_rate*hill_function_value-mRNA_degradation_rate*mRNA

    return np.array( [dmRNA,dprotein] )

def generate_deterministic_goodfellow_trajectory( duration = 7200,
                                                  basal_mRNA_transcription_rate = 1.0,
                                                  basal_miRNA_transcription_rate = 1.0,
                                                  translation_rate = 10,
                                                  repression_threshold_protein_on_mRNA = 100,
                                                  repression_threshold_protein_on_miRNA = 100,
                                                  repression_threshold_miRNA_on_mRNA = 100,
                                                  repression_threshold_miRNA_on_protein = 100,
                                                  hill_coefficient_protein_on_mRNA = 5,
                                                  hill_coefficient_protein_on_miRNA = 5,
                                                  hill_coefficient_miRNA_on_mRNA = 5,
                                                  hill_coefficient_miRNA_on_protein = 100,
                                                  transcription_delay = 19,
                                                  upper_mRNA_degradation_rate = 0.03,
                                                  lower_mRNA_degradation_rate = 0.03,
                                                  protein_degradation_rate = 0.03,
                                                  miRNA_degradation_rate = 0.00001,
                                                  initial_mRNA = 3,
                                                  initial_protein = 100,
                                                  initial_miRNA = 1,
                                                  for_negative_times='initial'):
    '''Generate one trace of the Goodfellow model. This function implements the deterministic model in
    Goodfellow, Nature Communications (2014).

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold_protein_on_mRNA : float
        repression threshold, Hes autorepresses its own transcription if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Goodfellow paper

    repression_threshold_protein_on_miRNA : float
        repression threshold, Hes represses production of micro RNA if the Hes copynumber is larger
        than this repression threshold. Corresponds to P1 in the Goodfellow paper.

    repression_threshold_miRNA_on_mRNA : float
        repression threshold, the micro RNA represses Hes transcription if the micro RNA copynumber is larger
        than this repression threshold. Corresponds to r0 in the Goodfellow paper

    hill_coefficient_protein_on_mRNA : float
        exponent in the hill function regulating the Hes autorepression of its own transcription.
        Small values make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold, corresponds to n0 in the
        Goodfellow paper.

    hill_coefficient_miRNA_on_mRNA : float
        exponent in the hill function regulating the impact of the micro RNA on mRNA translation. Small values
        make the response more shallow, whereas large values will lead to a switch-like response if the miRNA
        concentration exceeds the repression threshold. Corresponds to m0 in the Goodfellow paper.

    hill_coefficient_protein_on_miRNA : float
        exponent in the hill function regulating the repression of miRNA transcription by Hes. Small values
        make the response more shallow, whereas large values will lead to a switch-like response if the protein
        concentration exceeds the repression threshold. Corresponds to n1 in the Goodfellow paper.

    upper_mRNA_degradation_rate : float
        upper bound for the rate at which mRNA is degraded, in copynumber per minute. Corresponds to b_l in the
        Goodfellow paper.

    lower_mRNA_degradation_rate : float
        lower bound for the rate at which mRNA is degraded, in copynumber per minute. Corresponds to b_u in the
        Goodfellow paper.

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    for_negative_times : string
        decides what protein and MRNA values are assumed for negative times. This
        is necessary since function values for t-tau are required for all t>0.
        The values 'initial', 'zero' and 'no_negative' are supported. The default 'initial' will assume that protein and
        mRNA numbers were constant at the values of the initial condition for all negative times.
        If 'zero' is chosen, then the protein and mRNA numbers are assumed to be 0 at negative times.
        If 'no_negative' is chosen, no assumptions are made for negative times, and transcription
        is blocked until transcription_delay has passed.

    Returns
    -------

    trace : ndarray
        2 dimenstional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number, fourth column is miRNA copy number.
    '''
    hes5_dde = PyDDE.dde()
    initial_condition = np.array([initial_mRNA,initial_miRNA,initial_protein])
    # The coefficients (constants) in the equations
    if for_negative_times == 'initial':
        negative_times_indicator = 0.0
    elif for_negative_times == 'zero':
        negative_times_indicator = 1.0
    elif for_negative_times == 'no_negative':
        negative_times_indicator = 2.0
    else:
        ValueError("The parameter set for for_negative_times could not be interpreted.")

    parameters = np.array([ basal_mRNA_transcription_rate,
                            basal_miRNA_transcription_rate,
                            translation_rate,
                            repression_threshold_protein_on_mRNA,
                            repression_threshold_protein_on_miRNA,
                            repression_threshold_miRNA_on_mRNA,
                            repression_threshold_miRNA_on_protein,
                            hill_coefficient_protein_on_mRNA,
                            hill_coefficient_protein_on_miRNA,
                            hill_coefficient_miRNA_on_mRNA,
                            hill_coefficient_miRNA_on_protein,
                            transcription_delay,
                            upper_mRNA_degradation_rate,
                            lower_mRNA_degradation_rate,
                            protein_degradation_rate,
                            miRNA_degradation_rate,
                            negative_times_indicator ])

    hes5_dde.dde(y=initial_condition, times=np.arange(0.0, duration, 1.0),
                 func=goodfellow_ddegrad, parms=parameters,
                 tol=0.00005, dt=1.0, hbsize=1000, nlag=1, ssc=[0.0, 0.0,0.0])
                 #hbsize is buffer size, I believe this is how many values in the past are stored
                 #nlag is the number of delay variables (tau_1, tau2, ... taun_lag)
                 #ssc means "statescale" and would somehow only matter for values close to 0

    this_data = hes5_dde.data
    return hes5_dde.data

def goodfellow_ddegrad(y, parameters, time):
    '''Gradient of the Hes5 delay differential equation for
    deterministic runs of the model.
    It evaluates the right hand side of DDE 1 in Monk(2003).

    Parameters
    ----------
    y : ndarray
        vector of the form [mRNA, miRNA, protein] contain the concentration of these species at time t

    parameters : ndarray
        vector of the form [ repression_threshold_protein_on_mRNA, repression_threshold_protein_on_miRNA,
                             repression_threshold_miRNA_on_mRNA, hill_coefficient_protein_on_mRNA,
                             hill_coefficient_miRNA_on_mRNA, hill_coefficient_protein_on_miRNA,
                             transcription_delay, upper_mRNA_degradation_rate, lower_mRNA_degradation_rate,
                             protein_degradation_rate, miRNA_degradation_rate, negative_times_indicator ]
        containing the value of these parameters.
        The value of negative_times_indicator corresponds to for_negative_times in generate_deterministic_trajectory().
        The value 0.0 corresponds to the option 'initial', whereas 1.0 corresponds to 'zero',
        and 2.0 corresponds to 'no_negative'.

    time : float
        time at which the gradient is calculated

    Returns
    -------

    gradient : ndarray
        vector of the form [dmRNA, dmiRNA, dProtein] containing the evaluated right hand side of the
        delay differential equation for the species concentrations provided in y, the given
        parameters, and at time t.
    '''
    basal_mRNA_transcription_rate = float(parameters[0])
    basal_miRNA_transcription_rate = float(parameters[1])
    translation_rate = float(parameters[2])
    repression_threshold_protein_on_mRNA = float(parameters[3])
    repression_threshold_protein_on_miRNA = float(parameters[4])
    repression_threshold_miRNA_on_mRNA = float(parameters[5])
    repression_threshold_miRNA_on_protein = float(parameters[6])
    hill_coefficient_protein_on_mRNA = float(parameters[7])
    hill_coefficient_protein_on_miRNA = float(parameters[8])
    hill_coefficient_miRNA_on_mRNA = float(parameters[9])
    hill_coefficient_miRNA_on_protein = float(parameters[10])
    transcription_delay = float(parameters[11])
    upper_mRNA_degradation_rate = float(parameters[12])
    lower_mRNA_degradation_rate = float(parameters[13])
    protein_degradation_rate = float(parameters[14])
    miRNA_degradation_rate = float(parameters[15])
    negative_times_indicator = float(parameters[16])

    if negative_times_indicator == 0.0:
        for_negative_times = 'initial'
    elif negative_times_indicator == 1.0:
        for_negative_times = 'zero'
    elif negative_times_indicator == 2.0:
        for_negative_times = 'no_negative'
    else:
        ValueError("Could not interpret the value of for_negative_times")

    mRNA = float(y[0])
    protein = float(y[1])
    miRNA = float(y[2])

    if (time>transcription_delay):
        past_protein = PyDDE.pastvalue(1,time-time_delay,0)
    elif time>0.0:
        if for_negative_times == 'initial':
            past_protein = PyDDE.pastvalue(1,0.0,0)
        elif for_negative_times == 'zero':
            past_protein = 0.0
    else:
        past_protein = protein

    ## rate of protein change
    translation_hill_function_value = 1.0/(1.0 + pow(miRNA/repression_threshold_miRNA_on_protein,
                                                     hill_coefficient_miRNA_on_protein))
    dprotein = translation_hill_function_value*mRNA - protein_degradation_rate*protein

    ## hill functions for miRNA and mRNA
    miRNA_production_hill_function_value = 1.0/(1.0+pow(past_protein/repression_threshold_protein_on_miRNA,
                                           hill_coefficient_protein_on_miRNA))

    transcription_hill_function_value = 1.0/(1.0+pow(past_protein/repression_threshold_protein_on_mRNA,
                                       hill_coefficient_protein_on_mRNA))

    effective_mRNA_degradation_rate = ( upper_mRNA_degradation_rate +
                                        ( lower_mRNA_degradation_rate - upper_mRNA_degradation_rate )/
                                        ( 1.0 + pow(miRNA/repression_threshold_miRNA_on_mRNA,
                                                   hill_coefficient_miRNA_on_mRNA) ) )

    if for_negative_times != 'no_negative': #in this case look up protein at negative times, which has been set above
        dmRNA = basal_mRNA_transcription_rate*translation_hill_function_value - effective_mRNA_degradation_rate*mRNA
        dmiRNA = basal_miRNA_transcription_rate - miRNA_degradation_rate*miRNA
    else:
        if time < time_delay:
            dmRNA = -effective_mRNA_degradation_rate*mRNA
            dmiRNA = -miRNA_degradation_rate*miRNA
        else:
            dmRNA = basal_transcription_rate*translation_hill_function_value - effective_mRNA_degradation_rate*mRNA
            dmiRNA = basal_miRNA_transcription_rate - miRNA_degradation_rate*miRNA

    return np.array( [dmRNA, dmiRNA, dprotein] )

def is_parameter_point_stochastically_oscillatory( repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29):
    '''Perform bifurcation analysis on the Linear Noise Approximation to test whether the given parameter combination falls
    within the regime where the stochastic solutions oscillate. A parameterpoint is considered oscillatory
    if there is a non-zero maximum in the power spectrum. The power spectrum has been derived by Tobias Galla in equation
    32 of the paper

    Galla, Phys. Rev. E 80 (2009)

    Parameters
    ----------

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    Returns
    -------

    is_oscillatory : bool
        True if in oscillatory regime, false if otherwise
    '''

    power_spectrum = calculate_theoretical_protein_power_spectrum_at_parameter_point(basal_transcription_rate = basal_transcription_rate,
                                                                             translation_rate = translation_rate,
                                                                             repression_threshold = repression_threshold,
                                                                             transcription_delay = transcription_delay,
                                                                             mRNA_degradation_rate = mRNA_degradation_rate,
                                                                             protein_degradation_rate = protein_degradation_rate,
                                                                             hill_coefficient = hill_coefficient)

    max_index = np.argmax(power_spectrum[:,1])

    if max_index > 0:
        is_oscillatory = True
    else:
        is_oscillatory = False

    return is_oscillatory

def is_parameter_point_deterministically_oscillatory( repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29):
    '''Perform bifurcation analysis of the deterministic DDE to test whether the given parameter combination falls
    within the regime where the DDE solutions oscillate. Conditions are for example derived in

    X.P. Wu, M. Eshete. Commun Nonlinear Sci Numer Simulat 16 (2011)

    and can be expressed as

    (1) mu_m*mu_p < alpha_m*alpha_p*abs(G'(p*))
    (2) define w = sqrt(1/2[sqrt( (mu_p^2-mu_m^2)^2+4(alph_m*alpha_p*G')^2 ) - mu_m^2 - mu_p^2])
        if w*tau > pi:
           then w*tau < 2pi - arccos((mu_m*mu_p -omega^2)/(alpha_m*alpha*p*G'(p*)) needs to be fulfilled,
           if it is not fulfilled then we cannot currently say whether the trace will oscillate or not
        else:
           then w*tau > arccos((mu_m*mu_p -omega^2)/(alpha_m*alpha*p*G'(p*)) needs to be fulfilled

    Here, the same notation as in Monk, Curr. Biol. (2003) was chosen.

    Parameters
    ----------

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    Returns
    -------

    is_oscillatory : bool
        True if in oscillatory regime, false if otherwise
    '''
    repression_threshold = float(repression_threshold)
    mean_mRNA, mean_protein = calculate_steady_state_of_ode( repression_threshold,
                                                             hill_coefficient,
                                                             mRNA_degradation_rate,
                                                             protein_degradation_rate,
                                                             basal_transcription_rate,
                                                             translation_rate )

    hill_denominator = 1+np.power(mean_protein/repression_threshold, hill_coefficient)
    abs_of_hill_derivative = ( hill_coefficient/(repression_threshold*hill_denominator*hill_denominator)*
                            np.power(mean_protein/repression_threshold, hill_coefficient - 1) )

    condition_one_fulfilled = ( protein_degradation_rate*mRNA_degradation_rate <
                                basal_transcription_rate*translation_rate*abs_of_hill_derivative)

    if not condition_one_fulfilled:
        is_oscillatory = False
    else:
        squared_degradation_difference = protein_degradation_rate*protein_degradation_rate - mRNA_degradation_rate*mRNA_degradation_rate
        squared_degradation_sum = protein_degradation_rate*protein_degradation_rate + mRNA_degradation_rate*mRNA_degradation_rate
        derivative_term = basal_transcription_rate*translation_rate*abs_of_hill_derivative

        omega = np.sqrt(0.5*(np.sqrt(squared_degradation_difference*squared_degradation_difference
                               + 4*derivative_term*derivative_term) -
                               squared_degradation_sum))
        arccos_value = np.arccos( ( omega*omega - protein_degradation_rate*mRNA_degradation_rate)/
                                    derivative_term )
        if omega*transcription_delay>np.pi:
            if omega*transcription_delay < ( 2*np.pi - arccos_value ):
                is_oscillatory = True
            else:
                print('Cannot determine if parameter point oscillates')
                print([basal_transcription_rate, translation_rate, repression_threshold,
                       transcription_delay, hill_coefficient, mRNA_degradation_rate, protein_degradation_rate])
                is_oscillatory = False
#                 raise ValueError("cannot determine if parameter point oscillates")
        else:
            if omega*transcription_delay > arccos_value:
                is_oscillatory = True
            else:
                is_oscillatory = False

    return is_oscillatory

def measure_period_and_amplitude_of_signal(x_values, signal_values):
    '''Measure the period of a signal generated with an ODE or DDE.
    This function will identify all peaks in the signal that are not at the boundary
    and return the average distance of consecutive peaks. Will also return the relative amplitude
    defined as the signal difference of each peak to the previous lowest dip relative to the mean
    of the signal; and the variation of that amplitude.

    Warning: This function will return nonsense on stochastic data

    Parameters
    ----------

    x_values : ndarray
        list of values, usually in time, at which the signal is measured

    signal_values : ndarray
        list of measured values

    Returns
    -------

    period : float
        period that was detected in the signal

    amplitude : float
        mean amplitude of the signal

    amplitude_variation : float
        standard variation of the signal amplitudes
    '''
    signal_mean = np.mean(signal_values)
    peak_indices = scipy.signal.argrelmax(signal_values)
    x_peaks = x_values[peak_indices]
    if len(x_peaks) < 2:
        return 0.0, 0.0, 0.0
    peak_distances = np.zeros( len( x_peaks ) - 1)
    peak_amplitudes = np.zeros( len( x_peaks ) - 1)
    for x_index, x_peak in enumerate( x_peaks[1:] ):
        previous_peak = x_peaks[ x_index ]
        interval_values = signal_values[peak_indices[0][x_index]:peak_indices[0][x_index + 1]]
        peak_value = signal_values[peak_indices[0][x_index + 1]] # see for loop indexing
        previous_dip_value = np.min(interval_values)
        amplitude_difference = np.abs(peak_value - previous_dip_value)/signal_mean
        peak_distances[ x_index ] = x_peak - previous_peak
        peak_amplitudes[x_index] = amplitude_difference

    return np.mean(peak_distances), np.mean(peak_amplitudes), np.std(peak_amplitudes)

@jit(nopython=True)
def generate_stochastic_trajectory( duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0,
#                                     explicit typing necessary for jit
                                    transcription_schedule = np.array([-1.0]),
                                    sampling_timestep = 1.0,
                                    vary_repression_threshold = False):
    '''Generate one trace of the Hes5 model. This function implements a stochastic version of
    the model model in Monk, Current Biology (2003). It applies the Gillespie-rejection method described
    in Cai et al, J. Chem. Phys. (2007) as Algorithm 2. This method is an exact method to calculate
    the temporal evolution of stochastic reaction systems with delay.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    transcription_schedule : ndarray
        1D numpy array of floats, corresponding to a list of pre-scheduled transcription events.
        This can be used to synchronise traces, as is done in generate_multiple_trajectories().
        This schedule will be ignored if its first entry is negative, as in the standard value.

    vary_repression_threshold : bool
        if true then the repression threshold will decrease after 1000 min

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''

    trace, remaining_transcription_times = generate_stochastic_trajectory_and_transcription_times( duration,
                                    repression_threshold,
                                    hill_coefficient,
                                    mRNA_degradation_rate,
                                    protein_degradation_rate,
                                    basal_transcription_rate,
                                    translation_rate,
                                    transcription_delay,
                                    initial_mRNA,
                                    initial_protein,
                                    equilibration_time,
                                    transcription_schedule,
                                    sampling_timestep,
                                    vary_repression_threshold)

    return trace

def calculate_approximate_protein_standard_deviation_at_parameter_point(basal_transcription_rate = 1.0,
                                                                translation_rate = 1.0,
                                                                repression_threshold = 100,
                                                                transcription_delay = 18.5,
                                                                mRNA_degradation_rate = 0.03,
                                                                protein_degradation_rate = 0.03,
                                                                hill_coefficient = 5
                                                                ):
    '''Approximate the standard deviation of the signal using linear noise approximation. The
    standard deviation in the linear noise approximation can be calculated using the integral of the
    power spectrum derived by Galla (2009).

    Parameters
    ----------

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    Returns
    -------

    standard_deviation : float
        theoretical standard deviaiton of the signal
    '''
    power_spectrum = calculate_theoretical_protein_power_spectrum_at_parameter_point(basal_transcription_rate = basal_transcription_rate,
                                                                             translation_rate = translation_rate,
                                                                             repression_threshold = repression_threshold,
                                                                             transcription_delay = transcription_delay,
                                                                             mRNA_degradation_rate = mRNA_degradation_rate,
                                                                             protein_degradation_rate = protein_degradation_rate,
                                                                             hill_coefficient = hill_coefficient,
                                                                             normalise = False)
    #use fourier-like frequency definition
    power_spectrum[:,0]*= 2*np.pi
    integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
    standard_deviation = np.sqrt(integral/np.pi)

    return standard_deviation

def calculate_theoretical_protein_power_spectrum_at_parameter_point(basal_transcription_rate = 1.0,
                                                            translation_rate = 1.0,
                                                            repression_threshold = 100,
                                                            transcription_delay = 18.5,
                                                            mRNA_degradation_rate = 0.03,
                                                            protein_degradation_rate = 0.03,
                                                            hill_coefficient = 5,
                                                            normalise = True
                                                            ):
    '''Calculate the theoretical power spectrum of the protein of the Monk (2003) model
    at a parameter point using equation 32 in Galla (2009), PRE.

    Parameters
    ----------

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    normalise : bool
        If True, normalise power spectrum to one.

    Returns
    -------

    power_spectrum : ndarray
        two coloumns, first column contains frequencies, second column contains power spectrum values
    '''
    actual_frequencies = np.linspace(0,0.01,1000)
    pi_frequencies = actual_frequencies*2*np.pi
    steady_state_mrna, steady_state_protein = calculate_steady_state_of_ode( repression_threshold = float(repression_threshold),
                                    hill_coefficient = hill_coefficient,
                                    mRNA_degradation_rate = mRNA_degradation_rate,
                                    protein_degradation_rate = protein_degradation_rate,
                                    basal_transcription_rate = basal_transcription_rate,
                                    translation_rate = translation_rate)

    steady_state_hill_function_value = 1.0/(1.0 + np.power( steady_state_protein/float(repression_threshold),
                                                            hill_coefficient ))

    steady_state_hill_derivative = -hill_coefficient*np.power(steady_state_protein/float(repression_threshold),
                                                            hill_coefficient - 1)/(repression_threshold*
                                    np.power(1.0+np.power(steady_state_protein/float(repression_threshold),
                                                        hill_coefficient),2))

#     steady_state_hill_derivative = -hill_coefficient/float(repression_threshold)*np.power(
#                                      1.0 + steady_state_protein/float(repression_threshold),
#                                                     hill_coefficient)

    power_spectrum_values = ( np.power(translation_rate,2)*
                       ( basal_transcription_rate * steady_state_hill_function_value +
                         mRNA_degradation_rate*steady_state_mrna)
                      +
                        (np.power(pi_frequencies,2) + np.power(mRNA_degradation_rate,2))*
                         ( translation_rate*steady_state_mrna + protein_degradation_rate*steady_state_protein)
                      )/(np.power(-np.power(pi_frequencies,2) +
                                  protein_degradation_rate*mRNA_degradation_rate
                                  - basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.cos(pi_frequencies*transcription_delay),2)
                         +
                         np.power((protein_degradation_rate+mRNA_degradation_rate)*
                                  pi_frequencies +
                                  basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.sin(pi_frequencies*transcription_delay), 2)
                         )

    power_spectrum = np.vstack((actual_frequencies, power_spectrum_values)).transpose()
    if normalise:
        integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        power_spectrum[:,1] /= integral

    return power_spectrum

### this part is standard deviation for the mRNA
def calculate_approximate_mRNA_standard_deviation_at_parameter_point(basal_transcription_rate = 1.0,
                                                                translation_rate = 1.0,
                                                                repression_threshold = 100,
                                                                transcription_delay = 18.5,
                                                                mRNA_degradation_rate = 0.03,
                                                                protein_degradation_rate = 0.03,
                                                                hill_coefficient = 5
                                                                ):
    '''Approximate the standard deviation of the signal using linear noise approximation. The
    standard deviation in the linear noise approximation can be calculated using the integral of the
    power spectrum derived by Galla (2009).

    Parameters
    ----------

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    Returns
    -------

    standard_deviation : float
        theoretical standard deviaiton of the signal
    '''
    power_spectrum = calculate_theoretical_mRNA_power_spectrum_at_parameter_point(basal_transcription_rate = basal_transcription_rate,
                                                                             translation_rate = translation_rate,
                                                                             repression_threshold = repression_threshold,
                                                                             transcription_delay = transcription_delay,
                                                                             mRNA_degradation_rate = mRNA_degradation_rate,
                                                                             protein_degradation_rate = protein_degradation_rate,
                                                                             hill_coefficient = hill_coefficient,
                                                                             normalise = False)
    #use fourier-like frequency definition
    power_spectrum[:,0]*= 2*np.pi
    integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
    standard_deviation = np.sqrt(integral/np.pi)

    return standard_deviation

def calculate_theoretical_mRNA_power_spectrum_at_parameter_point(basal_transcription_rate = 1.0,
                                                            translation_rate = 1.0,
                                                            repression_threshold = 100,
                                                            transcription_delay = 18.5,
                                                            mRNA_degradation_rate = 0.03,
                                                            protein_degradation_rate = 0.03,
                                                            hill_coefficient = 5,
                                                            normalise = True
                                                            ):
    '''Calculate the theoretical power spectrum of the mRNA of the Monk (2003) model
    at a parameter point using equation 31 in Galla (2009), PRE.

    Parameters
    ----------

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    normalise : bool
        If True, normalise power spectrum to one.

    Returns
    -------

    power_spectrum : ndarray
        two coloumns, first column contains frequencies, second column contains power spectrum values
    '''
    actual_frequencies = np.linspace(0,0.01,1000)
    pi_frequencies = actual_frequencies*2*np.pi
    steady_state_mrna, steady_state_protein = calculate_steady_state_of_ode( repression_threshold = float(repression_threshold),
                                    hill_coefficient = hill_coefficient,
                                    mRNA_degradation_rate = mRNA_degradation_rate,
                                    protein_degradation_rate = protein_degradation_rate,
                                    basal_transcription_rate = basal_transcription_rate,
                                    translation_rate = translation_rate)

    steady_state_hill_function_value = 1.0/(1.0 + np.power( steady_state_protein/float(repression_threshold),
                                                            hill_coefficient ))

    steady_state_hill_derivative = -hill_coefficient*np.power(steady_state_protein/float(repression_threshold),
                                                            hill_coefficient - 1)/(repression_threshold*
                                    np.power(1.0+np.power(steady_state_protein/float(repression_threshold),
                                                        hill_coefficient),2))

#     steady_state_hill_derivative = -hill_coefficient/float(repression_threshold)*np.power(
#                                      1.0 + steady_state_protein/float(repression_threshold),
#                                                     hill_coefficient)

    power_spectrum_values = ( (np.power(pi_frequencies,2)+np.power(protein_degradation_rate,2))*(
                               basal_transcription_rate*steady_state_hill_function_value + mRNA_degradation_rate*steady_state_mrna)
                               +
                               np.power(basal_transcription_rate*steady_state_hill_derivative,2)*(translation_rate*steady_state_mrna +
                               protein_degradation_rate*steady_state_protein)
                             )/(np.power(- np.power(pi_frequencies,2) +
                                  protein_degradation_rate*mRNA_degradation_rate
                                  - basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.cos(pi_frequencies*transcription_delay),2)
                         +
                         np.power((protein_degradation_rate+mRNA_degradation_rate)*
                                  pi_frequencies +
                                  basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.sin(pi_frequencies*transcription_delay), 2)
                         )

    power_spectrum = np.vstack((actual_frequencies, power_spectrum_values)).transpose()
    if normalise:
        integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        power_spectrum[:,1] /= integral

    return power_spectrum

@jit(nopython=True)
def generate_stochastic_trajectory_and_transcription_times( duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0,
                                    #explicit typing necessary for jit
                                    transcription_schedule = np.array([-1.0]),
                                    sampling_timestep = 1.0,
                                    vary_repression_threshold = False):
    '''Generate one trace of the Hes5 model. This function implements a stochastic version of
    the model model in Monk, Current Biology (2003). It applies the rejection method described
    in Cai et al, J. Chem. Phys. (2007) as Algorithm 2. This method is an exact method to calculate
    the temporal evolution of stochastic reaction systems with delay. At the end of the trajectory,
    transcription events will have been scheduled to occur after the trajectory has terminated.
    This function returns this transcription schedule, as well.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        trajectory in order to get rid of any overshoots, for example

    return_transcription_times : bool
        at the end of each simulation there are a set of already-scheduled but not yet executed
        transcription times that have been calculated. If this option is true, these
        transcription times will be returned as second return parameter.

    transcription_schedule : ndarray
        1D numpy array of floats, corresponding to a list of pre-scheduled transcription events.
        This can be used to synchronise traces, as is done in generate_multiple_trajectories()
        This schedule will be ignored if its first entry is negative, as for the standard value.

    vary_repression_threshold : bool
        if true then the repression threshold will decrease after 1000 min
    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number

    remaining_transcription_times : list
        list entries are floats. Each entry corresponds to the time of one transcription event.
    '''
    total_time = duration + equilibration_time
    sample_times = np.arange(equilibration_time, total_time, sampling_timestep)
    trace = np.zeros((len(sample_times), 3))
    trace[:,0] = sample_times

    repression_threshold = float(repression_threshold)
    # inital_condition
    current_mRNA = initial_mRNA
    current_protein =  initial_protein
#     trace[0,1] = initial_mRNA
#     trace[0,2] = initial_protein
    propensities = np.array([ basal_transcription_rate/(1.0+ np.power(current_protein/repression_threshold,
                                                                      hill_coefficient)), # transcription
                              initial_mRNA*translation_rate, # translation
                              initial_protein*protein_degradation_rate, # Protein degradation
                              initial_mRNA*mRNA_degradation_rate ] ) # mRNA degradation

    # set up the gillespie algorithm: We first
    # need a list where we store any delayed reaction times
#     delayed_transcription_times = collections.deque()
    # this requires a bit of faff for jit to compile
    delayed_transcription_times = [-1.0]
    if transcription_schedule[0] < 0:
        delayed_transcription_times.pop(0)
        #this is now an empty list
    else:
#         delayed_transcription_times += [time for time in transcription_schedule]
        delayed_transcription_times += [time for time in transcription_schedule]
        delayed_transcription_times.pop(0)
        # this is now a list containing all the transcription times passed to this function

    # This following index is to keep track at which index of the trace entries
    # we currently are (see definition of trace above). This is necessary since
    # the SSA will calculate reactions at random times and we need to transform
    # calculated reaction times to the sampling time points
    sampling_index = 0

    time = 0.0
    while time < sample_times[-1]:
        base_propensity = propensities[0] + propensities[1] + propensities[2] + propensities[3]
        # two random numbers for Gillespie algorithm
        first_random_number, second_random_number = np.random.rand(2)
        # time to next reaction
        time_to_next_reaction = -1.0/base_propensity*np.log(first_random_number)
        if ( len(delayed_transcription_times)> 0 and
             delayed_transcription_times[0] < time + time_to_next_reaction): # delayed transcription execution
            current_mRNA += 1
#             time = delayed_transcription_times.popleft()
            time = delayed_transcription_times.pop(0)
            propensities[1] = current_mRNA*translation_rate
            propensities[3] = current_mRNA*mRNA_degradation_rate
        else:
            time += time_to_next_reaction
            # identify which of the four reactions occured
            reaction_index = identify_reaction(second_random_number, base_propensity, propensities)
            # execute reaction
            if reaction_index == 0:  #transcription initiation
                delayed_transcription_times.append(time + time_to_next_reaction + transcription_delay)
                # propensities don't change
            elif reaction_index == 1: #protein translation
                current_protein += 1
                if vary_repression_threshold and time > equilibration_time + 2000.0:
                    current_repression_threshold = repression_threshold*(1.0-(time-equilibration_time-2000.0)/1000.0*0.7)
#                     current_repression_threshold = repression_threshold*(0.5)
#                     current_repression_threshold = max(current_repression_threshold, 0.01)
                else:
                    current_repression_threshold = repression_threshold
                propensities[0] = basal_transcription_rate/(1.0+
                                                            np.power(current_protein/current_repression_threshold,
                                                                     hill_coefficient))
                propensities[2] = current_protein*protein_degradation_rate
            elif reaction_index == 2: #protein degradation
                current_protein -=1
                if vary_repression_threshold and time > equilibration_time + 2000.0:
                    current_repression_threshold = repression_threshold*(1.0-(time-equilibration_time-2000.0)/1000.0*0.7)
#                     current_repression_threshold = repression_threshold*(0.5)
#                     current_repression_threshold = repression_threshold*(1.0-(time-equilibration_time-2000.0)/2500.0)
                    current_repression_threshold = max(current_repression_threshold, 0.01)
                else:
                    current_repression_threshold = repression_threshold
                propensities[0] = basal_transcription_rate/(1.0+
                                                            np.power(current_protein/current_repression_threshold,
                                                                     hill_coefficient))
                propensities[2] = current_protein*protein_degradation_rate
            elif reaction_index == 3: #mRNA degradation
                current_mRNA -= 1
                propensities[1] = current_mRNA*translation_rate
                propensities[3] = current_mRNA*mRNA_degradation_rate
            else:
                raise(RuntimeError("Couldn't identify reaction. This should not happen."))

        # update trace for all entries until the current time
        while ( sampling_index < len(sample_times) and
                time > trace[ sampling_index, 0 ] ):
            trace[ sampling_index, 1 ] = current_mRNA
            trace[ sampling_index, 2 ] = current_protein
            sampling_index += 1

    trace[:,0] -= equilibration_time

#     if return_transcription_times:
#        return trace, delayed_transcription_times
#     else:
    return trace, delayed_transcription_times

def generate_multiple_trajectories( number_of_trajectories = 10,
                                    duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0,
                                    synchronize = False,
                                    number_of_cpus = number_of_available_cores,
                                    sampling_timestep = 1.0):
    '''Generate multiple stochastic traces the Hes5 model by using
       generate_stochastic_trajectory.

    Parameters
    ----------

    number_of_trajectories : int

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    synchronize : bool
        if True, only one trajectory will be run for the equilibration period, and all recorded traces
        will start along this trajectory.

    number_of_cpus : int
        number of cpus that should be used to calculate the traces. Traces will be calculated in paralell on
        this number of cpus.

    Returns
    -------

    mRNA_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of mRNA copy numbers

    protein_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of protein copy numbers
    '''

    if synchronize:
        equilibrated_trace, transcription_times = generate_stochastic_trajectory_and_transcription_times(
                                                     equilibration_time, #duration
                                                     repression_threshold,
                                                     hill_coefficient,
                                                     mRNA_degradation_rate,
                                                     protein_degradation_rate,
                                                     basal_transcription_rate,
                                                     translation_rate,
                                                     transcription_delay,
                                                     initial_mRNA,
                                                     initial_protein,
                                                     equilibration_time = 0.0,
                                                     )
        initial_mRNA = equilibrated_trace[-1,1]
        initial_protein = equilibrated_trace[-1,2]
        transcription_schedule = np.array([transcription_time for transcription_time in
                                           transcription_times])
        transcription_schedule -= equilibration_time
        equilibration_time = 0.0
    else:
        transcription_schedule = np.array([-1.0])

    ## only spawn subprocesses if we have more than one core
    if number_of_cpus > 1:
        pool_of_processes = mp.Pool(processes = number_of_cpus)
        arguments = [ (duration, repression_threshold, hill_coefficient,
                      mRNA_degradation_rate, protein_degradation_rate,
                      basal_transcription_rate, translation_rate,
                      transcription_delay, initial_mRNA, initial_protein,
                      equilibration_time, transcription_schedule, sampling_timestep, False) ]*number_of_trajectories
#                       equilibration_time, transcription_schedule) ]*number_of_trajectories
        process_results = [ pool_of_processes.apply_async(generate_stochastic_trajectory, args=x)
                            for x in arguments ]

        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        list_of_traces = []
        for result in process_results:
            this_trace = result.get()
            list_of_traces.append(this_trace)

        first_trace = list_of_traces[0]

        sample_times = first_trace[:,0]
        mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
        protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time

        mRNA_trajectories[:,0] = sample_times
        protein_trajectories[:,0] = sample_times

        for trajectory_index, this_trace in enumerate(list_of_traces):
            # offset one index for time column
            mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1]
            protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]

    elif number_of_cpus == 1:
        first_trace = generate_stochastic_trajectory(duration,
                                                     repression_threshold,
                                                     hill_coefficient,
                                                     mRNA_degradation_rate,
                                                     protein_degradation_rate,
                                                     basal_transcription_rate,
                                                     translation_rate,
                                                     transcription_delay,
                                                     initial_mRNA,
                                                     initial_protein,
                                                     equilibration_time,
                                                     transcription_schedule,
                                                     sampling_timestep,
                                                     False)

        sample_times = first_trace[:,0]
        mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
        protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time

        mRNA_trajectories[:,0] = sample_times
        protein_trajectories[:,0] = sample_times
        mRNA_trajectories[:,1] = first_trace[:,1]
        protein_trajectories[:,1] = first_trace[:,2]

        for trajectory_index in range(1,number_of_trajectories):
        # offset one index for time column
            this_trace = generate_stochastic_trajectory(duration,
                                                        repression_threshold,
                                                        hill_coefficient,
                                                        mRNA_degradation_rate,
                                                        protein_degradation_rate,
                                                        basal_transcription_rate,
                                                        translation_rate,
                                                        transcription_delay,
                                                        initial_mRNA,
                                                        initial_protein,
                                                        equilibration_time,
                                                        transcription_schedule,
                                                        sampling_timestep,
                                                        False)

            mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1]
            protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]

    return mRNA_trajectories, protein_trajectories

@jit(nopython = True)
def identify_reaction(random_number, base_propensity, propensities):
    '''Choose a reaction from a set of possiblities using a random number and the corresponding
    reaction propensities. To be used, for example, in a Gillespie SSA.

    This function will find j such that

    sum_0^(j-1) propensities[j] < random_number*sum(propensities) < sum_0^(j) propensities[j]

    Parameters
    ----------

    random_number : float
        needs to be between 0 and 1

    base_propensity : float
        the sum of all propensities in the propensities argument. This is a function argument
        to avoid repeated calculation of this value throughout the algorithm, which improves
        performance

    propensities : ndarray
        one-dimensional array of arbitrary length. Each entry needs to be larger than zero.

    Returns
    -------

    reaction_index : int
        The reaction index
    '''
    scaled_random_number = random_number*base_propensity
    propensity_sum = 0.0
    for reaction_index, propensity in enumerate(propensities):
        if scaled_random_number < propensity_sum + propensity:
            return reaction_index
        else:
            propensity_sum += propensity

    ##Make sure we never exit the for loop:
    raise(RuntimeError("This line should never be reached."))

def get_period_measurements_from_signal(time_points, signal, smoothen = False):
    '''Use hilbert transforms and the analytical signal to get single period measurements
    from the given time series. Period is extracted from the time-points of phase-reset.

    Parameters:
    -----------

    time_points : ndarray
        1-dimensional array, contains time-point values at which the signal is measured.
        The entries in the array are assumed to be equidistant.

    signal : ndarray
        1-dimensional array, contains signal values at time_points

    smoothen : bool
        if True then a smoothening savitzki-goyal filter with a length of 75 minutes will be applied.
        For this filter we assume that time is sampled once per minute.

    Returns:
    --------

    period_values : ndarray
        contains all periods measured in the signal
    '''
    if smoothen:
        signal_for_measurement = scipy.signal.savgol_filter(signal,
                                                            75,
                                                            3)
    else:
        signal_for_measurement = signal
    analytic_signal = scipy.signal.hilbert(signal_for_measurement - np.mean(signal_for_measurement))
    phase = np.angle(analytic_signal)
    #this will find the index just before zero-crossings from plus to minus
    phase_reset_indices = np.where(np.diff(np.signbit(phase).astype(int))>0)
#     phase_reset_times = time_points[phase_reset_indices][1:]
    phase_reset_times = time_points[phase_reset_indices]
    period_values = np.diff(phase_reset_times)

    return period_values

def calculate_power_spectrum_of_trajectories(trajectories, method = 'standard',
                                             normalize = True,
                                             power_spectrum_smoothing_window = 0.001):
    '''Calculate the power spectrum, coherence, and period of a set
    of trajectories. Works by applying discrete fourier transforms to the mean
    of the trajectories. We define the power spectrum as the square of the
    absolute of the fourier transform, and we define the coherence as in
    Phillips et al (eLife, 2016): the relative area of the power spectrum
    occopied by a 20% frequency band around the maximum frequency.
    The maximum frequency corresponds to the inverse of the period.
    The returned power spectum excludes the frequency 0 and thus neglects
    the mean of the signal. The power spectrum is normalised such that
    all entries add to one.

    Warning: a correction factor may be needed if Deltat ne 1

    Parameters:
    ----------

    trajectories : ndarray
        2D array. First column is time, each further column contains one realisation
        of the signal that is aimed to be analysed.

    method : string
        The options 'standard' and 'mean' are possible. If 'standard', then
        the fft will be calculated on each trajectory, and the average of all
        power spectra will be calculated. If 'mean', then the power spectrum
        of the mean will be calculated.

    normalize : bool
        If True the power spectrum will be normalised. Otherwise it won't.
        Only applies if method is 'standard'. Will be ignored otherwise.
        The zero-frequency entry will be removed from the power spectrum if normalize is true.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is temporarilly smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Result:
    ------

    power_spectrum : ndarray
        first column contains frequencies, second column contains the power spectrum
        |F(x)|^2, where F denotes the Fourier transform and x is the mean signal
        extracted from the argument `trajectories'.

    coherence : float
        coherence value for this trajectory, as defined by Phillips et al

    period : float
        period corresponding to the maximum observed frequency
    '''

    if method == 'mean':
        mean_trajectory = np.vstack((trajectories[:,0], np.mean(trajectories[:,1:], axis = 1))).transpose()
        power_spectrum, coherence, period = calculate_power_spectrum_of_trajectory(mean_trajectory)
    elif method == 'standard':
        times = trajectories[:,0]
        #calculate the first power spectrum separately to get the extracted frequency values
        first_compound_trajectory = np.vstack((times, trajectories[:,1])).transpose()
        first_power_spectrum,_,_ = calculate_power_spectrum_of_trajectory(first_compound_trajectory,
                                                                          normalize = False)
        frequency_values = first_power_spectrum[:,0]
        all_power_spectra = np.zeros((first_power_spectrum.shape[0], trajectories.shape[1] - 1))
        all_power_spectra[:,0] = first_power_spectrum[:,1]
        trajectory_index = 1
        for trajectory in trajectories[:,2:].transpose():
            this_compound_trajectory = np.vstack((times, trajectory)).transpose()
            this_power_spectrum,_,_ = calculate_power_spectrum_of_trajectory(this_compound_trajectory,
                                                                             normalize = False)
            all_power_spectra[:,trajectory_index] = this_power_spectrum[:,1]
            trajectory_index += 1
        mean_power_spectrum_without_frequencies = np.mean(all_power_spectra, axis = 1)
        power_spectrum = np.vstack((frequency_values, mean_power_spectrum_without_frequencies)).transpose()

        power_integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        normalized_power_spectrum = power_spectrum.copy()
        if power_integral > 0.0:
            normalized_power_spectrum[:,1] = power_spectrum[:,1]/power_integral
        smoothened_power_spectrum = smoothen_power_spectrum(power_spectrum, power_spectrum_smoothing_window)
        coherence, period = calculate_coherence_and_period_of_power_spectrum(smoothened_power_spectrum)
        if normalize:
            power_spectrum = normalized_power_spectrum
    else:
        raise ValueError("This method of period extraction could not be resolved. Only the options 'mean' and 'standard' are accepted.")

    return power_spectrum, coherence, period

def calculate_power_spectrum_of_trajectory(trajectory, normalize = True):
    '''Calculate the power spectrum, coherence, and period, of a trajectory.
    Works by applying discrete fourier transformation. We define the power spectrum as the square of the
    absolute of the fourier transform, and we define the coherence as in
    Phillips et al (eLife, 2016): the relative area of the power spectrum
    occopied by a 20% frequency band around the maximum frequency.
    The maximum frequency corresponds to the inverse of the period.
    The returned power spectum excludes the frequency 0 and thus neglects
    the mean of the signal. The power spectrum is normalised so that all entries add up to one,
    unless normalized = False is specified.

    Parameters:
    ----------

    trajectories : ndarray
        2D array. First column is time, second column contains the signal that is aimed to be analysed.

    normalize : bool
        If True then the power spectrum is normalised such that all entries add to one. Otherwise
        no normalization is performed.

    Result:
    ------

    power_spectrum : ndarray
        first column contains frequencies, second column contains the power spectrum
        |F(x)|^2, where F denotes the Fourier transform and x is the mean signal
        extracted from the argument `trajectories'. The spectrum is normalised
        to add up to one.

    coherence : float
        coherence value for this trajectory, as defined by Phillips et al

    period : float
        period corresponding to the maximum observed frequency
    '''
    # Calculate power spectrum
    number_of_data_points = len(trajectory)
    interval_length = trajectory[-1,0]
    trajectory_to_transform = trajectory[:,1] - np.mean(trajectory[:,1])
    fourier_transform = np.fft.fft(trajectory_to_transform, norm = 'ortho')
#     fourier_transform = np.fft.fft(trajectory_to_transform)
    fourier_frequencies = np.arange( 0,number_of_data_points/(2*interval_length),
                                                     1.0/(interval_length) )
    power_spectrum_without_frequencies = np.power(np.abs(fourier_transform[:(number_of_data_points//2)]),2)

    # this should really be a decision about the even/oddness of number of datapoints
    try:
        power_spectrum = np.vstack((fourier_frequencies, power_spectrum_without_frequencies)).transpose()
    except ValueError:
        power_spectrum = np.vstack((fourier_frequencies[:-1], power_spectrum_without_frequencies)).transpose()

    power_integral = np.trapz(power_spectrum[1:,1], power_spectrum[1:,0])
    normalized_power_spectrum = power_spectrum[1:].copy()
    if np.sum(normalized_power_spectrum[1:,1])!=0.0:
        normalized_power_spectrum[:,1] = power_spectrum[1:,1]/power_integral
    coherence, period = calculate_coherence_and_period_of_power_spectrum(normalized_power_spectrum)
    if normalize:
        power_spectrum = normalized_power_spectrum

    return power_spectrum, coherence, period

def generate_lookup_tables_for_abc( total_number_of_samples,
                                number_of_traces_per_sample = 10,
                                number_of_cpus = number_of_available_cores,
                                saving_name = 'sampling_results',
                                prior_bounds = {'basal_transcription_rate' : (0,100),
                                                'translation_rate' : (0,200),
                                                'repression_threshold' : (0,100000),
                                                'time_delay' : (5,40),
                                                'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
                                                'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)},
                                prior_dimension = 'hill',
                                model = 'langevin',
                                logarithmic = True,
                                simulation_timestep = 0.5,
                                simulation_duration = 1500,
                                power_spectrum_smoothing_window = 0.001):
    '''Generate a prior distribution of parameter combinations. For each parameter combination, calculate summary statistics.
    The table containing all parameter combinations is saved under test/output/[saving_name]_parameters.npy. The table containing all summary statistics
    is saved as test/output/[saving_name].npy. The order in both tables is the same, i.e. the summary statistics in table row 'n' correspond
    to the parameter combination in table row 'n'. For information on the order of parameters in one combination, see 'generate_prior_samples'.
    For the order of summary statistics, see 'calculate_langevin_summary_statistics_at_parameters' (This only applies if the model 'langevin' is selected).

    Parameters
    ----------

    number_of_samples : int
        number of samples to be generated from the posterior

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    saving_name : string
        base of the filename, without file ending. The files will be saved in the 'output' subfolder of the
        project's test folder. The saved results contain a results file with all the summary statistics, and one
        with the corresponding parameter choices.

    prior_bounds : dict
        python dictionary containing parameter names and the bounds of the respective uniform prior.

    prior_dimension : string
        'reduced', 'hill' or 'full', or 'agnostic', 'extrinsic_noise',
        'transcription_noise_amplification' are possible options. If 'full', then the mRNA and protein degradation rates
        will be inferred in addition to other model parameters. If 'hill',
        then all parameters exclucing the mRNA and protein degradation rates will be inferred.
        If the model is 'agnostic', then this option will be ignored.

    model : string
        options are 'langevin', 'gillespie', 'agnostic'

    logarithmic : bool
        if True then logarithmic priors will be used on the translation, transcription and extrinsic noise rate constants

    simulation_timestep : float
        The discretisation time step of the simulation. Only applies if the model is 'langevin' or 'agnostic'.

    simulation_duration : float
        The duration of the simulated time window for each trace. Only applies if the model is 'langevin' or 'agnostic'.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min. Only applies if the model is 'langevin' or 'agnostic'.

    Returns
    -------

    prior_samples : ndarray
        samples from the prior distribution. Each line contains a different parameter combination. The order of parameters in each line
        is described in generate_prior_samples().

    model_results : ndarray
        summary statistics at each parameter combination in prior_samples, in the same order. The order of summary statistics in each line is
        described in calculate_langevin_summary_statistics_at_parameters().
    '''
    if model == 'langevin' or model == 'agnostic':
        use_langevin = True

    if model == 'agnostic':
        prior_dimension = 'agnostic'

    # first: keep degradation rates infer translation, transcription, repression threshold,
    # and time delay
    prior_samples = generate_prior_samples( total_number_of_samples, use_langevin,
                                            prior_bounds, prior_dimension, logarithmic )

    # collect summary_statistics_at_parameters
    model_results = calculate_summary_statistics_at_parameters( prior_samples,
                                                                number_of_traces_per_sample,
                                                                number_of_cpus,
                                                                model,
                                                                simulation_timestep,
                                                                simulation_duration,
                                                                power_spectrum_smoothing_window )

    saving_path = os.path.join(os.path.dirname(__file__),'..','test','output',saving_name)

    np.save(saving_path + '.npy', model_results)
    np.save(saving_path + '_parameters.npy', prior_samples)

    return prior_samples, model_results

def plot_posterior_distributions( posterior_samples, logarithmic = True ):
    '''Plot the posterior samples in a pair plot. Only works if there are
    more than four samples present

    Parameters
    ----------

    posterior samples : np.array
        The samples from which the pairplot should be generated.
        Each row contains a parameter

    logarithmic : bool
        if bool then the transcription and translation rate axes have logarithmic scales

    Returns
    -------

    paiplot : matplotlib figure handle
       The handle for the pairplot on which the use can call 'save'
    '''
    sns.set()

#     posterior_samples[:,2]/=10000

    if posterior_samples.shape[1] == 4:
        data_frame = pd.DataFrame( data = posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay'])
    elif posterior_samples.shape[1] == 5:
        data_frame = pd.DataFrame( data = posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay',
                                             'Hill coefficient'])
    elif posterior_samples.shape[1] == 6:
        data_frame = pd.DataFrame( data = posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'Noise_strength'])
    elif posterior_samples.shape[1] == 7:
        data_frame = pd.DataFrame( data = posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation',
                                             'Protein degradation'])
    else:
        raise ValueError("Cannot plot posterior samples of this dimension.")

    pairplot = sns.PairGrid(data_frame)
    pairplot.map_diag(sns.distplot, kde = False, rug = False )
    pairplot.map_offdiag(sns.regplot, scatter_kws = {'alpha' : 0.4, 'rasterized' : True}, fit_reg=False)
    pairplot.set(xlim = (0,None), ylim = (0,None))
    if logarithmic:
        for artist in pairplot.diag_axes[0].get_children():
            try:
                artist.remove()
            except:
                pass
        for artist in pairplot.diag_axes[1].get_children():
            try:
                artist.remove()
            except:
                pass
        transcription_rate_bins = np.logspace(-1,2,20)
        translation_rate_bins = np.logspace(0,2.3,20)
        plt.sca(pairplot.diag_axes[0])
        transcription_histogram,_ = np.histogram( data_frame['Transcription rate'],
                                                  bins = transcription_rate_bins )
        sns.distplot(data_frame['Transcription rate'],
                     kde = False,
                     rug = False,
                     bins = transcription_rate_bins)
    #                  ax = pairplot.diag_axes[0])
#         pairplot.diag_axes[0].set_ylim(0,np.max(transcription_histogram)*1.2)
        plt.gca().set_xlim(0.5,100)

        plt.sca(pairplot.diag_axes[1])
        sns.distplot(data_frame['Translation rate'],
                     kde = False,
                     rug = False,
                     bins = translation_rate_bins)
        plt.gca().set_xlim(1,200)
    #
        pairplot.axes[-1,0].set_xscale("log")
        pairplot.axes[-1,0].set_xlim(0.1,100)
        pairplot.axes[-1,1].set_xscale("log")
        pairplot.axes[-1,1].set_xlim(1,200)
        pairplot.axes[0,0].set_yscale("log")
        pairplot.axes[0,0].set_ylim(0.1,100)
        pairplot.axes[1,0].set_yscale("log")
        pairplot.axes[1,0].set_ylim(1,200)

    for artist in pairplot.diag_axes[3].get_children():
            try:
                artist.remove()
            except:
                pass
    plt.sca(pairplot.diag_axes[3])
    time_delay_bins = np.linspace(5,40,10)
    time_delay_histogram,_ = np.histogram( data_frame['Transcription delay'],
                                                  bins = time_delay_bins )
    sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                     bins = time_delay_bins)
    #                  ax = pairplot.diag_axes[0])
#         pairplot.diag_axes[0].set_ylim(0,np.max(transcription_histogram)*1.2)
    plt.gca().set_xlim(5,40)

    pairplot.axes[-1,2].set_xlim(0,10)
    pairplot.axes[-1,3].set_xlim(5,40)

#     if posterior_samples.shape[1] == 6:
#         pairplot.axes[-1,4].locator_params(axis = 'x', nbins = 5)
#         pairplot.axes[-1,5].locator_params(axis = 'x', nbins = 5)
#         pairplot.axes[-1,4].set_xlim(0,0.04)
#         pairplot.axes[-1,5].set_xlim(0,0.04)

#     pairplot = sns.PairGrid(data_frame)
#     pairplot.map_diag(sns.kdeplot)
#     pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)

    sns.reset_orig()

    return pairplot


def plot_posterior_distributions_MiVe(data_frame, prior_bounds, logarithmic=True):
    '''Plot the posterior samples in a pair plot. Only works if there are
    more than four samples present
    **MiVe: Generalized function to plot performance DataFrame object in the same manner
    **MiVe: Extended parameter ranges and changed upper triangle of pair grid
    scatter plots with contour plots
    Parameters
    ----------

    data_frame : pd.DataFrame
        The samples from which the pairplot should be generated.
        Each row contains a parameter
    prior_bounds : pd.DataFrame
        The bounds that were used to generate the prior samples
        Each parameter has a lower bound on row 0 and an upper bound on row 1
    logarithmic : bool
        if bool then the transcription and translation rate axes have logarithmic scales

    Returns
    -------

    paiplot : matplotlib figure handle
       The handle for the pairplot on which the use can call 'save'
    '''
    sns.set()

    pairplot = sns.PairGrid(data_frame)
    pairplot.map_diag(sns.distplot, kde=False, rug=False)
    pairplot.map_lower(sns.regplot, scatter_kws={'alpha': 0.4, 'rasterized': True}, fit_reg=False)
    pairplot.map_upper(sns.kdeplot)
    pairplot.set(xlim=(0, None), ylim=(0, None))
    if logarithmic:
        for artist in pairplot.diag_axes[0].get_children():
            try:
                artist.remove()
            except:
                pass
        for artist in pairplot.diag_axes[1].get_children():
            try:
                artist.remove()
            except:
                pass
        transcription_rate_bins = np.logspace(np.log10(prior_bounds.loc[0,'Transcription rate [m/min]']),
                                              np.log10(prior_bounds.loc[1,'Transcription rate [m/min]']), 20)
        translation_rate_bins = np.logspace(np.log10(prior_bounds.loc[0,'Translation rate [m/min]']),
                                              np.log10(prior_bounds.loc[1,'Translation rate [m/min]']), 20)
        plt.sca(pairplot.diag_axes[0]) #Current figure updated to the parent of diag_axes[0] => to pairplot?
        transcription_histogram, _ = np.histogram(data_frame['Transcription rate [m/min]'],
                                                  bins=transcription_rate_bins)
        sns.distplot(data_frame['Transcription rate [m/min]'],
                     kde=False,
                     rug=False,
                     bins=transcription_rate_bins)
        #                  ax = pairplot.diag_axes[0])
        #         pairplot.diag_axes[0].set_ylim(0,np.max(transcription_histogram)*1.2)
        plt.gca().set_xlim(0.5, 100)

        plt.sca(pairplot.diag_axes[1])
        sns.distplot(data_frame['Translation rate [m/min]'],
                     kde=False,
                     rug=False,
                     bins=translation_rate_bins)
        plt.gca().set_xlim(1, 200)
        #
        for i in range(0,5):
            pairplot.axes[-i,0].set_xscale("log");
            pairplot.axes[-i,1].set_xscale("log");
            pairplot.axes[0,i].set_yscale("log");
            pairplot.axes[1,i].set_yscale("log");
            pairplot.axes[-1,i].set_xticklabels(pairplot.axes[-1,i].get_xticks(),rotation=30);
        pairplot.axes[-1, -1].set_xticklabels(np.arange(10,70,25), rotation=30);
        #pairplot.axes[i,0].set_yticklabels(pairplot.axes[i,0].get_ylabel(),rotation=45);
        # pairplot.axes[-1, 0].set_xscale("log")
        # #pairplot.axes[-1, 0].set_xlim(0.1, 100)
        # pairplot.axes[-1, 1].set_xscale("log")
        # #pairplot.axes[-1, 1].set_xlim(1, 200)
        # pairplot.axes[0, 0].set_yscale("log")
        # #pairplot.axes[0, 0].set_ylim(0.1, 100)
        # pairplot.axes[1, 0].set_yscale("log")
        # #pairplot.axes[1, 0].set_ylim(1, 200)
    if prior_bounds.columns[2] == 'Transcription delay':
        for artist in pairplot.diag_axes[3].get_children():
            try:
                artist.remove()
            except:
                pass
        plt.sca(pairplot.diag_axes[3])
        time_delay_bins = np.linspace(5, 40, 10)
        time_delay_histogram, _ = np.histogram(data_frame['Transcription delay'],
                                               bins=time_delay_bins)
        sns.distplot(data_frame['Transcription delay'],
                     kde=False,
                     rug=False,
                     bins=time_delay_bins)
        #                  ax = pairplot.diag_axes[0])
        #         pairplot.diag_axes[0].set_ylim(0,np.max(transcription_histogram)*1.2)
        plt.gca().set_xlim(5, 40)

    # pairplot.axes[-1, 2].set_xlim(0, 10)
    # pairplot.axes[-1, 3].set_xlim(5, 40)

    #Set all ranges according to prior bounds
    lowermargin = 0.9
    uppermargin = 1.1
    lowermargin2 = 0.95
    uppermargin2 = 1.5
    for i in range(0,data_frame.shape[1]):
        if pairplot.axes[-1,i].get_xscale == 'log':
               pairplot.axes[-1,i].set_xlim(lowermargin2*np.log10(prior_bounds.loc[0,pairplot.axes[-1,i].get_xlabel()]),
                                            uppermargin2*np.log10(prior_bounds.loc[1,pairplot.axes[-1,i].get_xlabel()]))
        else:
            pairplot.axes[-1,i].set_xlim(lowermargin*prior_bounds.loc[0,pairplot.axes[-1,i].get_xlabel()],
                                         uppermargin*prior_bounds.loc[1,pairplot.axes[-1,i].get_xlabel()])
        if pairplot.axes[i,0].get_yscale == 'log':
            pairplot.axes[i,0].set_ylim(lowermargin2*np.log10(prior_bounds.loc[0,pairplot.axes[i,0].get_ylabel()]),
                                        uppermargin2*np.log10(prior_bounds.loc[1,pairplot.axes[i,0].get_ylabel()]))
        else:
            pairplot.axes[i,0].set_ylim(lowermargin*prior_bounds.loc[0,pairplot.axes[i,0].get_ylabel()],
                                        uppermargin*prior_bounds.loc[1,pairplot.axes[i,0].get_ylabel()])

    #     if posterior_samples.shape[1] == 6:
    #         pairplot.axes[-1,4].locator_params(axis = 'x', nbins = 5)
    #         pairplot.axes[-1,5].locator_params(axis = 'x', nbins = 5)
    #         pairplot.axes[-1,4].set_xlim(0,0.04)
    #         pairplot.axes[-1,5].set_xlim(0,0.04)

    #     pairplot = sns.PairGrid(data_frame)
    #     pairplot.map_diag(sns.kdeplot)
    #     pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)

    sns.reset_orig()

    return pairplot


def select_posterior_samples(prior_samples, distance_table, acceptance_ratio):
    '''Collect the parameter values of all prior_samples whose distance_table entries are
    within the acceptance_ratio closest samples.

    Note: this function is obsolute and is not being used to generate paper figures
    sample selection is performed in the test files instead.

    Parameters
    ----------

    prior_samples : ndarray
        2D, each row contains one set of parameters

    distance_table : ndarray
        1D, each entry contains one distance

    acceptance_ratio : float
        ratio of values to be accepted

    Returns
    -------

    posterior_samples : ndarray
        the rows of prior_samples corresponding to the acceptance_ratio smallest entries
        of distance_table
    '''
    # sort distances in ascending order
    distance_table_sorted_indices = distance_table.argsort()
    # specify how many samples to keep
    number_of_accepted_samples = int(round(acceptance_ratio*len(distance_table)))
    # select these samples
    posterior_samples = prior_samples[distance_table_sorted_indices[:number_of_accepted_samples]]

    return posterior_samples

def calculate_distances_to_data(summary_statistics):
    '''Calculate the distance of the observed summary statistics to our real data.
    Only takes the mean and variance into account.

    Parameters:
    -----------

    summary_statistics : ndarray
        first column contains means, second column contains relative standard deviations

    Results
    -------

    distances : ndarray
        single column with distances to the datapoint (62000, 0.15)
    '''
    summary_statistics = summary_statistics[:,:2]
    rescaling_values = np.std(summary_statistics, axis = 0)
    summary_statistics /= rescaling_values

    reference_summary_statistic = np.array([62000,0.15])/rescaling_values

    return np.linalg.norm(summary_statistics - reference_summary_statistic, axis = 1)

def calculate_heterozygous_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 10,
                                                            number_of_cpus = number_of_available_cores):
    '''Calculate the mean, relative standard deviation, period, and coherence
    of protein traces at each parameter point in parameter_values.
    Will assume the arguments to be of the order described in
    generate_prior_samples.
    If Gillespie samples are used the coherence and period measures may be inaccurate.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    Returns
    -------

    summary_statistics : ndarray
        each row contains three entries: one for the total sum of the protein,
        and one for each allele. Entries contain the summary statistics (mean, std, period, coherence, mean mrna) for the corresponding
        parameter set in parameter_values and the corresponding allele protein/ total protein
    '''
    summary_statistics = np.zeros((parameter_values.shape[0], 3, 5))

    pool_of_processes = mp.Pool(processes = number_of_cpus)

    process_results = [ pool_of_processes.apply_async(calculate_heterozygous_summary_statistics_at_parameter_point,
                                                      args=(parameter_value, number_of_traces_per_sample))
                        for parameter_value in parameter_values ]

    # Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    for parameter_index, process_result in enumerate(process_results):
        these_summary_statistics = process_result.get()
        summary_statistics[ parameter_index ] = these_summary_statistics

    return summary_statistics

def calculate_heterozygous_summary_statistics_at_parameter_point(parameter_value, number_of_traces = 100):
    '''Calculate the mean, relative standard deviation, period, coherence and mean mRNA
    of protein traces at one parameter point using the langevin equation.
    Will assume the arguments to be of the order described in
    generate_prior_samples. This function is necessary to ensure garbage collection of
    unnecessary traces.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    Returns
    -------

    summary_statistics : ndarray
        One dimension, five entries. Contains the summary statistics (mean, std, period, coherence, mean_mRNA) for the parameters
        in parameter_values
    '''
    if parameter_value.shape[0] == 4:
        these_mrna_traces_1, these_protein_traces_1, these_mrna_traces_2, these_protein_traces_2 = generate_multiple_heterozygous_langevin_trajectories(
                                                                                           number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_value[2], #repression_threshold,
                                                                                           5, #hill_coefficient,
                                                                                           np.log(2)/30.0, #mRNA_degradation_rate,
                                                                                           np.log(2)/90.0, #protein_degradation_rate,
                                                                                           parameter_value[0], #basal_transcription_rate,
                                                                                           parameter_value[1], #translation_rate,
                                                                                           parameter_value[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           parameter_value[2], #initial_protein,
                                                                                           1000)
    if parameter_value.shape[0] == 5:
        these_mrna_traces_1, these_protein_traces_1, these_mrna_traces_2, these_protein_traces_2 = generate_multiple_heterozygous_langevin_trajectories(
                                                                                           number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_value[2], #repression_threshold,
                                                                                           parameter_value[4], #hill_coefficient,
                                                                                           np.log(2)/30.0, #mRNA_degradation_rate,
                                                                                           np.log(2)/90.0, #protein_degradation_rate,
                                                                                           parameter_value[0], #basal_transcription_rate,
                                                                                           parameter_value[1], #translation_rate,
                                                                                           parameter_value[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           parameter_value[2], #initial_protein,
                                                                                           1000)
    elif parameter_value.shape[0] == 7:
        these_mrna_traces_1, these_protein_traces_1, these_mrna_traces_2, these_protein_traces_2 = generate_multiple_heterozygous_langevin_trajectories(
                                                                                           number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_value[2], #repression_threshold,
                                                                                           parameter_value[4], #hill_coefficient,
                                                                                           parameter_value[5], #mRNA_degradation_rate,
                                                                                           parameter_value[6], #protein_degradation_rate,
                                                                                           parameter_value[0], #basal_transcription_rate,
                                                                                           parameter_value[1], #translation_rate,
                                                                                           parameter_value[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           parameter_value[2], #initial_protein,
                                                                                           1000)
    else:
        raise ValueError("This dimension of the prior sample is not recognised.")

    summary_statistics = np.zeros((3,5))
    these_full_protein_traces = np.zeros_like(these_protein_traces_1)
    these_full_protein_traces[:,0] = these_protein_traces_1[:,0]
    these_full_protein_traces[:,1:] = these_protein_traces_1[:,1:] + these_protein_traces_2[:,1:]
    _,this_full_coherence, this_full_period = calculate_power_spectrum_of_trajectories(these_full_protein_traces, power_spectrum_smoothing_window=0.001)

    these_full_mrna_traces = np.zeros_like(these_mrna_traces_1)
    these_full_mrna_traces[:,0] = these_mrna_traces_1[:,0]
    these_full_mrna_traces[:,1:] = these_mrna_traces_1[:,1:] + these_mrna_traces_2[:,1:]

    this_full_mean = np.mean(these_full_protein_traces[:,1:])
    this_full_std = np.std(these_full_protein_traces[:,1:])/this_full_mean
    this_full_mean_mRNA = np.mean(these_full_mrna_traces[:,1:])

    _,this_allele_coherence_1, this_allele_period_1 = calculate_power_spectrum_of_trajectories(these_protein_traces_1, power_spectrum_smoothing_window=0.001)
    this_allele_mean_1 = np.mean(these_protein_traces_1[:,1:])
    this_allele_std_1 = np.std(these_protein_traces_1[:,1:])/this_allele_mean_1
    this_allele_mean_mRNA_1 = np.mean(these_mrna_traces_1[:,1:])

    _,this_allele_coherence_2, this_allele_period_2 = calculate_power_spectrum_of_trajectories(these_protein_traces_2, power_spectrum_smoothing_window=0.001)
    this_allele_mean_2 = np.mean(these_protein_traces_2[:,1:])
    this_allele_std_2 = np.std(these_protein_traces_2[:,1:])/this_allele_mean_2
    this_allele_mean_mRNA_2 = np.mean(these_mrna_traces_2[:,1:])

    summary_statistics[0][0] = this_full_mean
    summary_statistics[0][1] = this_full_std
    summary_statistics[0][2] = this_full_period
    summary_statistics[0][3] = this_full_coherence
    summary_statistics[0][4] = this_full_mean_mRNA

    summary_statistics[1][0] = this_allele_mean_1
    summary_statistics[1][1] = this_allele_std_1
    summary_statistics[1][2] = this_allele_period_1
    summary_statistics[1][3] = this_allele_coherence_1
    summary_statistics[1][4] = this_allele_mean_mRNA_1

    summary_statistics[2][0] = this_allele_mean_2
    summary_statistics[2][1] = this_allele_std_2
    summary_statistics[2][2] = this_allele_period_2
    summary_statistics[2][3] = this_allele_coherence_2
    summary_statistics[2][4] = this_allele_mean_mRNA_2

    return summary_statistics

def calculate_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 10,
                                               number_of_cpus = number_of_available_cores,
                                               model = 'langevin',
                                               timestep = 0.5,
                                               simulation_duration = 1500,
                                               power_spectrum_smoothing_window = 0.001):
    '''Calculate the mean, relative standard deviation, period, and coherence
    of protein traces at each parameter point in parameter_values.
    Will assume the arguments to be of the order described in
    generate_prior_samples.
    If Gillespie samples are used the coherence and period measures may be inaccurate.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    model : string
        options are 'langevin', 'gillespie', 'agnostic', 'gillespie_sequential'
        gillespie_sequential means that traces from different parameters will be calculated in parallel,
        rather than multiple traces from the same parameter. For the Gillespie model options, not all summary statistics are implemented
        and some of the function parameters are ignored. Use these at your own peril.

    timestep : double
        discretization timestep of the numerical scheme. Will be ignored if model is not 'langevin'

    simulation_duration : float
        The duration of the simulated time window for each trace.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Returns
    -------

    summary_statistics : ndarray
        each row contains the summary statistics (mean, std, period, coherence) for the corresponding
        parameter set in parameter_values
    '''
    if model == 'langevin' or model == 'agnostic' or model == 'gillespie_sequential':
        summary_statistics = calculate_langevin_summary_statistics_at_parameters(parameter_values,
                                                                                 number_of_traces_per_sample,
                                                                                 number_of_cpus,
                                                                                 model,
                                                                                 timestep,
                                                                                 simulation_duration,
                                                                                 power_spectrum_smoothing_window)
    else:
        if parameter_values.shape[1] != 4:
            raise ValueError("Gillespie inference on full parameter space is not implemented.")
        summary_statistics = calculate_gillespie_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample,
                                                            number_of_cpus)

    return summary_statistics

def calculate_gillespie_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 200,
                                                         number_of_cpus = number_of_available_cores):
    '''Calculate the mean, relative standard deviation, period, and coherence
    of protein traces at each parameter point in parameter_values.
    Will assume the arguments to be of the order described in
    generate_prior_samples. Since the gillespie algorithm is used, this implementation
    benefits from parallelisation by calculating individual traces of one sample in parallel.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    Returns
    -------

    summary_statistics : ndarray
        each row contains the summary statistics (mean, std, period, coherence) for the corresponding
        parameter set in parameter_values
    '''
    # may need to include parameter conversion here
    summary_statistics = np.zeros((parameter_values.shape[0], 4))
    for parameter_index, parameter_value in enumerate(parameter_values):
        these_mRNA_traces, these_protein_traces = generate_multiple_trajectories(
                                                        number_of_trajectories = number_of_traces_per_sample,
                                                        duration = 1500*5,
                                                        basal_transcription_rate = parameter_value[0],
                                                        translation_rate = parameter_value[1],
                                                        repression_threshold = parameter_value[2],
                                                        transcription_delay = parameter_value[3],
                                                        hill_coefficient = parameter_value[4],
                                                        mRNA_degradation_rate = parameter_value[5],
                                                        protein_degradation_rate = parameter_value[6],
                                                        initial_mRNA = 0,
                                                        initial_protein = parameter_value[2],
                                                        equilibration_time = 2000,
                                                        number_of_cpus = number_of_cpus)
        _,this_coherence, this_period = calculate_power_spectrum_of_trajectories(these_protein_traces, power_spectrum_smoothing_window=0.001)
        this_mean = np.mean(these_protein_traces[:,1:])
        this_std = np.std(these_protein_traces[:,1:])/this_mean
        summary_statistics[parameter_index,0] = this_mean
        summary_statistics[parameter_index,1] = this_std
        summary_statistics[parameter_index,2] = this_period
        summary_statistics[parameter_index,3] = this_coherence

    return summary_statistics

def calculate_langevin_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 100,
                                                         number_of_cpus = number_of_available_cores,
                                                         model = 'langevin',
                                                         timestep = 0.5,
                                                         simulation_duration = 1500,
                                                         power_spectrum_smoothing_window = 0.001):
    '''Calculate the mean, relative standard deviation, period, coherence, and mean mrna
    of protein traces at each parameter point in parameter_values.
    Will assume the arguments to be of the order described in
    generate_prior_samples. Since the langevin algorithm is used, this implementation
    benefits from parallelisation by calculating samples in parallel.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    model : string
        options are 'langevin', 'gillespie_sequential', 'agnostic'

    timestep : double
        discretization timestep of the numerical scheme. Will be ignored if model is not 'langevin'

    simulation_duration : float
        The duration of the simulated time window for each trace.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Returns
    -------

    summary_statistics : ndarray
        each row contains the summary statistics (mean, std, period, coherence, mean_mrna) for the corresponding
        parameter set in parameter_values
    '''
    summary_statistics = np.zeros((parameter_values.shape[0], 12))

    pool_of_processes = mp.Pool(processes = number_of_cpus)

    process_results = [ pool_of_processes.apply_async(calculate_langevin_summary_statistics_at_parameter_point,
                                                      args=(parameter_value,
                                                            number_of_traces_per_sample,
                                                            model,
                                                            timestep,
                                                            simulation_duration,
                                                            power_spectrum_smoothing_window))
                        for parameter_value in parameter_values ]

    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    for parameter_index, process_result in enumerate(process_results):
        these_summary_statistics = process_result.get()
        summary_statistics[ parameter_index ] = these_summary_statistics

    return summary_statistics

def calculate_fluctuation_rates_at_parameters(parameter_values, number_of_traces_per_sample = 200,
                                         number_of_cpus = number_of_available_cores,
                                         sampling_duration = None):
    '''Calculate the fluctuation rates at the given parameters.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    number_of_cpus : int
        number of processes that should be used for calculating the samples, parallelisation happens
        on a per-sample basis, i.e. all number_of_traces_per_sample of one sample are calculated in parallel

    sampling_duration : float
        sampling duration that should be used to calculate the fluctuation rate. This value can safely be reduced
        to 12*60 or 24*60 minutes without reducing the accuracy

    Returns
    -------

    fluctuation_rates : ndarray
        each entry contains the fluctuation rate of the the corresponding
        parameter set in parameter_values
    '''
    fluctuation_rates = np.zeros(parameter_values.shape[0])

    pool_of_processes = mp.Pool(processes = number_of_cpus)

    process_results = [ pool_of_processes.apply_async(calculate_fluctuation_rate_at_parameter_point,
                                                      args=(parameter_value,
                                                            number_of_traces_per_sample,
                                                            sampling_duration))
                        for parameter_value in parameter_values ]

    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    for parameter_index, process_result in enumerate(process_results):
        this_fluctuation_rate = process_result.get()
        fluctuation_rates[ parameter_index ] = this_fluctuation_rate

    return fluctuation_rates

def calculate_fluctuation_rate_at_parameter_point(parameter_value, number_of_traces = 100, sampling_duration = None):
    '''Calculate the fluctuation rate at a given parameter point. Will run the forward model and then
    use the autocorrelation function for estimating the fluctuation rate.

    Parameters
    ----------

    parameter_value : ndarray
        contains one model parameter set

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    sampling_duration : float
        sampling duration that should be used to calculate the fluctuation rate. This value can safely be reduced
        to 12*60 or 24*60 minutes without reducing the accuracy

    Returns
    -------

    fluctuation_rate : float
        the fluctuation rate at parameter_value

    '''
    these_mrna_traces, these_protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                                       1500*5, #duration
                                                                                       parameter_value[2], #repression_threshold,
                                                                                       parameter_value[4], #hill_coefficient,
                                                                                       parameter_value[5], #mRNA_degradation_rate,
                                                                                       parameter_value[6], #protein_degradation_rate,
                                                                                       parameter_value[0], #basal_transcription_rate,
                                                                                       parameter_value[1], #translation_rate,
                                                                                       parameter_value[3], #transcription_delay,
                                                                                       10, #initial_mRNA,
                                                                                       parameter_value[2], #initial_protein,
                                                                                       2000)

    fluctuation_rate = approximate_fluctuation_rate_of_traces_theoretically(these_protein_traces, sampling_interval = 6,
                                                                            sampling_duration = sampling_duration)

    return fluctuation_rate

def get_full_parameter_for_reduced_parameter(reduced_parameter):
    '''Transforms a parameter value of varying prior dimensions to
    a full parameter, i.e. fills in missing entries for hill coefficient or degradation rates

    Parameters
    ----------

    reduced_parameter : ndarray
        either 4, 5, or 7 entries, corresponding to prior dimension 'reduced', 'hill', or 'full'

    Returns
    -------

    full_parameter : ndarray
        7 entries, corresponding to the full model dimension
    '''
    hill_coefficient = 5
    mrna_degradation_rate = np.log(2)/30.0
    protein_degradation_rate = np.log(2)/90.0
    extrinsic_noise = 0.0
    transcription_noise_amplification = 1.0
    if reduced_parameter.shape[0] == 4:
        full_parameter = np.zeros(9)
        full_parameter[:4] = reduced_parameter
        full_parameter[4] = hill_coefficient
        full_parameter[5] = mrna_degradation_rate
        full_parameter[6] = protein_degradation_rate
        full_parameter[7] = extrinsic_noise
        full_parameter[8] = transcription_noise_amplification
    elif reduced_parameter.shape[0] == 5:
        full_parameter = np.zeros(9)
        full_parameter[:5] = reduced_parameter
        full_parameter[5] = mrna_degradation_rate
        full_parameter[6] = protein_degradation_rate
        full_parameter[7] = extrinsic_noise
        full_parameter[8] = transcription_noise_amplification
    elif reduced_parameter.shape[0] == 6:
        full_parameter = np.zeros(9)
        full_parameter[:5] = reduced_parameter[:5]
        full_parameter[5] = mrna_degradation_rate
        full_parameter[6] = protein_degradation_rate
        full_parameter[7] = reduced_parameter[-1]
        full_parameter[7] = extrinsic_noise
        full_parameter[8] = transcription_noise_amplification
    elif reduced_parameter.shape[0] == 7:
        full_parameter = np.zeros(9)
        full_parameter[:7] = reduced_parameter[:7]
        full_parameter[7] = extrinsic_noise
        full_parameter[8] = transcription_noise_amplification
    elif reduced_parameter.shape[0] == 8:
        full_parameter = np.zeros(9)
        full_parameter[:8] = reduced_parameter
        full_parameter[8] = transcription_noise_amplification
    elif reduced_parameter.shape[0] == 9:
        full_parameter = reduced_parameter
    else:
        raise ValueError("This dimension of the prior sample is not recognised.")

    return full_parameter

def calculate_langevin_summary_statistics_at_parameter_point(parameter_value, number_of_traces = 100,
                                                             model = 'langevin',
                                                             timestep = 0.5,
                                                             simulation_duration = 1500,
                                                             power_spectrum_smoothing_window = 0.001):
    '''Calculate the mean, relative standard deviation, period, coherence and mean mRNA
    of protein traces at one parameter point using the langevin equation.
    Will assume the arguments to be of the order described in
    generate_prior_samples. This function is necessary to ensure garbage collection of
    unnecessary traces.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    model : string
        options are 'langevin', 'gillespie_sequential', 'agnostic'

    timestep : double
        discretization timestep of the numerical scheme. Will be ignored if model is not 'langevin'

    Returns
    -------

    summary_statistics : ndarray
        One dimension, five entries. Contains the summary statistics (mean, std, period, coherence, mean_mRNA) for the parameters
        in parameter_values
    '''
    full_parameter = get_full_parameter_for_reduced_parameter(parameter_value)
    if model == 'langevin':
        these_mrna_traces, these_protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                                           simulation_duration, #duration
                                                                                           full_parameter[2], #repression_threshold,
                                                                                           full_parameter[4], #hill_coefficient,
                                                                                           full_parameter[5], #mRNA_degradation_rate,
                                                                                           full_parameter[6], #protein_degradation_rate,
                                                                                           full_parameter[0], #basal_transcription_rate,
                                                                                           full_parameter[1], #translation_rate,
                                                                                           full_parameter[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           full_parameter[2], #initial_protein,
                                                                                           2000,#equilibration_time
                                                                                           full_parameter[7],#extrinsic noise
                                                                                           full_parameter[8],#transcription noise amplification
                                                                                           timestep)
    elif model == 'agnostic':
        these_mrna_traces, these_protein_traces = generate_multiple_agnostic_trajectories( number_of_traces, # number_of_trajectories
                                                                                           simulation_duration, #duration
                                                                                           full_parameter[2], #repression_threshold,
                                                                                           full_parameter[4], #hill_coefficient,
                                                                                           full_parameter[5], #mRNA_degradation_rate,
                                                                                           full_parameter[6], #protein_degradation_rate,
                                                                                           full_parameter[0], #basal_transcription_rate,
                                                                                           full_parameter[1], #translation_rate,
                                                                                           full_parameter[3], #transcription_delay,
                                                                                           full_parameter[7], #mrna noise_strength
                                                                                           full_parameter[8], #protein noise_strength
                                                                                           10, #initial_mRNA,
                                                                                           full_parameter[2], #initial_protein,
                                                                                           1000)
    elif model == 'gillespie_sequential':
        these_mrna_traces, these_protein_traces = generate_multiple_trajectories( number_of_trajectories = number_of_traces, # number_of_trajectories
                                                                                  duration = simulation_duration, #duration
                                                                                  repression_threshold = full_parameter[2], #repression_threshold,
                                                                                  hill_coefficient = full_parameter[4], #hill_coefficient,
                                                                                  mRNA_degradation_rate = full_parameter[5], #mRNA_degradation_rate,
                                                                                  protein_degradation_rate = full_parameter[6], #protein_degradation_rate,
                                                                                  basal_transcription_rate = full_parameter[0], #basal_transcription_rate,
                                                                                  translation_rate = full_parameter[1], #translation_rate,
                                                                                  transcription_delay = full_parameter[3], #transcription_delay,
                                                                                  initial_mRNA = 10, #initial_mRNA,
                                                                                  initial_protein = full_parameter[2], #initial_protein,
                                                                                  equilibration_time = 2000,
                                                                                  number_of_cpus = 1 )

    this_deterministic_trace = generate_deterministic_trajectory(1500*5+2000,
                                                                full_parameter[2],
                                                                full_parameter[4],
                                                                full_parameter[5],
                                                                full_parameter[6],
                                                                full_parameter[0],
                                                                full_parameter[1],
                                                                full_parameter[3],
                                                                10,
                                                                full_parameter[2],
                                                                for_negative_times = 'no_negative')

    ## for debugging:
#     these_mrna_traces = these_mrna_traces[::10]
#     these_protein_traces = these_protein_traces[::10]
    ###
    this_deterministic_trace = this_deterministic_trace[this_deterministic_trace[:,0]>2000] # remove equilibration time
#     this_deterministic_trace = np.vstack((these_protein_traces[:,0],
#                                           these_mrna_traces[:,1],
#                                           these_protein_traces[:,1])).transpose()
    summary_statistics = np.zeros(12)
    this_power_spectrum,this_coherence, this_period = calculate_power_spectrum_of_trajectories(these_protein_traces,
                                                                                               normalize = False,
                                                                                               power_spectrum_smoothing_window = power_spectrum_smoothing_window)
    this_mean = np.mean(these_protein_traces[:,1:])
    this_std = np.std(these_protein_traces[:,1:])/this_mean
    this_mean_mRNA = np.mean(these_mrna_traces[:,1:])
    this_deterministic_mean = np.mean(this_deterministic_trace[:,2])
    this_deterministic_mean_mRNA = np.mean(this_deterministic_trace[:,1])
    this_deterministic_std = np.std(this_deterministic_trace[:,2])/this_deterministic_mean
    deterministic_protein_trace = np.vstack((this_deterministic_trace[:,0] - 2000,
                                            this_deterministic_trace[:,2])).transpose()
    _,this_deterministic_coherence, this_deterministic_period = calculate_power_spectrum_of_trajectories(deterministic_protein_trace,
                                                                                                         normalize = False,
                                                                                                         power_spectrum_smoothing_window = power_spectrum_smoothing_window)
    this_fluctuation_rate = approximate_fluctuation_rate_of_traces_theoretically(these_protein_traces, sampling_interval = 6,
                                                                                 sampling_duration = 12*60)
    this_high_frequency_weight = calculate_noise_weight_from_power_spectrum(this_power_spectrum)

    summary_statistics[0] = this_mean
    summary_statistics[1] = this_std
    summary_statistics[2] = this_period
    summary_statistics[3] = this_coherence
    summary_statistics[4] = this_mean_mRNA
    summary_statistics[5] = this_deterministic_mean
    summary_statistics[6] = this_deterministic_std
    summary_statistics[7] = this_deterministic_period
    summary_statistics[8] = this_deterministic_coherence
    summary_statistics[9] = this_deterministic_mean_mRNA
    summary_statistics[10] = this_high_frequency_weight
    summary_statistics[11] = this_fluctuation_rate

    return summary_statistics

def calculate_hilbert_periods_at_parameter_points(parameter_points,
                                                  measurement_interval = None,
                                                  per_cell = False,
                                                  smoothen = False,
                                                  samples_per_parameter_point = None):
    '''Extract all hilbert periods that can be identified across all given parameter points
    from a distribution with prior dimension 'hill', i.e. where protein and degradation rates are
    fixed at the experimental values.

    Parameters
    ----------

    parameter_points : ndarray
        must have 5 columns, as specified by prior dimension 'hill'

    measurement_interval : float
        the measurement interval to which the hilbert transform is applied. If 'None', then the full data
        length will be used. Otherwise, the data will get chopped into chunks of measurement_interval length

    per_cell : bool
        if True, get average Hilbert period for individual cells, rather than all Hilbert periods
        in one cell

    smoothen : bool
        if True then a smoothening savitzki-goyal filter with a length of 75 minutes will be applied
        For this filter we assume that time is sampled once per minute.

    samples_at_parameter_point : int
        number of samples that should be drawn at each parameter point. Only considered if per_cell is True.
        If none then 200*5*1500/measurement_interval samples will be generated

    Returns
    -------

    all_hilbert_periods: ndarray
        one-dimensional array corresponding to a list containing all obtainable hilbert periods.
    '''

    pool_of_processes = mp.Pool(processes = number_of_available_cores)

    process_results = [ pool_of_processes.apply_async(calculate_hilbert_periods_at_parameter_point,
                                                      args=(parameter_point,
                                                            measurement_interval,
                                                            per_cell,
                                                            smoothen,
                                                            samples_per_parameter_point))
                        for parameter_point in parameter_points ]

    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    all_hilbert_periods = []
    for parameter_index, process_result in enumerate(process_results):
        these_hilbert_periods = process_result.get()
        all_hilbert_periods += these_hilbert_periods

    all_hilbert_periods = np.array(all_hilbert_periods)

    return all_hilbert_periods

def calculate_hilbert_periods_at_parameter_point(parameter_point,
                                                 measurement_interval = None,
                                                 per_cell = False,
                                                 smoothen = False,
                                                 number_samples = None):
    '''Calculate all hilbert periods at a parameter point from a distribution with prior dimension 'hill',
    i.e. where protein and degradation rates are fixed at the experimental values.

    Parameters
    ----------

    parameter_point : ndarray
        must have 5 entries, as specified by prior dimension 'hill'

    measurement_interval : float
        the measurement interval to which the hilbert transform is applied. If 'None', then the full data
        length will be used. Otherwise, the data will get chopped into chunks of measurement_interval length


    per_cell : bool
        if True, get average Hilbert period for individual cells, rather than all Hilbert periods
        in one cell

    smoothen : bool
        if True then a smoothening savitzki-goyal filter with a length of 75 minutes will be applied
        For this filter we assume that time is sampled once per minute.

    number_samples : int
        number of samples that should be drawn at each parameter point. Only considered if per_cell is True.
        If none then 200*5*1500/measurement_interval samples will be generated

    Returns
    -------

    power_spectra: ndarray
        first column corresponds to frequency values, second column is the power spectrum from
        the parameter point
    '''
    if number_samples is not None and per_cell is True:
        number_of_traces = number_samples
    else:
        number_of_traces = 200

    if parameter_point.shape[0] == 5:
        mrna_traces, protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                               1500*5, #duration
                                                                               parameter_point[2], #repression_threshold,
                                                                               parameter_point[4], #hill_coefficient,
                                                                               np.log(2)/30.0, #mRNA_degradation_rate,
                                                                               np.log(2)/90.0, #protein_degradation_rate,
                                                                               parameter_point[0], #basal_transcription_rate,
                                                                               parameter_point[1], #translation_rate,
                                                                               parameter_point[3], #transcription_delay,
                                                                               10, #initial_mRNA,
                                                                               parameter_point[2], #initial_protein,
                                                                               1000)
    elif parameter_point.shape[0] == 7:
        mrna_traces, protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_point[2], #repression_threshold,
                                                                                           parameter_point[4], #hill_coefficient,
                                                                                           parameter_point[5], #mRNA_degradation_rate,
                                                                                           parameter_point[6], #protein_degradation_rate,
                                                                                           parameter_point[0], #basal_transcription_rate,
                                                                                           parameter_point[1], #translation_rate,
                                                                                           parameter_point[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           parameter_point[2], #initial_protein,
                                                                                           1000)
    else:
        ValueError("Shape of parameter_points not identified, must correspond to prior dimension 'hill' or 'full' ")

    hilbert_periods = []
    time_points = protein_traces[:,0]
    for protein_trace in protein_traces[:,1:].transpose():
        if measurement_interval == None:
            these_hilbert_periods = get_period_measurements_from_signal(protein_traces[:,0], protein_trace, smoothen)
            if per_cell:
                mean_hilbert_period = np.mean(these_hilbert_periods)
                hilbert_periods.append(mean_hilbert_period)
            else:
                hilbert_periods += these_hilbert_periods.tolist()
        else:
            final_time = time_points[-1]
            number_of_intervals = final_time/measurement_interval
            for interval_index in range(int(number_of_intervals)):
                interval_mask = np.logical_and(time_points>=(interval_index*measurement_interval),
                                               time_points<((interval_index+1)*measurement_interval))
                time_points_in_interval = time_points[interval_mask]
                signal_values_in_interval = protein_trace[interval_mask]
                these_hilbert_periods = get_period_measurements_from_signal(time_points_in_interval,
                                                                            signal_values_in_interval, smoothen)
                if per_cell:
                    if len(these_hilbert_periods) >= 1:
                        mean_hilbert_period = np.mean(these_hilbert_periods)
                    else:
                        mean_hilbert_period = measurement_interval
                    hilbert_periods.append(mean_hilbert_period)
                    if number_samples is not None and len(hilbert_periods)>= number_samples:
                        break
                else:
                    hilbert_periods += these_hilbert_periods.tolist()

            if number_samples is not None and per_cell is True and len(hilbert_periods)>= number_samples:
                break

    return hilbert_periods

def calculate_power_spectra_at_parameter_points(parameter_points):
    '''Calculate power spectra at parameter points from a distribution with prior dimension 'hill',
    i.e. where protein and degradation rates are fixed at the experimental values.

    Parameters
    ----------

    parameter_points : ndarray
        must have 5 columns, as specified by prior dimension 'hill'

    Returns
    -------

    power_spectra: ndarray
        first column corresponds to frequency values, each further column is a power spectrum from
        one parameter point
    '''
    first_power_spectrum = calculate_power_spectrum_at_parameter_point(parameter_points[0])
    power_spectra = np.zeros((first_power_spectrum.shape[0],parameter_points.shape[0] + 1))
    power_spectra[:,0] = first_power_spectrum[:,0]
    power_spectra[:,1] = first_power_spectrum[:,1]

    pool_of_processes = mp.Pool(processes = number_of_available_cores)

    process_results = [ pool_of_processes.apply_async(calculate_power_spectrum_at_parameter_point,
                                                      args=(parameter_point,))
                        for parameter_point in parameter_points[1:] ]

    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    for parameter_index, process_result in enumerate(process_results):
        this_power_spectrum = process_result.get()
        power_spectra[:, parameter_index + 2 ] = this_power_spectrum[:,1]

    return power_spectra

def calculate_power_spectrum_at_parameter_point(parameter_point):
    '''Calculate power spectrum at a parameter point from a distribution with prior dimension 'hill',
    i.e. where protein and degradation rates are fixed at the experimental values.

    Parameters
    ----------

    parameter_point : ndarray
        must have 5 entries, as specified by prior dimension 'hill'

    Returns
    -------

    power_spectra: ndarray
        first column corresponds to frequency values, second column is the power spectrum from
        the parameter point
    '''
    number_of_traces = 200
    if parameter_point.shape[0] == 5:
        mrna_traces, protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                               1500*5, #duration
                                                                               parameter_point[2], #repression_threshold,
                                                                               parameter_point[4], #hill_coefficient,
                                                                               np.log(2)/30.0, #mRNA_degradation_rate,
                                                                               np.log(2)/90.0, #protein_degradation_rate,
                                                                               parameter_point[0], #basal_transcription_rate,
                                                                               parameter_point[1], #translation_rate,
                                                                               parameter_point[3], #transcription_delay,
                                                                               10, #initial_mRNA,
                                                                               parameter_point[2], #initial_protein,
                                                                               1000)
    elif parameter_point.shape[0] == 7:
        mrna_traces, protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_point[2], #repression_threshold,
                                                                                           parameter_point[4], #hill_coefficient,
                                                                                           parameter_point[5], #mRNA_degradation_rate,
                                                                                           parameter_point[6], #protein_degradation_rate,
                                                                                           parameter_point[0], #basal_transcription_rate,
                                                                                           parameter_point[1], #translation_rate,
                                                                                           parameter_point[3], #transcription_delay,
                                                                                           10, #initial_mRNA,
                                                                                           parameter_point[2], #initial_protein,
                                                                                           1000)
    else:
        ValueError("Shape of parameter_points not identified, must correspond to prior dimension 'hill' or 'full' ")

    power_spectrum, _, _ = calculate_power_spectrum_of_trajectories(protein_traces)

    return power_spectrum

def generate_prior_samples(number_of_samples, use_langevin = True,
                           prior_bounds = {'basal_transcription_rate' : (0,100),
                                           'translation_rate' : (0,200),
                                           'repression_threshold' : (0,100000),
                                           'time_delay' : (5,40),
                                           'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
                                           'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)},
                           prior_dimension = 'reduced',
                           logarithmic = True
                           ):
    '''Sample from the prior. Provides samples of the form
    (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)

    Parameters
    ----------

    number_of_samples : int
        number of samples to draw from the prior distribution

    use_langevin : bool
        if True then time_delay samples will be of integer value

    prior_bounds : dict
        python dictionary containing parameter names and the bounds of the respective uniform prior.

    prior_dimension : string
        'reduced' or 'full', or 'hill', or 'agnostic' are possible options. If 'full', then the mRNA and protein degradation rates
        will be inferred in addition to other model parameters.

    logarithmic : bool
        if True then logarithmic priors will be assumed on the translation and transcription rates.

    Returns
    -------

    prior_samples : ndarray
        array of shape (number_of_samples,4) with columns corresponding to
        (basal_transcription_rate, translation_rate, repression_threshold, time_delay)
    '''
    if prior_dimension != 'agnostic':
        index_to_parameter_name_lookup = {0: 'basal_transcription_rate',
                                          1: 'translation_rate',
                                          2: 'repression_threshold',
                                          3: 'time_delay',
                                          4: 'hill_coefficient',
                                          5: 'mRNA_degradation_rate',
                                          6: 'protein_degradation_rate',
                                          7: 'extrinsic_noise_rate',
                                          8: 'transcription_noise_amplification'}
    else:
        index_to_parameter_name_lookup = {0: 'basal_transcription_rate',
                                          1: 'translation_rate',
                                          2: 'repression_threshold',
                                          3: 'time_delay',
                                          4: 'hill_coefficient',
                                          5: 'noise_strength',
                                          6: 'mRNA_degradation_rate',
                                          7: 'protein_degradation_rate'}

    if 'mRNA_half_life' in prior_bounds:
        index_to_parameter_name_lookup[5] = 'mRNA_half_life'

    standard_prior_bounds = {'basal_transcription_rate' : (0.5,100),
                             'translation_rate' : (1,200),
                             'repression_threshold' : (0,100000),
                             'time_delay' : (5,40),
                             'hill_coefficient' : (2,7),
                             'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
                             'protein_degradation_rate': (np.log(2)/500, np.log(2)/5),
                             'noise_strength' : (0,50),
                             'extrinsic_noise_rate' : (0.0,0.0),
                             'transcription_noise_amplification' : (1.0,1.0)}

    # depending on the function argument prior_dimension we create differently sized prior tables
    if prior_dimension == 'full':
        number_of_dimensions = 7
    elif prior_dimension == 'reduced':
        number_of_dimensions = 4
    elif prior_dimension == 'hill':
        number_of_dimensions = 5
    elif prior_dimension == 'agnostic':
        number_of_dimensions = 6
    elif prior_dimension == 'extrinsic_noise':
        number_of_dimensions = 8
    elif prior_dimension == 'transcription_noise_amplification':
        number_of_dimensions = 9
    else:
        raise ValueError("The value for prior_dimension is not recognised, must be 'reduced', 'hill, or 'full'.")

    #initialise as random numbers between (0,1)
    prior_samples = np.random.rand(number_of_samples, number_of_dimensions)

    #now scale and shift each entry as appropriate
    for parameter_index in range(number_of_dimensions):
        this_parameter_name = index_to_parameter_name_lookup[parameter_index]
        if this_parameter_name in prior_bounds:
           these_parameter_bounds = prior_bounds[this_parameter_name]
        else:
           these_parameter_bounds = standard_prior_bounds[this_parameter_name]
        if logarithmic and this_parameter_name in ['translation_rate',
                                                   'basal_transcription_rate',
                                                   'extrinsic_noise_rate']:
            if these_parameter_bounds[0] != 0.0 and these_parameter_bounds[1] != 0:
                prior_samples[:,parameter_index] = these_parameter_bounds[0]*np.power(
                                                   these_parameter_bounds[1]/float(these_parameter_bounds[0]),
                                                   prior_samples[:,parameter_index])
            else:
                prior_samples[:,parameter_index] *= 0.0
        elif this_parameter_name == 'mRNA_half_life':
            prior_samples[:,parameter_index] *= these_parameter_bounds[1] - these_parameter_bounds[0]
            prior_samples[:,parameter_index] += these_parameter_bounds[0]
            prior_samples[:,parameter_index] = np.log(2)/prior_samples[:,parameter_index]
        else:
            prior_samples[:,parameter_index] *= these_parameter_bounds[1] - these_parameter_bounds[0]
            prior_samples[:,parameter_index] += these_parameter_bounds[0]
        if this_parameter_name == 'time_delay' and use_langevin:
            these_parameter_bounds = np.around(these_parameter_bounds)
            prior_samples[:,parameter_index] = np.random.randint(these_parameter_bounds[0],
                                                                 these_parameter_bounds[1] + 1,
                                                                 number_of_samples)

    return prior_samples

def calculate_coherence_and_period_of_power_spectrum(power_spectrum):
    '''Calculate the coherence and peridod from a power spectrum.
    We define the coherence as in Phillips et al (eLife, 2016) as
    the relative area of the power spectrum
    occopied by a 20% frequency band around the maximum frequency.
    The maximum frequency corresponds to the inverse of the period.

    Parameters
    ----------

    power_spectrum : ndarray
        2D array of float values. First column contains frequencies,
        the second column contains power spectum values at these frequencies.

    Results
    -------

    coherence : float
        Coherence value as calculated around the maximum frequency band.

    period : float
        The inverse of the maximum observed frequency.
    '''
    # Calculate coherence:
    max_index = np.argmax(power_spectrum[:,1])
    max_power_frequency = power_spectrum[max_index,0]

    power_spectrum_interpolation = scipy.interpolate.interp1d(power_spectrum[:,0], power_spectrum[:,1])

    coherence_boundary_left = max_power_frequency - max_power_frequency*0.1
    coherence_boundary_right = max_power_frequency + max_power_frequency*0.1

    if coherence_boundary_left < power_spectrum[0,0]:
        coherence_boundary_left = power_spectrum[0,0]

    if coherence_boundary_right > power_spectrum[-1,0]:
        coherence_boundary_right = power_spectrum[-1,0]

    first_left_index = np.min(np.where(power_spectrum[:,0]>coherence_boundary_left))
    last_right_index = np.min(np.where(power_spectrum[:,0]>=coherence_boundary_right))
    integration_axis = np.hstack(([coherence_boundary_left],
                                  power_spectrum[first_left_index:last_right_index,0],
                                  [coherence_boundary_right]))

    interpolation_values = power_spectrum_interpolation(integration_axis)
    coherence_area = np.trapz(interpolation_values, integration_axis)
    full_area = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
    if full_area > 0.0:
        coherence = coherence_area/full_area
    else:
        coherence = 0.0

    # calculate period:
    period = 1./max_power_frequency

    return coherence, period

def smoothen_power_spectrum(power_spectrum, power_spectrum_smoothing_window = 0.001):
    """Smoothes a power spectrum using a savitzki golay filter. The number of frequency steps to use is rounded from
    the power_spectrum_smoothing_window.

    Parameters:
    -----------

    power_spectrum : ndarray
        first column contains frequencies, second column contains the power spectrum

    power_spectrum_smoothing_window : float
        The size of the smoothing window in frequency space.
        The units are 1/min.

    Returns:
    --------

    smoothened_power_spectrum : ndarray
        first column contains frequencies, second column contains the smoothened power spectrum values
    """
    # reserve memory
    smoothened_spectrum = np.zeros_like(power_spectrum[1:])
    smoothened_spectrum[:,0] = power_spectrum[1:,0]
    # figure out how many datapoints we want to consider
    # do this by figuring out how many datapoints fit in a frequency band of width 0.001
    frequency_window = power_spectrum_smoothing_window
    frequency_step = power_spectrum[1,0] - power_spectrum[0,0]
    if frequency_step < frequency_window:
        window_length = int(round(frequency_window/frequency_step))
        if window_length%2 == 0:
            window_length -= 1
        if window_length < 4:
            poly_order = window_length - 1
        else:
            poly_order = 3
        smoothened_spectrum[:,1] = scipy.signal.savgol_filter(power_spectrum[1:,1],
                                                              window_length,
                                                              poly_order)
    else:
        smoothened_spectrum = power_spectrum

    return smoothened_spectrum

def calculate_steady_state_of_ode(  repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1
                                  ):
    '''Calculate the steady state of the ODE of the Monk model at a given parameter point.
    Note that the steady state of the ODE does not depend on the delay.

    This function needs to be adjusted if the expected steady state is above 100 times
    the repression threshold.

    Parameters
    ----------

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    Returns
    -------

    steady_mRNA : float
        Number of mRNA molecules at the ODE steady state

    steady_protein : float
        Number of protein molecules at the ODE steady state.
    '''

    characteristic_constant = (translation_rate*basal_transcription_rate/
                               (mRNA_degradation_rate*protein_degradation_rate*
                                repression_threshold))

    relative_protein = scipy.optimize.brentq( ode_root_function, 0.0 , 100.0, args=(characteristic_constant, hill_coefficient))

    steady_protein = relative_protein*repression_threshold

    steady_mRNA = steady_protein*protein_degradation_rate/float(translation_rate)

    return steady_mRNA, steady_protein

def ode_root_function(x, characteristic_constant, hill_coefficient):
    '''Helper function calculate_steady_state_of_ode. Returns

       f(x) = x*(1+x^n) - D

       where D is the characteristic_constant and n is the hill_coefficient. The
       variable x measures ratios of protein over repression threshold, i.e.

       x = p/p0

       The root of this function corresponds to the steady-state protein expression
       of the ODE.

    Parameters
    ----------

    x : float
        Ratio of protein over repression threshold

    characteristic_constant :
        can be calculated as translation_rate*transcription_rate/
        (mRNA_degradation_rate*protein_degradation_rate*repression_threshold)

    Returns
    -------

    function_value : float
        the value of f(x) in the function description
    '''

    function_value = x*(1.0+np.power(x,hill_coefficient)) - characteristic_constant

    return function_value

@jit(nopython = True)
def generate_heterozygous_langevin_trajectory( duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0
                                    ):
    '''Generate one trace of the protein-autorepression model using a langevin approximation
    of the heterozygous model.
    This function implements the Ito integral of

    dM_1(2)/dt = -mu_m*M_1(2) + alpha_m*G((P_1+P_2)(t-tau) + sqrt(mu_m*M_1(2)+alpha_m*G((P1+P2(t-tau))d(ksi)
    dP_1(2)/dt = -mu_p*P_1(2) + alpha_p*M_1(2) + sqrt(mu_p*P_1(2) + alpha_p*M_1(2))d(ksi)

    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p are
    rates of mRNA degradation, protein degradation, basal transcription, and translation; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient.

    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.

    Warning : The time step of integration is chosen as 1 minute, and hence the time-delay is only
              implemented with this accuracy.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number from one allele,
        third column is Hes5 protein copy number from one allele, third column is mRNA number
        from the other allele, fourth column is protein number from the other allele.
    '''
    total_time = duration + equilibration_time
    delta_t = 0.5
    sample_times = np.arange(0.0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 5))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein
    repression_threshold = float(repression_threshold)

    mRNA_degradation_rate_per_timestep = mRNA_degradation_rate*delta_t
    protein_degradation_rate_per_timestep = protein_degradation_rate*delta_t
    basal_transcription_rate_per_timestep = basal_transcription_rate*delta_t/2.0
    translation_rate_per_timestep = translation_rate*delta_t
    delay_index_count = int(round(transcription_delay/delta_t))

    for time_index, sample_time in enumerate(sample_times[1:]):
        last_mRNA_1 = full_trace[time_index,1]
        last_protein_1 = full_trace[time_index,2]
        last_mRNA_2 = full_trace[time_index,3]
        last_protein_2 = full_trace[time_index,4]
        if time_index + 1 < delay_index_count:
            this_average_mRNA_1_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA_1
            d_mRNA_1 = (-this_average_mRNA_1_degradation_number
                      +np.sqrt(this_average_mRNA_1_degradation_number)*np.random.randn())
            this_average_mRNA_2_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA_2
            d_mRNA_2 = (-this_average_mRNA_2_degradation_number
                      +np.sqrt(this_average_mRNA_2_degradation_number)*np.random.randn())
        else:
            protein_1_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            protein_2_at_delay = full_trace[time_index + 1 - delay_index_count,4]
            protein_at_delay = protein_1_at_delay + protein_2_at_delay
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number_1 = mRNA_degradation_rate_per_timestep*last_mRNA_1
            this_average_mRNA_degradation_number_2 = mRNA_degradation_rate_per_timestep*last_mRNA_2
            d_mRNA_1 = (-this_average_mRNA_degradation_number_1
                      +this_average_transcription_number
                      +np.sqrt(this_average_mRNA_degradation_number_1
                               +this_average_transcription_number)*np.random.randn())

            d_mRNA_2 = (-this_average_mRNA_degradation_number_2
                      +this_average_transcription_number
                      +np.sqrt(this_average_mRNA_degradation_number_2
                            +this_average_transcription_number)*np.random.randn())

        this_average_protein_degradation_number_1 = protein_degradation_rate_per_timestep*last_protein_1
        this_average_protein_degradation_number_2 = protein_degradation_rate_per_timestep*last_protein_2
        this_average_translation_number_1 = translation_rate_per_timestep*last_mRNA_1
        this_average_translation_number_2 = translation_rate_per_timestep*last_mRNA_2
        d_protein_1 = (-this_average_protein_degradation_number_1
                     +this_average_translation_number_1
                     +np.sqrt(this_average_protein_degradation_number_1+
                           this_average_translation_number_1)*np.random.randn())

        d_protein_2 = (-this_average_protein_degradation_number_2
                     +this_average_translation_number_2
                     +np.sqrt(this_average_protein_degradation_number_2+
                           this_average_translation_number_2)*np.random.randn())

        current_mRNA_1 = max(last_mRNA_1 + d_mRNA_1, 0.0)
        current_protein_1  = max(last_protein_1 + d_protein_1, 0.0)
        current_mRNA_2 = max(last_mRNA_2 + d_mRNA_2, 0.0)
        current_protein_2  = max(last_protein_2 + d_protein_2, 0.0)
        full_trace[time_index + 1,1] = current_mRNA_1
        full_trace[time_index + 1,2] = current_protein_1
        full_trace[time_index + 1,3] = current_mRNA_2
        full_trace[time_index + 1,4] = current_protein_2

    # get rid of the equilibration time now
    trace = full_trace[ full_trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time

    return trace

@jit(nopython = True)
def generate_langevin_trajectory( duration = 720,
                                  repression_threshold = 10000,
                                  hill_coefficient = 5,
                                  mRNA_degradation_rate = np.log(2)/30,
                                  protein_degradation_rate = np.log(2)/90,
                                  basal_transcription_rate = 1,
                                  translation_rate = 1,
                                  transcription_delay = 29,
                                  initial_mRNA = 0,
                                  initial_protein = 0,
                                  equilibration_time = 0.0,
                                  extrinsic_noise_rate = 0.0,
                                  transcription_noise_amplification = 1.0,
                                  timestep = 0.5
                                  ):
    '''Generate one trace of the protein-autorepression model using a langevin approximation.
    This function implements the Ito integral of

    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(mu_m+alpha_m*G(P(t-tau) + sigma_e)d(ksi)
    dP/dt = -mu_p*P + alpha_p*M + sqrt(mu_p*alpha_p)d(ksi)

    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p, sigma_e are
    rates of mRNA degradation, protein degradation, basal transcription, translation, and extrinsic noise; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient.

    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.

    Warning : The time step of integration is chosen as 1 minute, and hence the time-delay is only
              implemented with this accuracy.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    extrinsic_noise_rate : float
        quantifies the effect of extrinsic noise, for example through upstream signal fluctuations,
        always positive

    transcription_noise_amplification : float
        similar to extrinsic_noise_rate. While extrinsic_noise_rate is additive and independent of expression levels,
        this parameter adds extrinsic transcriptional noise by amplifying the intrinsic noise associated with transcription.
        This paramerer is specified as a ratio between actual transcription noise and the amount of transcription noise one
        would expect if transcription was a simple, poisson / rate process.

    timestep : double
        discretization timestep of the numerical scheme. Will be ignored if model is not 'langevin',
        note that the sampling timestep of the generated traces will always be min(1 minute, timestep).

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''

    total_time = duration + equilibration_time
    delta_t = timestep
    sample_times = np.arange(0.0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 3))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein
    repression_threshold = float(repression_threshold)

    mRNA_degradation_rate_per_timestep = mRNA_degradation_rate*delta_t
    protein_degradation_rate_per_timestep = protein_degradation_rate*delta_t
    basal_transcription_rate_per_timestep = basal_transcription_rate*delta_t
    translation_rate_per_timestep = translation_rate*delta_t
    extrinsic_noise_rate_per_timestep = extrinsic_noise_rate*delta_t
    delay_index_count = int(round(transcription_delay/delta_t))

    for time_index, sample_time in enumerate(sample_times[1:]):
        last_mRNA = full_trace[time_index,1]
        last_protein = full_trace[time_index,2]
        if time_index + 1 < delay_index_count:
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +np.sqrt(this_average_mRNA_degradation_number)*np.random.randn())
        else:
            protein_at_delay = full_trace[time_index - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(this_average_mRNA_degradation_number
                            +this_average_transcription_number*transcription_noise_amplification
                            +extrinsic_noise_rate_per_timestep)*np.random.randn())

        this_average_protein_degradation_number = protein_degradation_rate_per_timestep*last_protein
        this_average_translation_number = translation_rate_per_timestep*last_mRNA
        d_protein = (-this_average_protein_degradation_number
                     +this_average_translation_number
                     +np.sqrt(this_average_protein_degradation_number+
                           this_average_translation_number)*np.random.randn())

        current_mRNA = max(last_mRNA + d_mRNA, 0.0)
        current_protein = max(last_protein + d_protein, 0.0)
        full_trace[time_index + 1,1] = current_mRNA
        full_trace[time_index + 1,2] = current_protein

    # get rid of the equilibration time now
    trace = full_trace[ full_trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time

    # ensure we only sample every minute in the final trace
    if timestep>=1.0:
        sampling_timestep_multiple = 1
    else:
        sampling_timestep_multiple = int(round(1.0/timestep))

    trace_to_return = trace[::sampling_timestep_multiple]

    return trace_to_return

@jit(nopython  = True)
def generate_time_dependent_deterministic_trajectory( duration = 720,
                                                      all_repression_thresholds = np.array([10000]*720),
                                                      all_hill_coefficients = np.array([5]*720),
                                                      all_mRNA_degradation_rates = np.array([np.log(2)/30]*720),
                                                      all_protein_degradation_rates = np.array([np.log(2)/90]*720),
                                                      all_basal_transcription_rates = np.array([1]*720),
                                                      all_translation_rates = np.array([1]*720),
                                                      all_transcription_delays = np.array([29]*720),
                                                      initial_mRNA = 0,
                                                      initial_protein = 0,
                                                      equilibration_time = 0.0
                                                      ):
    '''Generate one trace of the protein-autorepression deterministic model.
    Parameters are passed as vectors that describe temporal variations in one-minute intervals.
    This means, if a duration of N minutes is to be simulated, each parameter should be passed as a vector of length
    N+1, prescribing parameter value at each time point (including t=0.0).

    This function implements the Euler integral of

    dM/dt = -mu_m*M + alpha_m*G(P(t-tau)
    dP/dt = -mu_p*P + alpha_p*M

    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p are
    rates of mRNA degradation, protein degradation, basal transcription, and translation; in that order.
    The function G represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient.

    For negative times we assume that there was no transcription.

    Warning : The time step of integration is chosen as 1 minute, and hence the time-delay is only
              implemented with this accuracy.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    all_repression_thresholds : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    all_hill_coefficients : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    all_mRNA_degradation_rates : float
        Rate at which mRNA is degraded, in copynumber per minute

    all_protein_degradation_rates : float
        Rate at which Hes protein is degraded, in copynumber per minute

    all_basal_transcription_rates : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    all_translation_rates : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    all_transcription_delays : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    initial_mRNA : float
        amount of mRNA the integrator is initialised with.

    initial_protein : float
        amount of protein the integrator is initialised with.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
    total_time = duration + equilibration_time
    delta_t = 1
    sample_times = np.arange(0.0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 3))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein

    all_delay_index_count = all_transcription_delays//delta_t
    equilibration_index_count = int(round(equilibration_time/delta_t))+1

    for time_index, sample_time in enumerate(sample_times[1:]):
        if sample_time < equilibration_time:
            mRNA_degradation_rate_per_timestep =    all_mRNA_degradation_rates[0]
            protein_degradation_rate_per_timestep = all_protein_degradation_rates[0]
            basal_transcription_rate_per_timestep = all_basal_transcription_rates[0]
            translation_rate_per_timestep =         all_translation_rates[0]
            delay_index_count =                     all_delay_index_count[0]
            repression_threshold =                  all_repression_thresholds[0]
            hill_coefficient =                      all_hill_coefficients[0]
        else:
            mRNA_degradation_rate_per_timestep =    all_mRNA_degradation_rates[time_index - equilibration_index_count]
            protein_degradation_rate_per_timestep = all_protein_degradation_rates[time_index - equilibration_index_count]
            basal_transcription_rate_per_timestep = all_basal_transcription_rates[time_index - equilibration_index_count]
            translation_rate_per_timestep =         all_translation_rates[time_index - equilibration_index_count]
            delay_index_count =                     all_delay_index_count[time_index - equilibration_index_count]
            repression_threshold =                  all_repression_thresholds[time_index - equilibration_index_count]
            hill_coefficient =                      all_hill_coefficients[time_index - equilibration_index_count]
        last_mRNA = full_trace[time_index,1]
        last_protein = full_trace[time_index,2]
        if time_index + 1 < delay_index_count:
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = -this_average_mRNA_degradation_number
        else:
            protein_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number)

        this_average_protein_degradation_number = protein_degradation_rate_per_timestep*last_protein
        this_average_translation_number = translation_rate_per_timestep*last_mRNA
        d_protein = (-this_average_protein_degradation_number
                     +this_average_translation_number)

        current_mRNA = max(last_mRNA + d_mRNA, 0.0)
        current_protein = max(last_protein + d_protein, 0.0)
        full_trace[time_index + 1,1] = current_mRNA
        full_trace[time_index + 1,2] = current_protein

    # get rid of the equilibration time now
    trace = full_trace[ full_trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time

    return trace

@jit(nopython  = True)
def generate_time_dependent_langevin_trajectory( duration = 720,
                                  all_repression_thresholds = np.array([10000]*720),
                                  all_hill_coefficients = np.array([5]*720),
                                  all_mRNA_degradation_rates = np.array([np.log(2)/30]*720),
                                  all_protein_degradation_rates = np.array([np.log(2)/90]*720),
                                  all_basal_transcription_rates = np.array([1]*720),
                                  all_translation_rates = np.array([1]*720),
                                  all_transcription_delays = np.array([29]*720),
                                  initial_mRNA = 0,
                                  initial_protein = 0,
                                  equilibration_time = 0.0
                                  ):
    '''Generate one trace of the protein-autorepression model using a langevin approximation.
    Parameters are passed as vectors that describe temporal variations in one-minute intervals.
    This means, if a duration of N minutes is to be simulated, each parameter should be passed as a vector of length
    N+1, prescribing parameter value at each time point (including t=0.0).

    This function implements the Ito integral of

    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(mu_m+alpha_m*G(P(t-tau))d(ksi)
    dP/dt = -mu_p*P + alpha_p*M + sqrt(mu_p*alpha_p)d(ksi)

    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p are
    rates of mRNA degradation, protein degradation, basal transcription, and translation; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient.

    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.

    Warning : The time step of integration is chosen as 1 minute, and hence the time-delay is only
              implemented with this accuracy.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    all_repression_thresholds : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    all_hill_coefficients : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    all_mRNA_degradation_rates : float
        Rate at which mRNA is degraded, in copynumber per minute

    all_protein_degradation_rates : float
        Rate at which Hes protein is degraded, in copynumber per minute

    all_basal_transcription_rates : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    all_translation_rates : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    all_transcription_delays : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    initial_mRNA : float
        amount of mRNA the integrator is initialised with.

    initial_protein : float
        amount of protein the integrator is initialised with.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
    total_time = duration + equilibration_time
    delta_t = 1
    sample_times = np.arange(0.0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 3))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein

    all_delay_index_count = all_transcription_delays//delta_t
    equilibration_index_count = int(round(equilibration_time/delta_t))+1

    for time_index, sample_time in enumerate(sample_times[1:]):
        if sample_time < equilibration_time:
            mRNA_degradation_rate_per_timestep =    all_mRNA_degradation_rates[0]
            protein_degradation_rate_per_timestep = all_protein_degradation_rates[0]
            basal_transcription_rate_per_timestep = all_basal_transcription_rates[0]
            translation_rate_per_timestep =         all_translation_rates[0]
            delay_index_count =                     all_delay_index_count[0]
            repression_threshold =                  all_repression_thresholds[0]
            hill_coefficient =                      all_hill_coefficients[0]
        else:
            mRNA_degradation_rate_per_timestep =    all_mRNA_degradation_rates[time_index - equilibration_index_count]
            protein_degradation_rate_per_timestep = all_protein_degradation_rates[time_index - equilibration_index_count]
            basal_transcription_rate_per_timestep = all_basal_transcription_rates[time_index - equilibration_index_count]
            translation_rate_per_timestep =         all_translation_rates[time_index - equilibration_index_count]
            delay_index_count =                     all_delay_index_count[time_index - equilibration_index_count]
            repression_threshold =                  all_repression_thresholds[time_index - equilibration_index_count]
            hill_coefficient =                      all_hill_coefficients[time_index - equilibration_index_count]
        last_mRNA = full_trace[time_index,1]
        last_protein = full_trace[time_index,2]
        if time_index + 1 < delay_index_count:
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +np.sqrt(this_average_mRNA_degradation_number)*np.random.randn())
        else:
            protein_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(this_average_mRNA_degradation_number
                            +this_average_transcription_number)*np.random.randn())

        this_average_protein_degradation_number = protein_degradation_rate_per_timestep*last_protein
        this_average_translation_number = translation_rate_per_timestep*last_mRNA
        d_protein = (-this_average_protein_degradation_number
                     +this_average_translation_number
                     +np.sqrt(this_average_protein_degradation_number+
                           this_average_translation_number)*np.random.randn())

        current_mRNA = max(last_mRNA + d_mRNA, 0.0)
        current_protein = max(last_protein + d_protein, 0.0)
        full_trace[time_index + 1,1] = current_mRNA
        full_trace[time_index + 1,2] = current_protein

    # get rid of the equilibration time now
    trace = full_trace[ full_trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time

    return trace

@jit(nopython = True)
def generate_agnostic_noise_trajectory( duration = 720,
                                        repression_threshold = 10000,
                                        hill_coefficient = 5,
                                        mRNA_degradation_rate = np.log(2)/30,
                                        protein_degradation_rate = np.log(2)/90,
                                        basal_transcription_rate = 1,
                                        translation_rate = 1,
                                        transcription_delay = 29,
                                        mRNA_noise_strength = 10,
                                        protein_noise_strength = 10,
                                        initial_mRNA = 0,
                                        initial_protein = 0,
                                        equilibration_time = 0.0,
                                        time_step = 1.0,
                                        sampling_frequency = 1.0
                                        ):
    '''Generate one trace of the protein-autorepression model using a langevin approximation.
    This function implements the Ito integral of

    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(sigma_mRNA)d(ksi_m)
    dP/dt = -mu_p*P + alpha_p*M+ sqrt(sigma_protein)d(ksi_p)

    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p are
    rates of mRNA degradation, protein degradation, basal transcription, and translation; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient.

    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.

    Warning : The time step of integration is chosen as 1 minute, and hence the time-delay is only
              implemented with this accuracy.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    noise_strength : float
        strength of the noise term, sigma

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    time_step : float
        the time step used in the integration

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''

    total_time = duration + equilibration_time
    delta_t = time_step
    sample_times = np.arange(0.0, total_time, delta_t)
    sampling_times = np.linspace(equilibration_time, total_time,
                                 int(round((total_time-equilibration_time)/sampling_frequency)))[:-1]
    sampled_trace = np.zeros(( len(sampling_times), 3))
    sampled_trace[:,0] = sampling_times
    full_trace = np.zeros(( len(sample_times), 3))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein
    repression_threshold = float(repression_threshold)

    mRNA_degradation_rate_per_timestep = mRNA_degradation_rate*delta_t
    protein_degradation_rate_per_timestep = protein_degradation_rate*delta_t
    basal_transcription_rate_per_timestep = basal_transcription_rate*delta_t
    translation_rate_per_timestep = translation_rate*delta_t
    mRNA_noise_rate_per_timestep = mRNA_noise_strength*delta_t
    protein_noise_rate_per_timestep = protein_noise_strength*delta_t
    delay_index_count = int(round(transcription_delay/delta_t))

    sampling_index = 0
    for time_index, sample_time in enumerate(sample_times[1:]):
        last_mRNA = full_trace[time_index,1]
        last_protein = full_trace[time_index,2]
        if time_index + 1 < delay_index_count:
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +np.sqrt(mRNA_noise_rate_per_timestep)*np.random.randn())
        else:
            protein_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(mRNA_noise_rate_per_timestep)*np.random.randn())

        this_average_protein_degradation_number = protein_degradation_rate_per_timestep*last_protein
        this_average_translation_number = translation_rate_per_timestep*last_mRNA
        d_protein = (-this_average_protein_degradation_number
                     +this_average_translation_number
                     +np.sqrt(protein_noise_rate_per_timestep)*np.random.randn())

        current_mRNA = max(last_mRNA + d_mRNA, 0.0)
        current_protein = max(last_protein + d_protein, 0.0)
        full_trace[time_index + 1,1] = current_mRNA
        full_trace[time_index + 1,2] = current_protein
        if sample_time >= sampling_times[sampling_index]:
            sampled_trace[sampling_index,1] = current_mRNA
            sampled_trace[sampling_index,2] = current_protein
            sampling_index += 1

    # get rid of the equilibration time now
#     trace = sampled_trace[ sampled_trace[:,0]>=equilibration_time ]
    trace = sampled_trace
    trace[:,0] -= equilibration_time

    return trace

def generate_multiple_agnostic_trajectories( number_of_trajectories = 10,
                                    duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    mRNA_noise_strength = 10,
                                    protein_noise_strength = 10,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0):
    '''Generate multiple langevin stochastic traces from the Monk model by using
       generate_langevin_trajectory.

    Parameters
    ----------

    number_of_trajectories : int
        number of trajectories that should be calculated

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    noise_strength : float
        strength of the noise term, sigma

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    Returns
    -------

    mRNA_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of mRNA copy numbers

    protein_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of protein copy numbers
    '''
    first_trace = generate_agnostic_noise_trajectory(duration,
                                                     repression_threshold,
                                                     hill_coefficient,
                                                     mRNA_degradation_rate,
                                                     protein_degradation_rate,
                                                     basal_transcription_rate,
                                                     translation_rate,
                                                     transcription_delay,
                                                     mRNA_noise_strength,
                                                     protein_noise_strength,
                                                     initial_mRNA,
                                                     initial_protein,
                                                     equilibration_time)

    sample_times = first_trace[:,0]
    mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time

    mRNA_trajectories[:,0] = sample_times
    protein_trajectories[:,0] = sample_times
    mRNA_trajectories[:,1] = first_trace[:,1]
    protein_trajectories[:,1] = first_trace[:,2]

    for trajectory_index in range(1,number_of_trajectories):
        # offset one index for time column
        this_trace = generate_agnostic_noise_trajectory(duration,
                                                  repression_threshold,
                                                  hill_coefficient,
                                                  mRNA_degradation_rate,
                                                  protein_degradation_rate,
                                                  basal_transcription_rate,
                                                  translation_rate,
                                                  transcription_delay,
                                                  mRNA_noise_strength,
                                                  protein_noise_strength,
                                                  initial_mRNA,
                                                  initial_protein,
                                                  equilibration_time)

        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1]
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]

    return mRNA_trajectories, protein_trajectories

def generate_multiple_heterozygous_langevin_trajectories( number_of_trajectories = 10,
                                    duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0):
    '''Generate multiple langevin stochastic traces from the heterozygous model by using
       generate_heterozygous_langevin_trajectory.

    Parameters
    ----------

    number_of_trajectories : int
        number of trajectories that should be calculated

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    Returns
    -------

    mRNA_trajectories_1 : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of mRNA copy numbers from one allele

    protein_trajectories_1 : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of protein copy numbers from one allele

    mRNA_trajectories_2 : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of mRNA copy numbers from the other allele

    protein_trajectories_2 : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of protein copy numbers from the other allele
    '''
    first_trace = generate_heterozygous_langevin_trajectory(duration,
                                               repression_threshold,
                                               hill_coefficient,
                                               mRNA_degradation_rate,
                                               protein_degradation_rate,
                                               basal_transcription_rate,
                                               translation_rate,
                                               transcription_delay,
                                               initial_mRNA,
                                               initial_protein,
                                               equilibration_time)

    sample_times = first_trace[:,0]
    mRNA_trajectories_1 = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories_1 = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    mRNA_trajectories_2 = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories_2 = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time

    mRNA_trajectories_1[:,0] = sample_times
    protein_trajectories_1[:,0] = sample_times
    mRNA_trajectories_1[:,1] = first_trace[:,1]
    protein_trajectories_1[:,1] = first_trace[:,2]
    mRNA_trajectories_2[:,0] = sample_times
    protein_trajectories_2[:,0] = sample_times
    mRNA_trajectories_2[:,1] = first_trace[:,3]
    protein_trajectories_2[:,1] = first_trace[:,4]

    for trajectory_index in range(1,number_of_trajectories):
        # offset one index for time column
        this_trace = generate_heterozygous_langevin_trajectory(duration,
                                               repression_threshold,
                                               hill_coefficient,
                                               mRNA_degradation_rate,
                                               protein_degradation_rate,
                                               basal_transcription_rate,
                                               translation_rate,
                                               transcription_delay,
                                               initial_mRNA,
                                               initial_protein,
                                               equilibration_time)

        mRNA_trajectories_1[:,trajectory_index + 1] = this_trace[:,1]
        protein_trajectories_1[:,trajectory_index + 1] = this_trace[:,2]
        mRNA_trajectories_2[:,trajectory_index + 1] = this_trace[:,3]
        protein_trajectories_2[:,trajectory_index + 1] = this_trace[:,4]

    return mRNA_trajectories_1, protein_trajectories_1, mRNA_trajectories_2, protein_trajectories_2

## jitting this function does not seem to improve runtimes further
def generate_multiple_langevin_trajectories( number_of_trajectories = 10,
                                    duration = 720,
                                    repression_threshold = 10000,
                                    hill_coefficient = 5,
                                    mRNA_degradation_rate = np.log(2)/30,
                                    protein_degradation_rate = np.log(2)/90,
                                    basal_transcription_rate = 1,
                                    translation_rate = 1,
                                    transcription_delay = 29,
                                    initial_mRNA = 0,
                                    initial_protein = 0,
                                    equilibration_time = 0.0,
                                    extrinsic_noise_rate = 0.0,
                                    transcription_noise_amplification = 1.0,
                                    timestep = 0.5):
    '''Generate multiple langevin stochastic traces from the Monk model by using
       generate_langevin_trajectory.

    Parameters
    ----------

    number_of_trajectories : int
        number of trajectories that should be calculated

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    hill_coefficient : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mRNA_degradation_rate : float
        Rate at which mRNA is degraded, in copynumber per minute

    protein_degradation_rate : float
        Rate at which Hes protein is degraded, in copynumber per minute

    basal_transcription_rate : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    translation_rate : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    transcription_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        in order to get rid of any overshoots, for example

    extrinsic_noise_rate : float
        quantifies the effect of extrinsic noise, for example through upstream signal fluctuations,
        always positive

    transcription_noise_amplification : float
        similar to extrinsic_noise_rate. While extrinsic_noise_rate is additive and independent of expression levels,
        this parameter adds extrinsic transcriptional noise by amplifying the intrinsic noise associated with transcription.
        This paramerer is specified as a ratio between actual transcription noise and the amount of transcription noise one
        would expect if transcription was a simple, poisson / rate process.

    timestep : double
        discretization timestep of the numerical scheme. Will be ignored if model is not 'langevin',
        note that the sampling timestep of the generated traces will always be min(1 minute, timestep).

    Returns
    -------

    mRNA_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of mRNA copy numbers

    protein_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time,
        each further column is one trace of protein copy numbers
    '''
    first_trace = generate_langevin_trajectory(duration,
                                               repression_threshold,
                                               hill_coefficient,
                                               mRNA_degradation_rate,
                                               protein_degradation_rate,
                                               basal_transcription_rate,
                                               translation_rate,
                                               transcription_delay,
                                               initial_mRNA,
                                               initial_protein,
                                               equilibration_time,
                                               extrinsic_noise_rate,
                                               transcription_noise_amplification,
                                               timestep)

    sample_times = first_trace[:,0]
    mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time

    mRNA_trajectories[:,0] = sample_times
    protein_trajectories[:,0] = sample_times
    mRNA_trajectories[:,1] = first_trace[:,1]
    protein_trajectories[:,1] = first_trace[:,2]

    for trajectory_index in range(1,number_of_trajectories):
        # offset one index for time column
        this_trace = generate_langevin_trajectory(duration,
                                               repression_threshold,
                                               hill_coefficient,
                                               mRNA_degradation_rate,
                                               protein_degradation_rate,
                                               basal_transcription_rate,
                                               translation_rate,
                                               transcription_delay,
                                               initial_mRNA,
                                               initial_protein,
                                               equilibration_time,
                                               extrinsic_noise_rate,
                                               transcription_noise_amplification,
                                               timestep)

        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1]
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]

    return mRNA_trajectories, protein_trajectories

def conduct_all_parameter_sweeps_at_parameters(parameter_samples,
                                               number_of_sweep_values = 20,
                                               number_of_traces_per_parameter = 200,
                                               relative = False,
                                               relative_range = (0.1,2.0),
                                               simulation_timestep = 1.0,
                                               simulation_duration = 1500*5,
                                               power_spectrum_smoothing_window = 0.001):
    '''Conduct a parameter sweep in reasonable ranges of each of the parameter points in
    parameter_samples. The parameter_samples are four-dimensional, as produced, for example, by
    generate_prior_samples() with the 'reduced' dimension. At each parameter point the function
    will sweep over number_of_sweep_values of the parameters

    basal_transcription_rate, translation_rate, repression_threshold, time_delay, hill_coefficient

    , and from number_of_trajectories langevin traces the summary statistics [mean expression
    standard_deviation, period, coherence] will be returned.

    Parameters:
    -----------

    parameter_samples : ndarray
        four columns, each row corresponds to one parameter. The columns are in the order returned by
        generate_prior_samples().

    number_of_degradation_values : int
        number of different protein degradation rates to consider. These number of values will be evenly spaced
        between 0.0001 and np.log(2)/15

    number_of_traces_per_parameter : int
        number of traces that should be used to calculate summary statistics

    relative : bool
        If true x values will not be parameter values but percentage values in changes specified in relative_range

    relative_range : tuple of two floats
        the proportion ranges that is considered. The number_of_sweep_values will be evenly spaced between relative_range[0]*actual_parameter_value to
        relative_range[1]*actual_parameter_value

    simulation_timestep : float
        The discretisation time step of the simulation.

    simulation_duration : float
        The duration of the simulated time window for each trace.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Results:
    --------

    sweep_results : dict()
        Keys are parameter names as in the main function description.
        Each entry contains a three-dimensional array. Each entry along the first dimension corresponds to one
        parameter in parameter_samples and contains a 2d array where the first column is a
        protein_degradation_rate and each further column contains the summary statistics in
        the order described above.
    '''
    parameter_names = ['basal_transcription_rate',
                       'translation_rate',
                       'repression_threshold',
                       'time_delay',
                       'mRNA_degradation_rate',
                       'protein_degradation_rate',
                       'hill_coefficient']

#     parameter_names = ['repression_threshold']

    sweep_results = dict()

    for parameter_name in parameter_names:
        these_results = conduct_parameter_sweep_at_parameters(parameter_name,
                                                              parameter_samples,
                                                              number_of_sweep_values,
                                                              number_of_traces_per_parameter,
                                                              relative,
                                                              relative_range,
                                                              simulation_timestep,
                                                              simulation_duration,
                                                              power_spectrum_smoothing_window)

        sweep_results[parameter_name] = these_results

    return sweep_results

def conduct_dual_parameter_sweep_at_parameters(parameter_samples,
                                               degradation_range = (0.1,2.0),
                                               translation_range = (0.1,2.0),
                                               degradation_interval_number = 20,
                                               translation_interval_number = 20,
                                               number_of_traces_per_parameter = 200,
                                               simulation_timestep = 0.5,
                                               simulation_duration = 1500,
                                               power_spectrum_smoothing_window = 0.02):
    '''Conduct a simultaneous (dual) parameter sweep of the mRNA degradation rate and
    the translation rate at each of the parameter points in
    parameter_samples. The parameter_samples are seven-dimensional, as produced, for example, by
    generate_prior_samples() with the 'full' dimension. At each parameter point the function
    will sweep over number_sweep_values of mrna degradation, and the same number of translation rate values,
    and from number_of_trajectories langevin traces the summary statistics [mean expression
    standard_deviation, period, coherence] will be returned.

    Parameters:
    -----------

    parameter_samples : ndarray
        four columns, each row corresponds to one parameter. The columns are in the order returned by
        generate_prior_samples().

    degradation_range : (float,float)
        lower and upper relative boundaries of the degradation parameter sweep.
        Relative to the original degradation rate.

    translation_range : (float,float)
        lower and upper relative boundaries of the degradation parameter sweep.
        Relative to the original degradation rate.

    degradation_interval_number : int
        number of different parameter values to consider. These number of values will be evenly spaced
        within degradation_range.

    translation_interval_number : int
        number of different parameter values to consider. These number of values will be evenly spaced
        within translation_range. The total number of parameter variations per input parameter
        that is generated in this sweep will be degradation_interval_number*translation_interval_number

    number_of_traces_per_parameter : int
        number of traces that should be used to calculate summary statistics

    simulation_timestep : float
        The discretisation time step of the simulation.

    simulation_duration : float
        The duration of the simulated time window for each trace.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Results:
    --------

    sweep_results : ndarray
        three-dimensional array. Each entry along the first dimension corresponds to one parameter
        in parameter_samples and contains a 2d array where the first column is a value of parameter_name
        and each further column contains the summary statistics in the order described above.
    '''
    # first: make a table of 7d parameters
    total_number_of_parameters_required = parameter_samples.shape[0]*(degradation_interval_number*translation_interval_number)
    all_parameter_values = np.zeros((total_number_of_parameters_required, parameter_samples.shape[1]))
    parameter_sample_index = 0
    for sample in parameter_samples:
        for degradation_proportion in np.linspace(degradation_range[0],degradation_range[1],degradation_interval_number):
            for translation_proportion in np.linspace(translation_range[0],translation_range[1],translation_interval_number):
                all_parameter_values[parameter_sample_index] = sample
                # now replace the parameter of interest with the actual parameter value
                # degradation rate
                all_parameter_values[parameter_sample_index, 5] *= degradation_proportion
                # translation rate
                all_parameter_values[parameter_sample_index, 1] *= translation_proportion
                parameter_sample_index += 1

    # pass these parameters to the calculate_summary_statistics_at_parameter_points
    all_summary_statistics = calculate_summary_statistics_at_parameters(parameter_values = all_parameter_values,
                                                                        number_of_traces_per_sample = number_of_traces_per_parameter,
                                                                        number_of_cpus = number_of_available_cores,
                                                                        model = 'langevin',
                                                                        timestep = simulation_timestep,
                                                                        simulation_duration = simulation_duration,
                                                                        power_spectrum_smoothing_window = power_spectrum_smoothing_window)

    # unpack and wrap the results in the output format
    sweep_results = np.zeros((parameter_samples.shape[0],
                              degradation_interval_number,
                              translation_interval_number,
                              14))
    parameter_sample_index = 0
    for sample_index, sample in enumerate(parameter_samples):
        degradation_proportion_index = 0
        for degradation_proportion in np.linspace(degradation_range[0],degradation_range[1],degradation_interval_number):
            translation_proportion_index = 0
            for translation_proportion in np.linspace(translation_range[0],translation_range[1],translation_interval_number):
                these_summary_statistics = all_summary_statistics[parameter_sample_index]
                # the first entry gets the degradation rate
                sweep_results[sample_index,degradation_proportion_index,
                              translation_proportion_index,:2] = [degradation_proportion, translation_proportion]
                # the remaining entries get the summary statistics. We discard the last summary statistic,
                # which is the mean mRNA
                sweep_results[sample_index,degradation_proportion_index,
                              translation_proportion_index,2:] = these_summary_statistics
                translation_proportion_index+=1
                parameter_sample_index+= 1
            degradation_proportion_index+=1

    return sweep_results

def conduct_parameter_sweep_at_parameters(parameter_name,
                                          parameter_samples,
                                          number_of_sweep_values = 20,
                                          number_of_traces_per_parameter = 200,
                                          relative = False,
                                          relative_range = (0.1,2.0),
                                          simulation_timestep = 1.0,
                                          simulation_duration = 1500*5,
                                          power_spectrum_smoothing_window = 0.001):
    '''Conduct a parameter sweep of the parameter_name parameter at each of the parameter points in
    parameter_samples. At each parameter point the function
    will sweep over number_sweep_values of parameter_name, and from
    number_of_trajectories langevin traces the summary statistics [mean expression
    standard_deviation, period, coherence] will be returned. Parameter ranges are hardcoded and
    can be seen at the top of the function implementation.

    Parameters:
    -----------

    parameter_samples : ndarray
        four columns, each row corresponds to one parameter. The columns are in the order returned by
        generate_prior_samples().

    number_of_sweep_values : int
        number of different parameter values to consider. These number of values will be evenly spaced
        between reasonable ranges for each parameter.

    number_of_traces_per_parameter : int
        number of traces that should be used to calculate summary statistics

    relative : bool
        If true x values will not be parameter values but percentage values in changes specified in relative_range

    relative_range : tuple of two floats
        the proportion ranges that is considered. The number_of_sweep_values will be evenly spaced between relative_range[0]*actual_parameter_value to
        relative_range[1]*actual_parameter_value

    simulation_timestep : float
        The discretisation time step of the simulation.

    simulation_duration : float
        The duration of the simulated time window for each trace.

    power_spectrum_smoothing_window : float
        When coherence and period are calculated from the power spectrum, the spectrum is first smoothed using a savitzki golay filter
        to reduce the impact of sampling noise. This parameter allows the user to define the size of that window in frequency space.
        The units are 1/min.

    Results:
    --------

    sweep_results : ndarray
        three-dimensional array. Each entry along the first dimension corresponds to one parameter
        in parameter_samples and contains a 2d array where the first column is a value of parameter_name
        and each further column contains the summary statistics in the order described above.
    '''
    parameter_indices_and_ranges = dict()
    parameter_indices_and_ranges['basal_transcription_rate'] = (0,(0.0,100.0))
    parameter_indices_and_ranges['translation_rate'] =         (1,(0.0,100.0))
    parameter_indices_and_ranges['repression_threshold'] =     (2,(1.0,100000.0))
    parameter_indices_and_ranges['time_delay'] =               (3,(5.0,40.0))
    parameter_indices_and_ranges['hill_coefficient'] =         (4,(1,10))
    parameter_indices_and_ranges['mRNA_degradation_rate'] =    (5,(np.log(2)/500,np.log(2)/15))
    parameter_indices_and_ranges['protein_degradation_rate'] = (6,(np.log(2)/5000,np.log(2)/15))

    # first: make a table of 7d parameters
    total_number_of_parameters_required = parameter_samples.shape[0]*number_of_sweep_values
    all_parameter_values = np.zeros((total_number_of_parameters_required, 7))
    parameter_sample_index = 0
    index_of_parameter_name = parameter_indices_and_ranges[parameter_name][0]
    if not relative:
        left_sweep_boundary = parameter_indices_and_ranges[parameter_name][1][0]
        right_sweep_boundary = parameter_indices_and_ranges[parameter_name][1][1]
        for sample in parameter_samples:
            for sweep_value in np.linspace(left_sweep_boundary, right_sweep_boundary, number_of_sweep_values):
                if len(sample) == 4:
                    all_parameter_values[parameter_sample_index,:4] = sample
                    all_parameter_values[parameter_sample_index,4] = 5 #Hill coefficient
                    all_parameter_values[parameter_sample_index,5] = np.log(2)/30.0 # tag on mrna degradation rate
                    all_parameter_values[parameter_sample_index,6] = np.log(2)/90.0 # and protein degradation rate
                elif len(sample) == 5:
                    all_parameter_values[parameter_sample_index,:5] = sample
                    all_parameter_values[parameter_sample_index,5] = np.log(2)/30.0 # tag on mrna degradation rate
                    all_parameter_values[parameter_sample_index,6] = np.log(2)/90.0 # and protein degradation rate
                elif len(sample) == 6:
                    all_parameter_values[parameter_sample_index,:4] = sample[:4]
                    all_parameter_values[parameter_sample_index,4] = 5 #Hill coefficient
                    all_parameter_values[parameter_sample_index,5:] = sample[4:] # tag on mrna and protein degradation rate
                else:
                    all_parameter_values[parameter_sample_index] = sample
                # now replace the parameter of interest with the actual parameter value
                all_parameter_values[parameter_sample_index, index_of_parameter_name] = sweep_value
                parameter_sample_index += 1
    else:
        for sample in parameter_samples:
            if parameter_indices_and_ranges[parameter_name][0] < len(sample):
                this_parameter = sample[ parameter_indices_and_ranges[parameter_name][0] ]
            elif parameter_name == 'mRNA_degradation_rate':
                this_parameter = np.log(2)/30.0
            elif parameter_name == 'protein_degradation_rate':
                this_parameter = np.log(2)/90.0
            elif parameter_name == 'hill_coefficient':
                this_parameter = 5.0
            for parameter_proportion in np.linspace(relative_range[0],relative_range[1],number_of_sweep_values):
                if len(sample) == 4:
                    all_parameter_values[parameter_sample_index,:4] = sample
                    all_parameter_values[parameter_sample_index,4] = 5 #Hill coefficient
                    all_parameter_values[parameter_sample_index,5] = np.log(2)/30.0 # tag on mrna degradation rate
                    all_parameter_values[parameter_sample_index,6] = np.log(2)/90.0 # and protein degradation rate
                elif len(sample) == 5:
                    all_parameter_values[parameter_sample_index,:5] = sample
                    all_parameter_values[parameter_sample_index,5] = np.log(2)/30.0 # tag on mrna degradation rate
                    all_parameter_values[parameter_sample_index,6] = np.log(2)/90.0 # and protein degradation rate
                elif len(sample) == 6:
                    all_parameter_values[parameter_sample_index,:4] = sample[:4]
                    all_parameter_values[parameter_sample_index,4] = 5 #Hill coefficient
                    all_parameter_values[parameter_sample_index,5:] = sample[4:] # tag on mrna and protein degradation rate
                else:
                    all_parameter_values[parameter_sample_index] = sample
                # now replace the parameter of interest with the actual parameter value
                all_parameter_values[parameter_sample_index, index_of_parameter_name] = parameter_proportion*this_parameter
                parameter_sample_index += 1

    # pass these parameters to the calculate_summary_statistics_at_parameter_points
    all_summary_statistics = calculate_summary_statistics_at_parameters(parameter_values = all_parameter_values,
                                                                        number_of_traces_per_sample = number_of_traces_per_parameter,
                                                                        number_of_cpus = number_of_available_cores,
                                                                        model = 'langevin',
                                                                        timestep = simulation_timestep,
                                                                        simulation_duration = simulation_duration,
                                                                        power_spectrum_smoothing_window=power_spectrum_smoothing_window)

    # unpack and wrap the results in the output format
    sweep_results = np.zeros((parameter_samples.shape[0], number_of_sweep_values, 13))
    parameter_sample_index = 0
    if not relative:
        for sample_index, sample in enumerate(parameter_samples):
            sweep_value_index = 0
            for sweep_value in np.linspace(left_sweep_boundary, right_sweep_boundary, number_of_sweep_values):
                these_summary_statistics = all_summary_statistics[parameter_sample_index]
                # the first entry gets the degradation rate
                sweep_results[sample_index,sweep_value_index,0] = sweep_value
                # the remaining entries get the summary statistics. We discard the last summary statistic,
                # which is the mean mRNA
                sweep_results[sample_index,sweep_value_index,1:] = these_summary_statistics
                sweep_value_index += 1
                parameter_sample_index += 1
    else:
        for sample_index, sample in enumerate(parameter_samples):
            proportion_index = 0
            for parameter_proportion in np.linspace(relative_range[0],relative_range[1],number_of_sweep_values):
                these_summary_statistics = all_summary_statistics[parameter_sample_index]
                # the first entry gets the degradation rate
                sweep_results[sample_index,proportion_index,0] = parameter_proportion
                # the remaining entries get the summary statistics. We discard the last summary statistic,
                # which is the mean mRNA
                sweep_results[sample_index,proportion_index,1:] = these_summary_statistics
                proportion_index += 1
                parameter_sample_index += 1
    # repack results into output array

    return sweep_results

def detrend_experimental_data(data):
    '''Detrend experimental data using Gaussian process regression with a squared exponential kernel (RBF),
    using a fixed lengthscale of roughly 3 times the period of the data (~1000 mins) and optimising the variance
    with BFGS.

    Parameters:
    ----------

    data : ndarray
        An nx2 array whose first column is time and second column is protein copy number.

    Returns:
    -------

    detrended_data : ndarray
        An nx2 array whose first column is time and second column is protein copy number.
        The long term trend from the original data set is removed, resulting in this array.

    y_gpr : ndarray
        An nx1 array which returns the posterior prediction from the Gaussian process regressor,
        for all specified time points.

    y_std : ndarray
        The standard deviation in the posterior prediction, y_gpr.
    '''
    length_scale = 1000
    gp_kernel = (gp.kernels.ConstantKernel(constant_value=np.var(data[:,1]), constant_value_bounds=(0.1*np.var(data[:,1]),2*np.var(data[:,1])))*
                 gp.kernels.RBF(length_scale=length_scale,length_scale_bounds=(length_scale,2*length_scale)) +
                 gp.kernels.WhiteKernel(1e2,noise_level_bounds=(1e-5,np.var(data[:,1]))))
    gpr = gp.GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(data[:,0].reshape(-1,1),data[:,1] - np.mean(data[:,1]))
    X_plot = np.linspace(0, np.int(data[-1,0]), np.int(data[-1,0])+1)[:, None]
    y_gpr, y_std = gpr.predict(X_plot,return_std=True)
    detrended_data = np.zeros((data.shape[0],data.shape[1]))
    detrended_data[:,0] = data[:,0]
    detrended_data[:,1] = data[:,1] - y_gpr[data[:,0].astype(int)]
    return detrended_data, y_gpr, y_std

def measure_fluctuation_rate_of_single_trace(trace, method = 'sklearn'):
    '''Calculate the fluctation rate of a trace. Will fit an Ornstein-Uhlenbeck Gaussian process
    to a time series and estimate the lengthscale parameter, alpha. Specifically, we estimate the parameter
    alpha = 1/rho
    in the matern kernel with parameter nu=1/2:
    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    https://gpflow.readthedocs.io/en/stable/_modules/gpflow/kernels.html

    Parameters:
    ----------

    trace : ndarray
        2D array. First column is time, second column contains the signal that is aimed to be analysed.

    method : string
        'gpflow', 'sklearn', 'gpy', and 'george' are possible. These are names of common libraries for
        Gaussian processes.

    Result:
    ------

    fluctuation_rate : float
        fluctuation rate of trace
    '''

    times = trace[:,0]
    times = times[:,np.newaxis]
    trace_around_mean = trace[:,1] - np.mean(trace[:,1])
    trace_around_mean = trace_around_mean[:,np.newaxis]

    if method == 'gpflow':
        with gpflow.defer_build():
            ornstein_kernel = gpflow.kernels.Matern12(input_dim = 1)
            regression_model = gpflow.models.GPR(times, trace_around_mean, kern=ornstein_kernel)
            regression_model.kern.lengthscales.prior = gpflow.priors.Uniform(1e-2, 10000)
        regression_model.compile()
        gpflow.train.ScipyOptimizer().minimize(regression_model)
        regression_values = regression_model.kern.read_values()
        this_lengthscale = regression_values['GPR/kern/lengthscales']
    elif method == 'sklearn':
        ornstein_kernel = ( gp.kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 2.0*np.var(trace_around_mean)))*
                            gp.kernels.Matern(nu=0.5, length_scale_bounds = (1e-2,10000)))
        my_gp_regressor = gp.GaussianProcessRegressor(kernel=ornstein_kernel, n_restarts_optimizer=1)
        my_fit = my_gp_regressor.fit(times, trace_around_mean)
        this_lengthscale = my_gp_regressor.kernel_.get_params()['k2__length_scale']
    elif method == 'gpy':
        ornstein_kernel = GPy.kern.OU(input_dim=1)
        my_regressor = GPy.models.GPRegression(times,trace_around_mean,ornstein_kernel)
        my_regressor.optimize_restarts(num_restarts = 1)
        this_lengthscale = my_regressor.parameters[0].lengthscale.values[0]
    elif method == 'george':
        kernel = george.kernels.Product(george.kernels.ConstantKernel(log_constant = 0.0),
                                        george.kernels.ExpKernel(metric = 1.0))
        gaussian_process = george.GP(kernel)
        #initialise the x-values of the process
        gaussian_process.compute(times[:,0])

        def objective_function(hyper_parameter):
            gaussian_process.set_parameter_vector(hyper_parameter)
            log_likelihood = gaussian_process.log_likelihood(trace_around_mean[:,0], quiet=True)
            return -log_likelihood if np.isfinite(log_likelihood) else 1e25

        def gradient_of_objective_function(hyper_parameter):
            gaussian_process.set_parameter_vector(hyper_parameter)
            return -gaussian_process.grad_log_likelihood(trace_around_mean[:,0], quiet=True)

        initial_value = gaussian_process.get_parameter_vector()

        results = scipy.optimize.minimize(objective_function,
                                          initial_value,
                                          jac=gradient_of_objective_function,
                                          method="L-BFGS-B",
                                          bounds = [(0,np.inf),(-9.21,18.42)])
        this_log_lengthscale_square = results.x[1]
        this_lengthscale_square = np.exp(this_log_lengthscale_square)
        this_lengthscale = np.sqrt(this_lengthscale_square)
    else:
        raise ValueError('cannot interpret method' + str(method))

    this_fluctuation_rate = 1.0/this_lengthscale

    return this_fluctuation_rate

def measure_fluctuation_rates_of_traces(traces, method = 'sklearn'):
    '''Convenience function to return the fluctuation rates of multiple traces.
    See measure_fluctuation_rate_of_single_trace for details of the implementation.

    Parameters:
    -----------

    traces : ndarray
        2d array. First column is time. Each further column is a signal trace at the corresponding times in the first columns.

    method : string
        'gpflow' or 'sklearn' are possible

    Returns:
    --------

    fluctuation_rates : ndarray
        1 dimensional array. The length is the number of signal traces that have been passed in traces.
    '''
    fluctuation_rates = np.zeros(traces.shape[1]-1)
    times = traces[:,0]
    for signal_index, signal_trace in enumerate(traces[:,1:].transpose()):
        this_compound_trace = np.vstack((times, signal_trace)).transpose()
        fluctuation_rates[signal_index] = measure_fluctuation_rate_of_single_trace(this_compound_trace, method)
        print('measured fluctuation rate is')
        print(fluctuation_rates[signal_index])

    return fluctuation_rates

def calculate_autocorrelation_from_power_spectrum(power_spectrum):
    '''Calculate autocorrelation

    K(tau) = <x(t)x(t+tau)>

    from the power spectrum by applying an inverse Fourier transform.

    Parameters:
    -----------

    power_spectrum : ndarray
        2D array. First column is frequency, second column is the value of the power spectrum at that frequency.

    Returns:
    --------

    autocorrelation : ndarray
        First column is time (tau in the definition above). Second column is the function value of K.
    '''
    length_of_power_spectrum = power_spectrum.shape[0]
    number_of_time_samples = (length_of_power_spectrum)*2-1
    full_power_spectrum_values = np.zeros(number_of_time_samples)
    full_power_spectrum_values[:number_of_time_samples//2+1] = power_spectrum[:,1]
    full_power_spectrum_values[number_of_time_samples//2+1:] = power_spectrum[1:,1][::-1]
#     inverse_fourier_transform = np.fft.ifft(full_power_spectrum_values, norm = 'ortho' )
    inverse_fourier_transform = np.fft.ifft(full_power_spectrum_values )

    smallest_frequency = power_spectrum[1,0]
    correlation_times = np.linspace(0,1.0/smallest_frequency,number_of_time_samples)

    complex_autocorrelation = np.vstack((correlation_times, inverse_fourier_transform)).transpose()

    autocorrelation = np.real(complex_autocorrelation)
#     autocorrelation[:,1] /= np.sqrt(number_of_time_samples)
    if np.abs(np.sum(np.imag(complex_autocorrelation))) > 1e-7:
        print('WARNING: autocorrelation has complex component')
        print(complex_autocorrelation)

    return autocorrelation

def estimate_fluctuation_rate_of_traces(traces, fix_variance = False):
    '''Estimate the fluctuation rate of traces by (1) calculate the power spectrum,
    (2) transform it into the autocorrelation function K1(t), (3) Fit the function
    K2(t)=v*exp(-alpha*t) to K1(t). The best-fit alpha will be returned as the
    fluctuation rate. The variance v will also be returned.

    Parameters:
    -----------

    traces : ndarray
        2D array. First column is time, Each further trace is one time series of the process that we are
        trying to fit.

    fix_variance : bool
        if True then the variance will not be estimated by optimizing the fit of the autocorrelation function
        but instead just fixed to the variance of the signal

    Returns:
    --------

    fluctuation_rate : float
        fluctuation rate that best fits the autocorellation function of the proposed traces

    variance : float
        variance for best fit
    '''
    power_spectrum, _, _ = calculate_power_spectrum_of_trajectories(traces, normalize = False)

    full_auto_correlation = calculate_autocorrelation_from_power_spectrum(power_spectrum)
    useful_number_of_timepoints = np.int(np.around(full_auto_correlation.shape[0]/2.0))
    useful_auto_correlation = full_auto_correlation[:useful_number_of_timepoints]

    if fix_variance:
        variance = np.var(traces[:,1:])
        def penalty_function(fluctuation_rate):
            mean_squared_errors = np.power(np.abs(useful_auto_correlation[:,1]) -
                                          variance*
                                          np.exp(-fluctuation_rate*useful_auto_correlation[:,0]), 2)
            mean_squared_error = np.sum(mean_squared_errors)
            return mean_squared_error

        results = scipy.optimize.minimize(penalty_function, x0 = [0.001], method = 'Nelder-Mead')
        fluctuation_rate = results.x[0]
    else:
        def penalty_function(fluctuation_rate_and_variance):
            fluctuation_rate = fluctuation_rate_and_variance[0]
            variance = fluctuation_rate_and_variance[1]
            mean_squared_errors = np.power(np.abs(useful_auto_correlation[:,1]) -
                                          variance*
                                          np.exp(-fluctuation_rate*useful_auto_correlation[:,0]), 2)
            mean_squared_error = np.sum(mean_squared_errors)
            return mean_squared_error

        results = scipy.optimize.minimize(penalty_function, x0 = [0.001, 1.0], method = 'Nelder-Mead')

        fluctuation_rate = results.x[0]
        variance = results.x[1]

    return fluctuation_rate, variance

def simulate_downstream_response_at_fluctuation_rate(fluctuation_rate,
                                                     number_of_samples,
                                                     include_upstream_feedback,
                                                     upstream_initial_level = 6.0,
                                                     feedback_delay = 0.0):
    '''Simulate a potential downstream response to a signal with a specific fluctuation rate.
    This function implemetns the network y-|X<-- , i.e y represses X and X self-activates.
                                            \__/

    by implementing the equations

    dX/dt = G_1(y) + G_2(X) - mu*X
    G_1(y) = k_1*1/(1+(y/y0)^n)
    G_2(X) = k_2*1/(1+(X/X0)^(-n)))

    Parameters:
    -----------

    fluctuation_rate : float
        value of the fluctuation rate for which in silico Her6 concentrations y should be generated
        and for which the downstream response of X should be simulated

    number_of_samples : int
        number of samples to generate

    include_upstream_feedback : bool
        whether to include negative feedback of X onto Y, thus changing the motif slightly

    upstream_initial_level : float
        initial level of the upstream signal Y. Currently, this is only implemented for the option
        include_upstream_feedback==True. If this option is false, this value for upstream_initial_level will be
        ignored.

    feedback_delay : float
        delay associated with the feedback - repression of y depends on x at time t-feedback_delay

    Returns:
    --------

    times : ndarray
        float entries, all times at which values of y and x are calculated

    y : ndarray
        float entries. Simulated values of Her6 expression, each column is a different trace

    x : ndarray
        float entries, simulated values downstream response gene expression X, each column is a different trace
    '''
    times = np.linspace(0,30,2000)
    input_variance = 1.7
    delta_t = times[1] - times[0]
    discrete_delay = np.int(np.round(feedback_delay/delta_t))
    if include_upstream_feedback == False:
        ornstein_kernel = ( gp.kernels.ConstantKernel(constant_value=input_variance)*
                            gp.kernels.Matern(nu=0.5, length_scale = 1.0/fluctuation_rate))
        my_gp_regressor_before = gp.GaussianProcessRegressor(kernel=ornstein_kernel )
        y = my_gp_regressor_before.sample_y(times[:, np.newaxis], n_samples = number_of_samples, random_state = None)
        y +=6

        # doing the same thing in a different library
#         ornstein_kernel = GPy.kern.OU(input_dim=1, variance = input_variance, lengthscale = 1/fluctuation_rate)
#         means = np.zeros(len(times))
#         vector of the means
#         covariances = ornstein_kernel.K(times[:, np.newaxis],times[:, np.newaxis])
#         covariance matrix
#         Generate 20 sample path with mean mu and covariance C
#         y = np.random.multivariate_normal(means, covariances)
#         y +=6

        index = 0
        x = np.zeros_like(y)
        for time in times[:-1]:
            this_x = x[index]
            this_y = y[index]
            dx = 0.5/(1+np.power(this_y/7.9,4)) + 10/(1+np.power(this_x/0.5,-4)) - 3*this_x
            index+=1
            x[index] = this_x+dx*delta_t

        return times, y, x
    else:
        x = np.zeros((len(times),number_of_samples))
        y = np.zeros_like(x)
        y[0] = upstream_initial_level
        index = 0
        for time in times[:-1]:
            past_index = index - discrete_delay
            if past_index < 0:
                past_index = 0
            this_x = x[index]
            this_y = y[index]
            past_x = x[past_index]
            index+=1
            dx = 0.5/(1+np.power(this_y/7.9,4)) + 10/(1+np.power(this_x/0.5,-4)) - 3*this_x
            x[index] = this_x+dx*delta_t
            y[index] = np.maximum(this_y+ delta_t*upstream_initial_level*fluctuation_rate/(1+np.power(past_x/2.0,4)) - fluctuation_rate*this_y*delta_t +
                                 np.sqrt(2*delta_t*fluctuation_rate*input_variance)*np.random.randn(number_of_samples), 0.0)
        return times, y, x

def approximate_fluctuation_rate_of_traces_theoretically(traces, sampling_interval = 1,
                                                         sampling_duration = None,
                                                         power_spectrum = None):
    '''Estimate the fluctuation rate of traces by minimising the kullback leibler divergence
    between the measured Gaussian process and the OU process. The best-fit alpha will be returned as the
    fluctuation rate.

    Parameters:
    -----------

    traces : ndarray
        2D array. First column is time, Each further trace is one time series of the process that we are
        trying to fit.

    sampling_interval : int
        sampling_interval-1 values will be skipped between measurements in traces, i.e. traces
        are downsampled by sampling_interval

    sampling_duration : float
        sampling duration that should be used to calculate the fluctuation rate. This value can safely be reduced
        to 12*60 or 24*60 minutes without reducing the accuracy

    power_spectrum : ndarray
        precalculated power spectrum of traces. Has to be the exact power spectrum of the presented traces.
        Added here as a command line argument to avoid recomputation for efficiency.

    Returns:
    --------

    fluctuation_rate : float
        fluctuation rate that best fits the autocorellation function of the proposed traces
    '''
    if power_spectrum is None:
        power_spectrum, _, _ = calculate_power_spectrum_of_trajectories(traces, normalize = False)
    full_auto_correlation = calculate_autocorrelation_from_power_spectrum(power_spectrum)
    if sampling_duration is None:
        sampling_duration = full_auto_correlation[-1,0]

    timestep = full_auto_correlation[1,0] - full_auto_correlation[0,0]
    signal_variance = np.var(traces[:,1:])
    full_auto_correlation[:,1]/=signal_variance

    # only half the number of fourier inverse timepoints are useful at all
    useful_number_of_timepoints = np.int(np.around(full_auto_correlation.shape[0]/2.0))
    sampled_number_of_timepoints = np.int(np.around(sampling_duration/timestep))
    number_of_timepoints_to_use = np.min([useful_number_of_timepoints, sampled_number_of_timepoints])
    useful_auto_correlation = full_auto_correlation[:number_of_timepoints_to_use:sampling_interval]
    all_indices = np.arange(0,useful_auto_correlation.shape[0],1)
    all_distances = scipy.spatial.distance.pdist(all_indices[:,np.newaxis]).astype(np.int)
    index_distance_matrix = scipy.spatial.distance.squareform(all_distances)
    signal_covariance_matrix = useful_auto_correlation[:,1][index_distance_matrix]

    all_times = useful_auto_correlation[:,0]

    # use this function to maximise KL divergence on both the fluctuation rate and variance
    def penalty_function(fluctuation_rate_and_variance):
        fluctuation_rate = fluctuation_rate_and_variance[0]
        variance = fluctuation_rate_and_variance[1]
        correlation_function = np.exp(-fluctuation_rate*all_times)
        new_covariance_matrix = correlation_function[index_distance_matrix]
        _, new_log_det = np.linalg.slogdet(new_covariance_matrix)
        new_inverse_matrix = np.linalg.inv(new_covariance_matrix)
        penalty_value = ( 1.0/variance*np.trace(new_inverse_matrix.dot(signal_covariance_matrix))
                          + new_log_det + new_covariance_matrix.shape[0]*np.log(variance))
        return penalty_value

    # This function only maximises alpha, since we know the variance will converge to the variance of the signal
    def alternative_penalty_function(fluctuation_rate):
        fluctuation_rate = fluctuation_rate[0]
        correlation_function = np.exp(-fluctuation_rate*all_times)
        new_covariance_matrix = correlation_function[index_distance_matrix]
        _, new_log_det = np.linalg.slogdet(new_covariance_matrix)
        new_inverse_matrix = np.linalg.inv(new_covariance_matrix)
        penalty_value = ( np.trace(new_inverse_matrix.dot(signal_covariance_matrix))
                          + new_log_det )
        return penalty_value

    # This is the jacobian of this previous penalty function
    def alternative_penalty_function_jacobian(fluctuation_rate):
        fluctuation_rate = fluctuation_rate[0]
        correlation_function = np.exp(-fluctuation_rate*all_times)
        derivative_correlation_function = -all_times*np.exp(-fluctuation_rate*all_times)
        new_covariance_matrix = correlation_function[index_distance_matrix]
        derivative_covariance_matrix = derivative_correlation_function[index_distance_matrix]
        np.fill_diagonal(derivative_covariance_matrix, 0)
        new_inverse_matrix = np.linalg.inv(new_covariance_matrix)
        d_fluctuation_rate = ( -np.trace(new_inverse_matrix.dot(
                                         derivative_covariance_matrix.dot(
                                         new_inverse_matrix.dot(
                                         signal_covariance_matrix))))
                               +np.trace(new_inverse_matrix.dot(derivative_covariance_matrix)) )
        return np.array([d_fluctuation_rate])

#     results = scipy.optimize.minimize(alternative_penalty_function, x0 = [0.01], jac = alternative_penalty_function_jacobian,
#                                     bounds = [(0.0001,None)], options = {'disp':True})
    results = scipy.optimize.minimize(alternative_penalty_function, x0 = [0.01],bounds = [(0.0001,None)])

    # it's also possible to get the same result by finding the root of the jacobian
    def alternative_root_function(fluctuation_rate):
        return alternative_penalty_function_jacobian([fluctuation_rate])

#     results = scipy.optimize.root_scalar(alternative_root_function_2, x0 = 0.008, bracket = [0.004,0.011])

    fluctuation_rate = results.x[0]

    return fluctuation_rate

def calculate_noise_weight_from_power_spectrum(power_spectrum, frequency_cutoff = 1./40.):
    '''Calculate the weight of the power spectrum that frequencies over 40 min contribute

    Parameters:
    -----------

    power_spectrum : ndarray
        two columns, first column contains frequencies, second column contains powers

    frequency_cutoff : float
        threshold above which frequencies are considered to contribute to noise

    Returns:
    -------

    noise_weight : float
        the area under the power spectrum for frequencies larger than frequency_cutoff
    '''
    first_left_index = np.min(np.where(power_spectrum[:,0]>frequency_cutoff))
    integration_axis = np.hstack(([frequency_cutoff], power_spectrum[first_left_index:,0]))
    power_spectrum_interpolation = scipy.interpolate.interp1d(power_spectrum[:,0], power_spectrum[:,1])
    interpolation_values = power_spectrum_interpolation(integration_axis)
    noise_weight = np.trapz(interpolation_values, integration_axis)
    return noise_weight
