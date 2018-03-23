import PyDDE
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.interpolate
import multiprocessing as mp
# import collections
from numba import jit, autojit
from numpy import ndarray
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
from fnmatch import translate

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
                                       for_negative_times = 'initial'):
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

    Returns
    -------
    
    trace : ndarray
        2 dimenstional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
    
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

    hes5_dde.dde(y=initial_condition, times=np.arange(0.0, duration, 0.1), 
                 func=hes5_ddegrad, parms=parameters, 
                 tol=0.000005, dt=0.01, hbsize=10000, nlag=1, ssc=[0.0, 0.0]) 
                 #hbsize is buffer size, I believe this is how many values in the past are stored
                 #nlag is the number of delay variables (tau_1, tau2, ... taun_lag)
                 #ssc means "statescale" and would somehow only matter for values close to 0

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

@autojit(nopython=True)
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
#                                     explicit typing necessary for autojit
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

def calculate_theoretical_power_spectrum_at_parameter_point(basal_transcription_rate = 1.0,
                                                            translation_rate = 1.0,
                                                            repression_threshold = 100,
                                                            transcription_delay = 18.5,
                                                            mRNA_degradation_rate = 0.03,
                                                            protein_degradation_rate = 0.03,
                                                            hill_coefficient = 5
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

    power_spectrum_values = ( translation_rate*translation_rate*
                       ( basal_transcription_rate * steady_state_hill_function_value +
                         mRNA_degradation_rate*steady_state_mrna) 
                      +
                       ( np.power(pi_frequencies,2) + mRNA_degradation_rate*mRNA_degradation_rate)*
                         ( translation_rate*steady_state_mrna + protein_degradation_rate*steady_state_protein)
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
    integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
    power_spectrum[:,1] /= integral

    return power_spectrum

@autojit(nopython=True)
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
                                    #explicit typing necessary for autojit
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
    # this requires a bit of faff for autojit to compile
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
                                    number_of_cpus = 3,
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

    pool_of_processes = mp.Pool(processes = number_of_cpus)
    arguments = [ (duration, repression_threshold, hill_coefficient,
                  mRNA_degradation_rate, protein_degradation_rate, 
                  basal_transcription_rate, translation_rate,
                  transcription_delay, initial_mRNA, initial_protein,
                  equilibration_time, transcription_schedule, sampling_timestep) ]*number_of_trajectories
#                   equilibration_time, transcription_schedule) ]*number_of_trajectories
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
 
    return mRNA_trajectories, protein_trajectories

@autojit(nopython = True)
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
        
def calculate_power_spectrum_of_trajectories(trajectories, method = 'standard'):
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
        first_power_spectrum,_,_ = calculate_power_spectrum_of_trajectory(first_compound_trajectory)
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
#         mean_power_spectrum_without_frequencies /= np.sum(mean_power_spectrum_without_frequencies)
        power_spectrum = np.vstack((frequency_values, mean_power_spectrum_without_frequencies)).transpose()
        power_integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        power_spectrum[:,1]/=power_integral
        coherence, period = calculate_coherence_and_period_of_power_spectrum(power_spectrum)
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
    fourier_transform = np.fft.fft(trajectory[:,1])/number_of_data_points
    fourier_frequencies = np.arange( 0,number_of_data_points/(2*interval_length), 
                                                     1.0/(interval_length) )[1:]
    power_spectrum_without_frequencies = np.power(np.abs(fourier_transform[1:(number_of_data_points//2)]),2)
    
    # this should really be a decision about the even/oddness of number of datapoints
    try:
        power_spectrum = np.vstack((fourier_frequencies, power_spectrum_without_frequencies)).transpose()
    except ValueError:
        power_spectrum = np.vstack((fourier_frequencies[:-1], power_spectrum_without_frequencies)).transpose()

    if normalize:
#         power_spectrum_without_frequencies /= np.sum(power_spectrum_without_frequencies)
        power_integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        power_spectrum[:,1]/=power_integral

    coherence, period = calculate_coherence_and_period_of_power_spectrum(power_spectrum)

    return power_spectrum, coherence, period
 
def generate_posterior_samples( total_number_of_samples, 
                                acceptance_ratio,
                                number_of_traces_per_sample = 10,
                                number_of_cpus = 3,
                                saving_name = 'sampling_results',
                                prior_bounds = {'basal_transcription_rate' : (0,100),
                                                'translation_rate' : (0,200),
                                                'repression_threshold' : (0,100000),
                                                'time_delay' : (5,40),
                                                'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
                                                'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)},
                                prior_dimension = 'reduced',
                                use_langevin = True,
                                logarithmic = True ):
    '''Draw samples from the posterior using normal ABC. Posterior is calculated
    using ABC and the summary statistics mean and relative standard deviation.
    Also saves all sampled model results.
    
    Parameters
    ----------
    
    number_of_samples : int
        number of samples to be generated from the posterior
       
    acceptance_ratio : float
        ratio of the posterior samples that should be accepted.
        
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
        'reduced', 'hill' or 'full' are possible options. If 'full', then the mRNA and protein degradation rates
        will be inferred in addition to other model parameters, excluding the Hill coefficient. If 'hill',
        then all parameters exclucing the mRNA and protein degradation rates will be inferred.
        
    use_langevin : bool
        if True then the results will be generated using the langevin equation rather than the full gillespie algorithm.
        
    logarithmic : bool
        if True then logarithmic priors will be used on the translation and transcription rate constants
        
    Returns
    -------
    
    posterior_samples : ndarray
        samples from the posterior distribution, are repression_threshold, hill_coefficient,
                                    mRNA_degradation_rate, protein_degradation_rate, basal_transcription_rate,
                                    translation_rate, transcription_delay
    '''
    # first: keep degradation rates infer translation, transcription, repression threshold,
    # and time delay
    prior_samples = generate_prior_samples( total_number_of_samples, use_langevin,
                                            prior_bounds, prior_dimension, logarithmic )

    # collect summary_statistics_at_parameters
    model_results = calculate_summary_statistics_at_parameters( prior_samples, 
                                                                number_of_traces_per_sample, 
                                                                number_of_cpus,
                                                                use_langevin )

    saving_path = os.path.join(os.path.dirname(__file__),'..','test','output',saving_name)
        
    np.save(saving_path + '.npy', model_results)
    np.save(saving_path + '_parameters.npy', prior_samples)

    # calculate distances to data
    distance_table = calculate_distances_to_data(model_results)
    
    posterior_samples = select_posterior_samples( prior_samples, 
                                                  distance_table, 
                                                  acceptance_ratio )
    
    return posterior_samples

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
    
    posterior_samples[:,2]/=10000

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
                                             'mRNA degradation',
                                             'Protein degradation'])
    elif posterior_samples.shape[1] == 7:
        data_frame = pd.DataFrame( data = posterior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient'
                                             'mRNA degradation',
                                             'Protein degradation'])
    else:
        raise ValueError("Cannot plot posterior samples of this dimension.")
        
    pairplot = sns.PairGrid(data_frame)
    pairplot.map_diag(sns.distplot, kde = False, rug = True )
    pairplot.map_offdiag(sns.regplot, scatter_kws = {'alpha' : 0.4}, fit_reg=False) 
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
        transcription_histogram,_ = np.histogram(data_frame['Transcription rate'], 
                                                 bins = transcription_rate_bins)
        sns.distplot(data_frame['Transcription rate'],
                     kde = False,
                     rug = True,
                     bins = transcription_rate_bins)
    #                  ax = pairplot.diag_axes[0])
#         pairplot.diag_axes[0].set_ylim(0,np.max(transcription_histogram)*1.2)
        plt.gca().set_xlim(0.5,100)
    
        plt.sca(pairplot.diag_axes[1])
        sns.distplot(data_frame['Translation rate'],
                     kde = False,
                     rug = True,
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

    pairplot.axes[-1,2].set_xlim(0,10)
    pairplot.axes[-1,3].set_xlim(5,40)

    if posterior_samples.shape[1] == 6:
        pairplot.axes[-1,4].locator_params(axis = 'x', nbins = 5)
        pairplot.axes[-1,5].locator_params(axis = 'x', nbins = 5)
        pairplot.axes[-1,4].set_xlim(0,0.04)
        pairplot.axes[-1,5].set_xlim(0,0.04)

#     pairplot = sns.PairGrid(data_frame)
#     pairplot.map_diag(sns.kdeplot)
#     pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)

    sns.reset_orig()

    return pairplot

def select_posterior_samples(prior_samples, distance_table, acceptance_ratio):
    '''Collect the parameter values of all prior_samples whose distance_table entries are 
    within the acceptance_ratio closest samples.
    
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
                                                            number_of_cpus = 3):
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
    _,this_full_coherence, this_full_period = calculate_power_spectrum_of_trajectories(these_full_protein_traces)

    these_full_mrna_traces = np.zeros_like(these_mrna_traces_1)
    these_full_mrna_traces[:,0] = these_mrna_traces_1[:,0]
    these_full_mrna_traces[:,1:] = these_mrna_traces_1[:,1:] + these_mrna_traces_2[:,1:]

    this_full_mean = np.mean(these_full_protein_traces[:,1:])
    this_full_std = np.std(these_full_protein_traces[:,1:])/this_full_mean
    this_full_mean_mRNA = np.mean(these_full_mrna_traces[:,1:])

    _,this_allele_coherence_1, this_allele_period_1 = calculate_power_spectrum_of_trajectories(these_protein_traces_1)
    this_allele_mean_1 = np.mean(these_protein_traces_1[:,1:])
    this_allele_std_1 = np.std(these_protein_traces_1[:,1:])/this_allele_mean_1
    this_allele_mean_mRNA_1 = np.mean(these_mrna_traces_1[:,1:])
    
    _,this_allele_coherence_2, this_allele_period_2 = calculate_power_spectrum_of_trajectories(these_protein_traces_2)
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
                                               number_of_cpus = 3, use_langevin = True):
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

    use_langevin : bool
        if True then the results will be generated using the langevin equation rather than the full gillespie algorithm.

    Returns
    -------
    
    summary_statistics : ndarray
        each row contains the summary statistics (mean, std, period, coherence) for the corresponding
        parameter set in parameter_values
    '''
    if use_langevin:
        summary_statistics = calculate_langevin_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample,
                                                            number_of_cpus)
    else:
        if parameter_values.shape[1] != 4:
            raise ValueError("Gillespie inference on full parameter space is not implemented.")
        summary_statistics = calculate_gillespie_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample,
                                                            number_of_cpus)

    return summary_statistics
    
def calculate_gillespie_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 10,
                                                         number_of_cpus = 3):
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
    summary_statistics = np.zeros((parameter_values.shape[0], 4))
    for parameter_index, parameter_value in enumerate(parameter_values):
        these_mRNA_traces, these_protein_traces = generate_multiple_trajectories( 
                                                        number_of_trajectories = number_of_traces_per_sample, 
                                                        duration = 1500,
                                                        basal_transcription_rate = parameter_value[0],
                                                        translation_rate = parameter_value[1], 
                                                        repression_threshold = parameter_value[2], 
                                                        transcription_delay = parameter_value[3],
                                                        mRNA_degradation_rate = np.log(2)/30, 
                                                        protein_degradation_rate = np.log(2)/90, 
                                                        initial_mRNA = 0,
                                                        initial_protein = parameter_value[2],
                                                        equilibration_time = 1000,
                                                        number_of_cpus = number_of_cpus)
        _,this_coherence, this_period = calculate_power_spectrum_of_trajectories(these_protein_traces)
        this_mean = np.mean(these_protein_traces[:,1:])
        this_std = np.std(these_protein_traces[:,1:])/this_mean
        summary_statistics[parameter_index,0] = this_mean
        summary_statistics[parameter_index,1] = this_std
        summary_statistics[parameter_index,2] = this_period
        summary_statistics[parameter_index,3] = this_coherence

    return summary_statistics
 
def calculate_langevin_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 100,
                                                         number_of_cpus = 3):
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

    Returns
    -------
    
    summary_statistics : ndarray
        each row contains the summary statistics (mean, std, period, coherence, mean_mrna) for the corresponding
        parameter set in parameter_values
    '''
    summary_statistics = np.zeros((parameter_values.shape[0], 5))

    pool_of_processes = mp.Pool(processes = number_of_cpus)

    process_results = [ pool_of_processes.apply_async(calculate_langevin_summary_statistics_at_parameter_point, 
                                                      args=(parameter_value, number_of_traces_per_sample))
                        for parameter_value in parameter_values ]

    ## Let the pool know that these are all so that the pool will exit afterwards
    # this is necessary to prevent memory overflows.
    pool_of_processes.close()

    for parameter_index, process_result in enumerate(process_results):
        these_summary_statistics = process_result.get()
        summary_statistics[ parameter_index ] = these_summary_statistics

    return summary_statistics

def calculate_langevin_summary_statistics_at_parameter_point(parameter_value, number_of_traces = 100):
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
        these_mrna_traces, these_protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
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
        these_mrna_traces, these_protein_traces = generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
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
                                                                                           1000)
    else: 
        raise ValueError("This dimension of the prior sample is not recognised.")
 
    summary_statistics = np.zeros(5)
    _,this_coherence, this_period = calculate_power_spectrum_of_trajectories(these_protein_traces)
    this_mean = np.mean(these_protein_traces[:,1:])
    this_std = np.std(these_protein_traces[:,1:])/this_mean
    this_mean_mRNA = np.mean(these_mrna_traces[:,1:])
    summary_statistics[0] = this_mean
    summary_statistics[1] = this_std
    summary_statistics[2] = this_period
    summary_statistics[3] = this_coherence
    summary_statistics[4] = this_mean_mRNA
    
    return summary_statistics

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
        'reduced' or 'full', or 'hill' are possible options. If 'full', then the mRNA and protein degradation rates
        will be inferred in addition to other model parameters.
        
    logarithmic : bool
        if True then logarithmic priors will be assumed on the translation and transcription rates.

    Returns
    -------
    
    prior_samples : ndarray
        array of shape (number_of_samples,4) with columns corresponding to
        (basal_transcription_rate, translation_rate, repression_threshold, time_delay)
    '''
    index_to_parameter_name_lookup = {0: 'basal_transcription_rate',
                                      1: 'translation_rate',
                                      2: 'repression_threshold',
                                      3: 'time_delay',
                                      4: 'hill_coefficient',
                                      5: 'mRNA_degradation_rate',
                                      6: 'protein_degradation_rate'}
    
    standard_prior_bounds = {'basal_transcription_rate' : (0.5,100),
                             'translation_rate' : (1,200),
                             'repression_threshold' : (0,100000),
                             'time_delay' : (5,40),
                             'hill_coefficient' : (2,7),
                             'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
                             'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}

    # depending on the function argument prior_dimension we create differently sized prior tables
    if prior_dimension == 'full':
        number_of_dimensions = 7
    elif prior_dimension == 'reduced':
        number_of_dimensions = 4
    elif prior_dimension == 'hill':
        number_of_dimensions = 5
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
                                                   'basal_transcription_rate']:
            prior_samples[:,parameter_index] = these_parameter_bounds[0]*np.power(
                                               these_parameter_bounds[1]/float(these_parameter_bounds[0]),
                                               prior_samples[:,parameter_index])
        else:
            prior_samples[:,parameter_index] *= these_parameter_bounds[1] - these_parameter_bounds[0]
            prior_samples[:,parameter_index] += these_parameter_bounds[0]
        if this_parameter_name == 'time_delay' and use_langevin:
            prior_samples[:,parameter_index] = np.around(prior_samples[:,parameter_index])

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
    last_right_index = np.min(np.where(power_spectrum[:,0]>coherence_boundary_right))
    integration_axis = np.hstack(([coherence_boundary_left], 
                                  power_spectrum[first_left_index:last_right_index,0],
                                  [coherence_boundary_right]))

    interpolation_values = power_spectrum_interpolation(integration_axis)
    coherence_area = np.trapz(interpolation_values, integration_axis)
    full_area = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
    coherence = coherence_area/full_area
    
    # calculate period: 
    period = 1./max_power_frequency
    
    return coherence, period
    
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

@autojit(nopython = True)
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
 
@autojit(nopython = True)
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
                                    equilibration_time = 0.0
                                    ):
    '''Generate one trace of the protein-autorepression model using a langevin approximation. 
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
    repression_threshold = float(repression_threshold)

    mRNA_degradation_rate_per_timestep = mRNA_degradation_rate*delta_t
    protein_degradation_rate_per_timestep = protein_degradation_rate*delta_t
    basal_transcription_rate_per_timestep = basal_transcription_rate*delta_t
    translation_rate_per_timestep = translation_rate*delta_t
    delay_index_count = int(round(transcription_delay/delta_t))
    
    for time_index, sample_time in enumerate(sample_times[1:]):
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

## autojitting this function does not seem to improve runtimes further
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
                                               equilibration_time)

        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
 
    return mRNA_trajectories, protein_trajectories

def conduct_all_parameter_sweeps_at_parameters(parameter_samples,
                                               number_of_sweep_values = 20,
                                               number_of_traces_per_parameter = 200,
                                               relative = False):
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
        If true x values will not be parameter values but percentage values in changes from 10% to
        200%
        
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
        print 'sweeping ' + parameter_name
        these_results = conduct_parameter_sweep_at_parameters(parameter_name,
                                                              parameter_samples,
                                                              number_of_sweep_values,
                                                              number_of_traces_per_parameter,
                                                              relative)
        print 'done sweeping ' + parameter_name

        sweep_results[parameter_name] = these_results   
        
    return sweep_results
    
def conduct_parameter_sweep_at_parameters(parameter_name,
                                          parameter_samples,
                                          number_of_sweep_values = 20,
                                          number_of_traces_per_parameter = 200,
                                          relative = False):
    '''Conduct a parameter sweep of the parameter_name parameter at each of the parameter points in
    parameter_samples. The parameter_samples are four-dimensional, as produced, for example, by 
    generate_prior_samples() with the 'reduced' dimension. At each parameter point the function
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
        If true x values will not be parameter values but percentage values in changes from 10% to
        200%

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
    parameter_indices_and_ranges['protein_degradation_rate'] = (6,(np.log(2)/500,np.log(2)/15))

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
            for parameter_proportion in np.linspace(0.1,2.0,number_of_sweep_values):
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
                                                                        number_of_cpus = 3, 
                                                                        use_langevin = True)
    
    # unpack and wrap the results in the output format
    sweep_results = np.zeros((parameter_samples.shape[0], number_of_sweep_values, 5))
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
                sweep_results[sample_index,sweep_value_index,1:] = these_summary_statistics[:-1]
                sweep_value_index += 1
                parameter_sample_index += 1
    else:
        for sample_index, sample in enumerate(parameter_samples):
            proportion_index = 0
            for parameter_proportion in np.linspace(0.1,2.0,number_of_sweep_values):
                these_summary_statistics = all_summary_statistics[parameter_sample_index]
                # the first entry gets the degradation rate
                sweep_results[sample_index,proportion_index,0] = parameter_proportion
                # the remaining entries get the summary statistics. We discard the last summary statistic, 
                # which is the mean mRNA
                sweep_results[sample_index,proportion_index,1:] = these_summary_statistics[:-1]
                proportion_index += 1
                parameter_sample_index += 1
    # repack results into output array
 
    return sweep_results
    
