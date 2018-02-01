import PyDDE
import numpy as np
import scipy.signal
import multiprocessing as mp
# import collections
from numba import jit, autojit
from numpy import ndarray
import os
import seaborn.apionly as sns
import pandas as pd

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
                                    transcription_schedule = np.array([-1.0])):
    '''Generate one trace of the Hes5 model. This function implements a stochastic version of
    the model model in Monk, Current Biology (2003). It applies the rejection method described
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
                                    transcription_schedule)

    return trace

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
                                    transcription_schedule = np.array([-1.0])):
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

    Returns
    -------
    
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
        
    remaining_transcription_times : list
        list entries are floats. Each entry corresponds to the time of one transcription event.
    '''
    total_time = duration + equilibration_time
    sample_times = np.linspace(equilibration_time, total_time, 100)
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
                propensities[0] = basal_transcription_rate/(1.0+
                                                            np.power(current_protein/repression_threshold, 
                                                                     hill_coefficient))
                propensities[2] = current_protein*protein_degradation_rate
            elif reaction_index == 2: #protein degradation
                current_protein -=1
                propensities[0] = basal_transcription_rate/(1.0+
                                                            np.power(current_protein/repression_threshold, 
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
                                    number_of_cpus = 3):
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
                  equilibration_time, transcription_schedule) ]*number_of_trajectories
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
            this_power_spectrum,_,_ = calculate_power_spectrum_of_trajectory(this_compound_trajectory)
            all_power_spectra[:,trajectory_index] = this_power_spectrum[:,1]
            trajectory_index += 1
        mean_power_spectrum_without_frequencies = np.mean(all_power_spectra, axis = 1)
        mean_power_spectrum_without_frequencies /= np.sum(mean_power_spectrum_without_frequencies)
        power_spectrum = np.vstack((frequency_values, mean_power_spectrum_without_frequencies)).transpose()
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
    if normalize:
        power_spectrum_without_frequencies/= np.sum(power_spectrum_without_frequencies)
    power_spectrum = np.vstack((fourier_frequencies, power_spectrum_without_frequencies)).transpose()

    coherence, period = calculate_coherence_and_period_of_power_spectrum(power_spectrum)

    return power_spectrum, coherence, period
 
def generate_posterior_samples( total_number_of_samples, 
                                acceptance_ratio,
                                number_of_traces_per_sample = 10,
                                number_of_cpus = 3,
                                saving_path = ''):
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

    saving_path : string
        where to save, specified without file ending
        
    Returns
    -------
    
    posterior_samples : ndarray
        samples from the posterior distribution, are repression_threshold, hill_coefficient,
                                    mRNA_degradation_rate, protein_degradation_rate, basal_transcription_rate,
                                    translation_rate, transcription_delay
    '''
    # first: keep degradation rates infer translation, transcription, repression threshold,
    # and time delay
    prior_samples = generate_prior_samples( total_number_of_samples )

    # collect summary_statistics_at_parameters
    model_results = calculate_summary_statistics_at_parameters( prior_samples, number_of_traces_per_sample, number_of_cpus )

    if saving_path == '':
        saving_path = os.path.join(os.path.dirname(__file__),'..','test','output','sampling_results')
        
    np.save(saving_path + '.npy', model_results)
    np.save(saving_path + '_parameters.npy', prior_samples)

    # calculate distances to data
    distance_table = calculate_distances_to_data(model_results)
    
    posterior_samples = select_posterior_samples( prior_samples, 
                                                  distance_table, 
                                                  acceptance_ratio )
    
    return posterior_samples

def plot_posterior_distributions( posterior_samples ):
    '''Plot the posterior samples in a pair plot. Only works if there are
    more than four samples present
    
    Parameters
    ----------
    
    posterior samples : np.array
        The samples from which the pairplot should be generated.
        Each row contains a parameter
    
    Returns
    -------
    
    paiplot : matplotlib figure handle
       The handle for the pairplot on which the use can call 'save'
    '''
    data_frame = pd.DataFrame( data = posterior_samples,
                               columns= ['transcription_rate', 
                                         'translation_rate', 
                                         'repression_threshold', 
                                         'transcription_delay'])
    pairplot = sns.PairGrid(data_frame)
    pairplot.map_diag(sns.kdeplot)
    pairplot.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=10)

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
    
def calculate_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 10,
                                               number_of_cpus = 3):
    '''Calculate the mean, relative standard deviation, period, and coherence
    of protein traces at each parameter point in parameter_values. 
    Will assume the arguments to be of the order described in
    generate_prior_samples.
    
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
    
def generate_prior_samples(number_of_samples):
    '''Sample from the prior. Provides samples of the form 
    (basal_transcription_rate, translation_rate, repression_threshold, transcription_delay)
    
    Parameters
    ----------
    
    number_of_samples : int
        number of samples to draw from the prior distribution
        
    Returns
    -------
    
    prior_samples : ndarray
        array of shape (number_of_samples,4) with columns corresponding to
        (basal_transcription_rate, translation_rate, repression_threshold, time_delay)
    '''
    #initialise as random numbers between (0,1)
    prior_samples = np.random.rand(number_of_samples, 4)
    #now scale and shift each entry as appropriate
    #basal_transcription_rate in range (0,100)
    prior_samples[:,0] *= 100
    #translation rate in range (0,100)
    prior_samples[:,1] *= 100
    #repression_threshold in range (0,100000)
    prior_samples[:,2] *= 100000
    #transcriptione_delay in range (5,40)
    prior_samples[:,3] *= 35
    prior_samples[:,3] += 5
    
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
    coherence_boundary_left = int(np.round(max_index - max_index*0.1))
    coherence_boundary_right = int(np.round(max_index + max_index*0.1))
    coherence_area = np.trapz(power_spectrum[coherence_boundary_left:(coherence_boundary_right+1),1])
    full_area = np.trapz(power_spectrum[:,1])
    coherence = coherence_area/full_area
    
    # calculate period: 
    max_power_frequency = power_spectrum[max_index,0]
    period = 1./max_power_frequency
    
    return coherence, period
    
