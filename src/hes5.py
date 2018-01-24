import PyDDE
import numpy as np
import scipy.signal
import multiprocessing as mp
# import collections
from numba import jit, autojit


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
        The values 'initial' or 'zero' are supported. The default 'initial' will assume that protein and 
        mRNA numbers were constant at the values of the initial condition for all negative times.
        If 'zero' is chosen, then the protein and mRNA numbers are assumed to be 0 at negative times. 
        If 'no_negative' is chosen, no assumptions are made for negative times, and transcription
        is blocked until transcription_delay is passed.

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
    deterministic runs of the model. This gradient function assumes
    that for t<0 the protein has the same value as the initial condition.
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
        The value 0.0 corresponds to the option 'initial', whereas 1.0 corresponds to 'zero'.
    
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
    and return their average distance from each other. Will also return the relative amplitude
    defined as the signal difference of each peak to the previous lowest dip relative to the mean
    of the signal; and the variation of that amplitude
    
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
                                    equilibration_time = 0.0):
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
        at the beginning of the trajectory in order to get rid of any overshoots, for example

    Returns
    -------
    
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
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
    delayed_transcription_times = []
    
    # This following index is to keep track at which index of the trace entries
    # we currently are (see definition of trace above). This is necessary since
    # the SSA will calculate reactions at random times and we need to transform
    # calculated reaction times to the samplin time points 
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

    return trace

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
                                    equilibration_time = 0.0):
    '''Generate multiple stochastic traces the Hes5 modelby using
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
        at the beginning of the trajectory in order to get rid of any overshoots, for example

    Returns
    -------
    
    mRNA_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time, 
        each further column is one trace of mRNA copy numbers 

    protein_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time, 
        each further column is one trace of protein copy numbers 
    '''

    pool_of_processes = mp.Pool(processes=4)
    arguments = [ (duration, repression_threshold, hill_coefficient,
                  mRNA_degradation_rate, protein_degradation_rate, 
                  basal_transcription_rate, translation_rate,
                  transcription_delay, initial_mRNA, initial_protein,
                  equilibration_time) ]*number_of_trajectories
    process_results = [ pool_of_processes.apply_async(generate_stochastic_trajectory, args=x)
                        for x in arguments ]
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

    for trajectory_index, this_trace in enumerate(list_of_traces): # range argument because the first 
                                                                 # trajectory has already been created
        # offset one index for time column
        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
 
    return mRNA_trajectories, protein_trajectories

#     first_trace = generate_stochastic_trajectory( duration, 
#                                                   repression_threshold, 
#                                                   hill_coefficient, 
#                                                   mRNA_degradation_rate, 
#                                                   protein_degradation_rate, 
#                                                   basal_transcription_rate, 
#                                                   translation_rate, 
#                                                   transcription_delay, 
#                                                   initial_mRNA, 
#                                                   initial_protein,
#                                                   equilibration_time)
#     

#     sample_times = first_trace[:,0]
    
#     mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
#     protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    
#     mRNA_trajectories[:,0] = sample_times
#     mRNA_trajectories[:,1] = first_trace[:,1]
#     protein_trajectories[:,0] = sample_times
#     protein_trajectories[:,1] = first_trace[:,2]

#     for trajectory_index in range(1,number_of_trajectories): # range argument because the first 
                                                             # trajectory has already been created
#         this_trace = generate_stochastic_trajectory( duration, 
#                                                   repression_threshold, 
#                                                   hill_coefficient, 
#                                                   mRNA_degradation_rate, 
#                                                   protein_degradation_rate, 
#                                                   basal_transcription_rate, 
#                                                   translation_rate, 
#                                                   transcription_delay, 
#                                                   initial_mRNA, 
#                                                   initial_protein)
# 
#         # offset one index for time column
#         mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
#         protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
    
#     return mRNA_trajectories, protein_trajectories
      
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
#     base_propensity = np.sum(propensities)
#     base_propensity = propensities[0] + propensities[1] + propensities[2] + propensities[3]
    scaled_random_number = random_number*base_propensity
    propensity_sum = 0.0
    for reaction_index, propensity in enumerate(propensities):
        if scaled_random_number < propensity_sum + propensity:
            return reaction_index
        else:
            propensity_sum += propensity
    
    ##Make sure we never exit the for loop:
    raise(RuntimeError("This line should never be reached."))
        
def calculate_power_spectrum_of_trajectories(trajectories):
    '''Calculate the power spectrum, coherence, and period, of a set
    of trajectories. Works by applying discrete fourier transforms to the mean
    of the trajectories. We define the power spectrum as the square of the
    absolute of the fourier transform, and we define the coherence as in 
    Phillips et al (eLife, 2016): the relative area of the power spectrum
    occopied by a 20% frequency band around the maximum frequency.
    The maximum frequency corresponds to the inverse of the period.
    The returned power spectum excludes the frequency 0 and thus neglects
    the mean of the signal.
    
    Parameters:
    ---------- 
    
    trajectories : ndarray
        2D array. First column is time, each further column contains one realisation
        of the signal that is aimed to be analysed.
    
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
        
    mean_trajectory = np.vstack((trajectories[:,0], np.mean(trajectories[:,1:], axis = 1))).transpose()
    
    power_spectrum, coherence, period = calculate_power_spectrum_of_trajectory(mean_trajectory)

    return power_spectrum, coherence, period
    
def calculate_power_spectrum_of_trajectory(trajectory):
    '''Calculate the power spectrum, coherence, and period, of a trajectory. 
    Works by applying discrete fourier transformation. We define the power spectrum as the square of the
    absolute of the fourier transform, and we define the coherence as in 
    Phillips et al (eLife, 2016): the relative area of the power spectrum
    occopied by a 20% frequency band around the maximum frequency.
    The maximum frequency corresponds to the inverse of the period.
    The returned power spectum excludes the frequency 0 and thus neglects
    the mean of the signal. The power spectrum is normalised so that all entries add up to one.
    
    Parameters:
    ---------- 
    
    trajectories : ndarray
        2D array. First column is time, second column contains the signal that is aimed to be analysed.
    
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
    power_spectrum_without_frequencies = np.power(np.abs(fourier_transform[1:(number_of_data_points/2)]),2)
    power_spectrum_without_frequencies/= np.sum(power_spectrum_without_frequencies)
    power_spectrum = np.vstack((fourier_frequencies, power_spectrum_without_frequencies)).transpose()


    # Calculate coherence:
    max_index = np.argmax(power_spectrum_without_frequencies)
    coherence_boundary_left = int(np.round(max_index - max_index*0.1))
    coherence_boundary_right = int(np.round(max_index + max_index*0.1))
    coherence_area = np.trapz(power_spectrum_without_frequencies[coherence_boundary_left:(coherence_boundary_right+1)])
    full_area = np.trapz(power_spectrum_without_frequencies)
    coherence = coherence_area/full_area
    
    # calculate period: 
    max_power_frequency = fourier_frequencies[max_index]
    period = 1./max_power_frequency

    return power_spectrum, coherence, period
 