import PyDDE
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.interpolate
import multiprocessing as mp
# import collections
from numba import jit, autojit
from numpy import ndarray, number
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
import socket
import jitcdde
from numpy.distutils.log import good
import hes5

domain_name = socket.getfqdn()
if domain_name == 'jochen-ThinkPad-S1-Yoga-12':
    number_of_available_cores = 2
else:
#     number_of_available_cores = 1
    number_of_available_cores = mp.cpu_count()

@autojit(nopython=True)
def generate_switching_trajectory( duration = 720, 
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
                                    sampling_timestep = 1.0,
                                    switching_rate = 1.0,
                                    initial_environment_on = True):
    '''Generate one trace of the Hes model with transcriptional switching. This function implements a stochastic version of
    the model in Monk, Current Biology (2003) where the hill function emerges from transcriptional switching. 
    It applies the Gillespie-rejection method described in Cai et al, J. Chem. Phys. (2007) as Algorithm 2. 
    This method is an exact method to calculate the temporal evolution of stochastic reaction systems with delay.     

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
        
    switching_rate : float
        rate of environmental switching

    initial_environment_on : bool
        True if the environment is in on state at the beginning
        
    Returns
    -------
    
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number, fourth column is environmental state
    '''
    
    total_time = duration + equilibration_time
    sample_times = np.arange(equilibration_time, total_time, sampling_timestep)
    trace = np.zeros((len(sample_times), 4))
    trace[:,0] = sample_times
    
    repression_threshold = float(repression_threshold)
    # inital_condition
    current_mRNA = initial_mRNA
    current_protein =  initial_protein
    current_environment_on = initial_environment_on
#     trace[0,1] = initial_mRNA
#     trace[0,2] = initial_protein
    switching_propensity = switching_rate
    if current_environment_on:
        switching_propensity = switching_rate*np.power(current_protein/repression_threshold, hill_coefficient)
    else:
        switching_propensity = switching_rate

    propensities = np.array([ basal_transcription_rate*current_environment_on, # transcription
                              initial_mRNA*translation_rate, # translation 
                              initial_protein*protein_degradation_rate, # Protein degradation
                              initial_mRNA*mRNA_degradation_rate, # mRNA degradation
                              switching_propensity] ) # environmental switching
   
    # set up the gillespie algorithm: We first
    # need a list where we store any delayed reaction times
#     delayed_transcription_times = collections.deque()
    # this requires a bit of faff for autojit to compile
    delayed_transcription_times = [-1.0]
    delayed_transcription_times.pop(0)
    #this is now an empty list
    
    # This following index is to keep track at which index of the trace entries
    # we currently are (see definition of trace above). This is necessary since
    # the SSA will calculate reactions at random times and we need to transform
    # calculated reaction times to the sampling time points 
    sampling_index = 0

    time = 0.0
    while time < sample_times[-1]:
        base_propensity = (propensities[0] + 
                           propensities[1] + 
                           propensities[2] + 
                           propensities[3] +
                           propensities[4])

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
            # identify which of the five reactions occured
            reaction_index = identify_reaction(second_random_number, base_propensity, propensities)
            # execute reaction
            if reaction_index == 0:  #transcription initiation
                delayed_transcription_times.append(time + time_to_next_reaction + transcription_delay)
                # propensities don't change
            elif reaction_index == 1: #protein translation
                current_protein += 1
                if current_environment_on:
                    propensities[4] = switching_rate*np.power(current_protein/repression_threshold, 
                                                                     hill_coefficient)
                propensities[2] = current_protein*protein_degradation_rate
            elif reaction_index == 2: #protein degradation
                current_protein -=1
                if current_environment_on:
                    propensities[4] = switching_rate*np.power(current_protein/repression_threshold, 
                                                                     hill_coefficient)
                propensities[2] = current_protein*protein_degradation_rate
            elif reaction_index == 3: #mRNA degradation
                current_mRNA -= 1
                propensities[1] = current_mRNA*translation_rate
                propensities[3] = current_mRNA*mRNA_degradation_rate
            elif reaction_index == 4: #environmental_switching
                current_environment_on = not current_environment_on #toggling the bool
                if current_environment_on:
                    propensities[4] = switching_rate*np.power(current_protein/repression_threshold, 
                                                                   hill_coefficient)
                else:
                    propensities[4] = switching_rate
                propensities[0] = basal_transcription_rate*current_environment_on
            else:
                raise(RuntimeError("Couldn't identify reaction. This should not happen."))
        
        # update trace for all entries until the current time
        while ( sampling_index < len(sample_times) and
                time > trace[ sampling_index, 0 ] ):
            trace[ sampling_index, 1 ] = current_mRNA
            trace[ sampling_index, 2 ] = current_protein
            trace[ sampling_index, 3 ] = current_environment_on
            sampling_index += 1

    trace[:,0] -= equilibration_time

#     if return_transcription_times: 
#        return trace, delayed_transcription_times
#     else:
    return trace

def generate_multiple_switching_trajectories( number_of_trajectories = 10,
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
                                    switching_rate = 1.0,
                                    initial_environment_on = True,
                                    number_of_cpus = number_of_available_cores,
                                    sampling_timestep = 1.0):
    '''Generate multiple stochastic traces the switching Hes model by using
       generate_switching_trajectory. The trajectories are simulated in parallel, i.e.
       individually on different processes.
     
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
         
    switching_rate : float
        rate of environmental switching
 
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
 
    pool_of_processes = mp.Pool(processes = number_of_cpus)
    arguments = [ (duration, repression_threshold, hill_coefficient,
                  mRNA_degradation_rate, protein_degradation_rate, 
                  basal_transcription_rate, translation_rate,
                  transcription_delay, initial_mRNA, initial_protein,
                  equilibration_time, sampling_timestep, switching_rate, initial_environment_on) ]*number_of_trajectories
#                   equilibration_time, transcription_schedule) ]*number_of_trajectories
    process_results = [ pool_of_processes.apply_async(generate_switching_trajectory, args=x)
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
    environment_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
     
    mRNA_trajectories[:,0] = sample_times
    protein_trajectories[:,0] = sample_times
    environment_trajectories[:,0] = sample_times
 
    for trajectory_index, this_trace in enumerate(list_of_traces): 
        # offset one index for time column
        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
        environment_trajectories[:,trajectory_index + 1] = this_trace[:,3]
  
    return mRNA_trajectories, protein_trajectories, environment_trajectories

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
        
@autojit(nopython = True)
def generate_switching_langevin_trajectory( duration = 720, 
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
                                  switching_rate = 1.0
                                  ):
    '''Generate one trace of the protein-autorepression model with transcriptional switching using a langevin approximation. 
    This function implements the Ito integral of 
     
    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(mu_m+alpha_m*G(P(t-tau) + 2*theta^2*alpha_m^2/lambda)d(ksi)
    dP/dt = -mu_p*P + alpha_p*M + sqrt(mu_p*alpha_p)d(ksi)
     
    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p, lambda are
    rates of mRNA degradation, protein degradation, basal transcription, translation, and switching; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G 
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient. Theta takes the form Theta^2 = (P(t-tau)/p_0)^n/(1+(p/p_0)^n)^3
     
    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.
     
    Warning : The time step of integration is chosen as 0.1 minute, and hence the time-delay is only
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
         
    switching_rate : float
        rate of environmental switching

    Returns
    -------
     
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
  
    total_time = duration + equilibration_time
    delta_t = 0.1
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
            repression_power = np.power(protein_at_delay/repression_threshold, hill_coefficient)
            switching_noise_strength = 2*delta_t*repression_power*basal_transcription_rate*basal_transcription_rate/(
                switching_rate*np.power(1+repression_power,3))
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(this_average_mRNA_degradation_number
                            +this_average_transcription_number
                            +switching_noise_strength)*np.random.randn())
             
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

@autojit(nopython = True)
def generate_switching_only_langevin_trajectory( duration = 720, 
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
                                  switching_rate = 1.0
                                  ):
    '''Generate one trace of the protein-autorepression model with transcriptional switching using a langevin approximation. 
    This function implements the Ito integral of 
     
    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(mu_m+alpha_m*G(P(t-tau) + 2*theta^2*alpha_m^2/lambda)d(ksi)
    dP/dt = -mu_p*P + alpha_p*M + sqrt(mu_p*alpha_p)d(ksi)
     
    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p, lambda are
    rates of mRNA degradation, protein degradation, basal transcription, translation, and switching; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G 
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient. Theta takes the form Theta^2 = (P(t-tau)/p_0)^n/(1+(p/p_0)^n)^3
     
    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.
     
    Warning : The time step of integration is chosen as 0.1 minute, and hence the time-delay is only
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
         
    switching_rate : float
        rate of environmental switching

    Returns
    -------
     
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
  
    total_time = duration + equilibration_time
    delta_t = 0.1
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
            d_mRNA = (-this_average_mRNA_degradation_number)
        else:
            protein_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            repression_power = np.power(protein_at_delay/repression_threshold, hill_coefficient)
            switching_noise_strength = 2*delta_t*repression_power*basal_transcription_rate*basal_transcription_rate/(
                switching_rate*np.power(1+repression_power,3))
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(switching_noise_strength)*np.random.randn())
             
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

def generate_switching_only_lna_trajectory( duration = 720, 
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
                                  switching_rate = 1.0
                                  ):
    '''Generate one trace of the protein-autorepression model with transcriptional switching using a linear noise approximation. 
    This function implements the Ito integral of 
     
    dM/dt = -mu_m*M + alpha_m*G(P(t-tau) + sqrt(mu_m+alpha_m*G(P(t-tau) + 2*theta^2*alpha_m^2/lambda)d(ksi)
    dP/dt = -mu_p*P + alpha_p*M + sqrt(mu_p*alpha_p)d(ksi)
     
    Here, M and P are mRNA and protein, respectively, and mu_m, mu_p, alpha_m, alpha_p, lambda are
    rates of mRNA degradation, protein degradation, basal transcription, translation, and switching; in that order.
    The variable ksi represents Gaussian white noise with delta-function auto-correlation and G 
    represents the Hill function G(P) = 1/(1+P/p_0)^n, where p_0 is the repression threshold
    and n is the Hill coefficient. Theta takes the form Theta^2 = (P(t-tau)/p_0)^n/(1+(p/p_0)^n)^3
     
    This model is an approximation of the stochastic version of the model in Monk, Current Biology (2003),
    which is implemented in generate_stochastic_trajectory(). For negative times we assume that there
    was no transcription.
     
    Warning : The time step of integration is chosen as 0.1 minute, and hence the time-delay is only
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
         
    switching_rate : float
        rate of environmental switching

    Returns
    -------
     
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
    total_time = duration + equilibration_time
    delta_t = 0.1
    sample_times = np.arange(0.0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 3))
    full_trace[:,0] = sample_times
    full_trace[0,1] = initial_mRNA
    full_trace[0,2] = initial_protein
    repression_threshold = float(repression_threshold)
    
    steady_state_mrna, steady_state_protein = hes5.calculate_steady_state_of_ode( repression_threshold = float(repression_threshold),
                                    hill_coefficient = hill_coefficient,
                                    mRNA_degradation_rate = mRNA_degradation_rate,
                                    protein_degradation_rate = protein_degradation_rate, 
                                    basal_transcription_rate = basal_transcription_rate,
                                    translation_rate = translation_rate)
 
    repression_power = np.power(steady_state_protein/repression_threshold, hill_coefficient)
    switching_noise_strength = 2*delta_t*repression_power*basal_transcription_rate*basal_transcription_rate/(
                switching_rate*np.power(1+repression_power,3))

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
            d_mRNA = (-this_average_mRNA_degradation_number)
        else:
            protein_at_delay = full_trace[time_index + 1 - delay_index_count,2]
            hill_function_value = 1.0/(1.0+np.power(protein_at_delay/repression_threshold,
                                                    hill_coefficient))
            this_average_transcription_number = basal_transcription_rate_per_timestep*hill_function_value
            this_average_mRNA_degradation_number = mRNA_degradation_rate_per_timestep*last_mRNA
            d_mRNA = (-this_average_mRNA_degradation_number
                      +this_average_transcription_number
                      +np.sqrt(switching_noise_strength)*np.random.randn())
             
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
 
def generate_multiple_switching_langevin_trajectories( number_of_trajectories = 10,
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
                                    switching_rate = 1.0,
                                    model = 'full'):
    '''Generate multiple langevin stochastic traces from the switching-modified Monk model by using
       generate_langevin_trajectory. This function operates sequentially, not in parallel.
     
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
        
    model : string
        if 'full', low-copy-number noise will be included. If 'switching_only', it will be excluded.
         
    Returns
    -------
     
    mRNA_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time, 
        each further column is one trace of mRNA copy numbers 
 
    protein_trajectories : ndarray
        2 dimensional array with [number_of_trajectories] columns, first column is time, 
        each further column is one trace of protein copy numbers 
    '''
    if model == 'full':
        trajectory_function = generate_switching_langevin_trajectory
    elif model =='switching_only':
        trajectory_function = generate_switching_only_langevin_trajectory
    elif model =='switching_only_lna':
        trajectory_function = generate_switching_only_lna_trajectory

    first_trace = trajectory_function(duration, 
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
                                      switching_rate)
 
    sample_times = first_trace[:,0]
    mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
     
    mRNA_trajectories[:,0] = sample_times
    protein_trajectories[:,0] = sample_times
    mRNA_trajectories[:,1] = first_trace[:,1]
    protein_trajectories[:,1] = first_trace[:,2]
 
    for trajectory_index in range(1,number_of_trajectories): 
        # offset one index for time column
        this_trace = trajectory_function(duration, 
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
                                         switching_rate)
 
        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
  
    return mRNA_trajectories, protein_trajectories

@autojit(nopython=True)
def generate_ode_switching_trajectory( duration = 720, 
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
                                    sampling_timestep = 1.0,
                                    switching_rate = 1.0,
                                    initial_environment_on = True,
                                    discretisation_time_step = 0.0001):
    '''Generate one trace of the Hes model with transcriptional switching. This function implements a stochastic version of
    the model in Monk, Current Biology (2003) where the hill function emerges from transcriptional switching. 
    This approximation is valid in the limit of large copy numbers, i.e. the only deviation from the
    dde description is the transcriptional bursting, which is explicitly calculated.

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
        
    switching_rate : float
        rate of environmental switching

    initial_environment_on : bool
        True if the environment is in on state at the beginning
        
    Returns
    -------
    
    trace : ndarray
        2 dimensional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number, fourth column is environmental state
    '''
    total_time = duration + equilibration_time

    discrete_delay = int(np.around(transcription_delay/discretisation_time_step))
    
    total_number_of_timesteps = int(np.around(total_time/discretisation_time_step))
    
    sampling_timestep_multiple = np.int(sampling_timestep/discretisation_time_step)
    total_number_of_sampling_timesteps = np.int(total_number_of_timesteps/sampling_timestep_multiple)
    
    trace = np.zeros((total_number_of_sampling_timesteps,4))
    buffer_size = discrete_delay+10
    environment_buffer = np.zeros(buffer_size, dtype = np.bool_)
    time_values = np.arange(0.0,total_time,sampling_timestep)
    trace[:,0] = time_values
    
    current_mRNA = initial_mRNA
    current_protein = initial_protein
    current_environment_on = initial_environment_on
    environment_buffer[0] = initial_environment_on
    
    sampling_index = 0

    for time_index in range(total_number_of_timesteps):

        if time_index%sampling_timestep_multiple == 0:
            trace[sampling_index,1] = current_mRNA 
            trace[sampling_index,2] = current_protein 
#             trace[sampling_index,3] = float(current_environment_on) 
            trace[sampling_index,3] = current_environment_on 
            sampling_index+=1
 
        if time_index < discrete_delay:
#             past_environment_on = float(initial_environment_on)
            past_environment_on = initial_environment_on
        else: 
            past_buffer_index = (time_index-discrete_delay)%buffer_size
#             past_environment_on = float(environment_buffer[past_buffer_index])
            past_environment_on = environment_buffer[past_buffer_index]

        dmRNA = ( basal_transcription_rate*past_environment_on
                  -mRNA_degradation_rate*current_mRNA )
        dprotein = ( translation_rate*current_mRNA -
                     protein_degradation_rate*current_protein)
        
        uniform_random_number = np.random.rand()
        if current_environment_on:
            switching_propensity = switching_rate*np.power(current_protein/repression_threshold,
                                                           hill_coefficient)*discretisation_time_step
            if uniform_random_number < switching_propensity:
                current_environment_on = False
        else:
            switching_propensity = switching_rate*discretisation_time_step
            if uniform_random_number < switching_propensity:
                current_environment_on = True
 
        current_mRNA += discretisation_time_step*dmRNA
        current_protein += discretisation_time_step*dprotein
        next_buffer_index = (time_index+1)%buffer_size
        environment_buffer[next_buffer_index] = current_environment_on
           
    # get rid of the equilibration time now
    trace = trace[ trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time
    return trace
  
def generate_multiple_switching_ode_trajectories( number_of_trajectories = 10,
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
                                    sampling_timestep = 1.0,
                                    switching_rate = 1.0,
                                    initial_environment_on = True,
                                    discretisation_time_step = 0.0001):
    '''Generate multiple langevin stochastic traces from the switching-modified Monk model by using
       generate_langevin_trajectory. This function operates sequentially, not in parallel.
     
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
    first_trace = generate_ode_switching_trajectory(duration, 
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
                                               sampling_timestep,
                                               switching_rate,
                                               initial_environment_on,
                                               discretisation_time_step)
 
    sample_times = first_trace[:,0]
    mRNA_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
    protein_trajectories = np.zeros((len(sample_times), number_of_trajectories + 1)) # one extra column for the time
     
    mRNA_trajectories[:,0] = sample_times
    protein_trajectories[:,0] = sample_times
    mRNA_trajectories[:,1] = first_trace[:,1]
    protein_trajectories[:,1] = first_trace[:,2]
 
    for trajectory_index in range(1,number_of_trajectories): 
        # offset one index for time column
        this_trace = generate_ode_switching_trajectory(duration, 
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
                                               sampling_timestep,
                                               switching_rate,
                                               initial_environment_on,
                                               discretisation_time_step)
 
        mRNA_trajectories[:,trajectory_index + 1] = this_trace[:,1] 
        protein_trajectories[:,trajectory_index + 1] = this_trace[:,2]
  
    return mRNA_trajectories, protein_trajectories

def calculate_theoretical_power_spectrum_at_parameter_point(basal_transcription_rate = 1.0,
                                                            translation_rate = 1.0,
                                                            repression_threshold = 100,
                                                            transcription_delay = 18.5,
                                                            mRNA_degradation_rate = 0.03,
                                                            protein_degradation_rate = 0.03,
                                                            hill_coefficient = 5,
                                                            switching_rate = 12.5,
                                                            normalise = False,
                                                            limits = [0,0.01]
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
        
    limits : list
        two float entries denoting the frequency limits in between which the theoretical power
        spectrum should be calculated.
       
    Returns
    -------
    
    power_spectrum : ndarray
        two coloumns, first column contains frequencies, second column contains power spectrum values
    '''
    actual_frequencies = np.linspace(limits[0],limits[1],1000)
    pi_frequencies = actual_frequencies*2*np.pi
    steady_state_mrna, steady_state_protein = hes5.calculate_steady_state_of_ode( repression_threshold = float(repression_threshold),
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

    power_spectrum_values = ((translation_rate**2*
                                (2*np.power(steady_state_protein/repression_threshold,hill_coefficient)/
                                (np.power(1+np.power(steady_state_protein/repression_threshold,hill_coefficient),3)*
                                 switching_rate)*basal_transcription_rate**2))/
                             ((-pi_frequencies**2 +
                                  protein_degradation_rate*mRNA_degradation_rate
                                  - basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.cos(pi_frequencies*transcription_delay))**2 
                               +
                               ((protein_degradation_rate+mRNA_degradation_rate)*
                                  pi_frequencies +
                                  basal_transcription_rate*translation_rate*steady_state_hill_derivative*
                                  np.sin(pi_frequencies*transcription_delay))**2)) 
                                 
    power_spectrum = np.vstack((actual_frequencies, power_spectrum_values)).transpose()
    if normalise:
        integral = np.trapz(power_spectrum[:,1], power_spectrum[:,0])
        power_spectrum[:,1] /= integral

    return power_spectrum

