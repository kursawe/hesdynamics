import math

def kalman_filter(protein_at_observations,model_parameters):
    """
    Perform Kalman-Bucy filter based on observation of protein
    copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

    Parameters
    ----------

    protein_at_observations : numpy array.
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        the corresponding time.

    model_parameters : numpy array.
        An array containing the model parameters in the following order.
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    Returns
    -------

    state_space_mean : numpy array.
        An array of dimension n x 3, which gives the number of observation time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein.

    state_space_variance : numpy array.
        An array of dimension 2n x 2n.
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    """
    ## include some initialisation here - use the functions calculate_steady_state_of_ode and
    ## calculate_approximate_standard_deviation_at_parameter_point form the hes5 module

    ## then loop through observations
    ## and at each observation implement prediction step and then the update step

    state_space_mean = something
    state_space_variance = something
    for observation_index, current_observation in enumerate(protein_at_observations):
        predicted_state_space_mean, predicted_state_space_variance = kalman_prediction_step(state_space_mean,
                                                                                            state_space_variance,
                                                                                            model_parameters,
                                                                                            observation_time_step)
        state_space_mean, state_space_variance = kalman_update_step(predicted_state_space_mean,
                                                                   predicted_state_space_variance,
                                                                   current_observation,
                                                                   time_delay)

    return state_space_mean, state_space_variance


def kalman_prediction_step(state_space_mean,state_space_variance,model_parameters,observation_time_step):
    """
    Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
    state space mean and variance. This gives rho_{t+\delta t-tau:t+\delta t} and P_{t+\delta t-tau:t+\delta t},
    using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018).

    Parameters
    ----------

    state_space_mean : numpy array.
    	The dimension is n x 3, where n is the number of previous observations until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein. It
        represents the information based on observations we have already made.

    state_space_variance : numpy array.
    	The dimension is 2n x 2n, where n is the number of previous observations until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    model_parameters : numpy array.
        An array containing the model parameters in the following order.
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    observation_time_step : float.
        This gives the time between each experimental observation, and allows us to calculate the number of
        discretisation time steps required for the forward Euler scheme.

    Returns
    -------
    predicted_state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of previous observations until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein.

    predicted_state_space_variance : numpy array.
    The dimension is 2n x 2n, where n is the number of previous observations until the current time.
        [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
          cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]number_of_obse

    """

    discretisation_time_step = 1.0
    number_of_hidden_states = int(observation_time_step/discretisation_time_step)
    number_of_observations = len(state_space_mean[:,0])

    repression_threshold = model_parameters[0]
    hill_coefficient = model_parameters[1]
    mRNA_degradation_rate = model_parameters[2]
    protein_degradation_rate = model_parameters[3]
    basal_transcription_rate = model_parameters[4]
    translation_rate = model_parameters[5]
    transcription_delay = model_parameters[6]

    discrete_delay = int(np.around(transcription_delay/discretisation_time_step))

    ## need to define P(t,s) and P(s,t) to use in the calculation of the variance.

    for time_index in range(number_of_hidden_states):

        hill_function_value = 1.0/(1.0+np.power(state_space_mean[time_index-discrete_delay,2]/repression_threshold,hill_coefficient))
        hill_function_derivative_value = (-hill_coefficient/repression_threshold)*np.power(1+(state_space_mean[time_index-discrete_delay,2]/repression_threshold),-(hill_coefficient+1))

        state_space_mean[time_index+1,(1,2)] = (state_space_mean[time_index,(1,2)] + number_of_hidden_states*(state_space_mean[time_index,(1,2)].dot(
                                                                                        np.array([[-mRNA_degradation_rate,translation_rate],
                                                                                                 [0,-protein_degradation_rate]]))
                                                                                   + np.array([basal_transcription_rate*hill_function_value,0])))

        state_space_variance[(time_index+1,time_index+number_of_observations+1),(time_index+1,time_index+number_of_observations+1)] = (

            state_space_variance[(time_index,time_index+number_of_observations),(time_index,time_index+number_of_observations)] + number_of_hidden_states*(
                np.array([[-mRNA_degradation_rate,0],[translation_rate,-protein_degradation_rate]]).dot(state_space_variance[(time_index,time_index+number_of_observations),(time_index,time_index+number_of_observations)])
                + np.transpose(state_space_variance[(time_index,time_index+number_of_observations),(time_index,time_index+number_of_observations)]).dot(np.array([[-mRNA_degradation_rate,translation_rate],
                         [0,-protein_degradation_rate]]))
                + np.array([[0,hill_function_derivative_value],[0,0]])*P(s,t) + P(s,t)*np.array([[0,0],[basal_transcription_rate*hill_function_derivative_value]])
                + np.array([[mRNA_degradation_rate*state_space_mean[time_index,1]+basal_transcription_rate*hill_function_value,0],
                            [0,translation_rate*state_space_mean[time_index,1]+protein_degradation_rate*state_space_mean[time_index,2]]])
            )
        )


def kalman_update_step(predicted_state_space_mean, predicted_state_space_variance,current_observation,time_delay):
    """
    Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
    This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
    This assumes that the observations are collected at fixed time intervals.

    Parameters
    ----------

    predicted_state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of previous observations until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein.

    predicted_state_space_variance : numpy array.
        The dimension is 2n x 2n, where n is the number of previous observations until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    current_observation : numpy array.
        The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

    time_delay : float.
        The fixed transciptional time delay in the system. This tells us how far back we need to update our
        state space estimates.

    Returns
    -------

    state_space_mean : numpy array.
        The dimension is n x 3, where the first column is time, and the second and third columns are the mean
        mRNA and mean protein levels respectively. This corresponds to \rho* in
        Calderazzo et al., Bioinformatics (2018).

    state_space_variance : numpy array.
        The dimension is 2n x 2n, where n is the number of previous observations until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ].

        This corresponds to P* in Calderazzo et al., Bioinformatics (2018).

    """
    ## first we need \rho_{t+\delta t-\tau:t+\delta t} and P_{t+\delta t-\tau:t+\delta t},
    ## which can be obtained using the differential equations in supplementary section 4.
    ## This will be done in the kalman_prediction_step function.
    ## We will call these 'state_space_mean' and 'state_space_variance',
    ## and they will be updated in the following way.

    # initialise updated mean and variance arrays.
    state_space_mean = predicted_state_space_mean
    state_space_variance = predicted_state_space_variance

    observation_time_step = state_space_mean[1,0] - state_space_mean[0,0]
    maximum_delay_index = int(math.ceil(time_delay/observation_time_step))
    number_of_observations = len(state_space_mean[:,0])

    # This is F in the paper
    observation_transform = np.array([0,1])
    observation_variance = 0.1
    helper_inverse = 1.0/(observation_transform.dot(state_space_variance[(number_of_observations-1,-1),
                                                     (number_of_observations-1,-1)].dot(np.transpose(observation_transform)))
                                                     +observation_variance)

    # need to define C (the coefficient of adaptation) somewhere
    # also there are a few things wrong with this. I think the right hand side should also
    # use the updated mean and variance, and also the observation y_{t+\delta t} in the first
    # equation is wrong.
    for past_time_index in range(len(state_space_mean),len(state_space_mean)-maximum_delay_index,-1):
        # need to double-check this derivation for the following line, this is C in the paper
        adaptation_coefficient = state_space_variance[(past_time_index-1,past_time_index-1),
                                    (number_of_observations-1,2*number_of_observations-1)].dot(
                                    transpose(observation_transform))*helper_inverse

    	state_space_mean[past_time_index,(1,2)] = (state_space_mean[past_time_index,(1,2)] +
    	    adaptation_coefficient*(current_observation[1]-observation_transform.dot(state_space_mean[-1,(1,2)])))

    	state_space_variance[past_time_index,(1,2)] = (state_space_variance[past_time_index,(1,2)] -
                                                       adaptation_coefficient*observation_transform.dot(state_space_variance[(number_of_observations-1,2*number_of_observations-1),
                                                       (past_time_index,(1,2)])))

	return state_space_mean, state_space_variance
