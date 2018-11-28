import math
import numpy as np
import hes5
from numpy import number

#discretisation_time_step=1.0

def kalman_filter(protein_at_observations,model_parameters,measurement_variance = 10):
    """
    Perform Kalman-Bucy filter based on observation of protein
    copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

    Parameters
    ----------

    protein_at_observations : numpy array.
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        that time. The filter assumes that observations are generated with a fixed, regular time interval.

    model_parameters : numpy array.
        An array containing the model parameters in the following order:
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    Returns
    -------

    state_space_mean : numpy array.
        An array of dimension n x 3, where n is the number of inferred time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein. Time points are generated every minute

    state_space_variance : numpy array.
        An array of dimension 2n x 2n.
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    predicted_observation_distributions : numpy array.
        An array of dimension n x 3 where n is the number of observation time points.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at each time point, respectively.
    """
    time_delay = model_parameters[6]

    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = int(np.around(time_delay/discretisation_time_step))

    observation_time_step = protein_at_observations[1,0]-protein_at_observations[0,0]
    # 'synthetic' observations, which allow us to update backwards in time
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))

    ## initialise "negative time" with the mean and standard deviations of the LNA
    initial_number_of_states = discrete_delay + 1
    initial_state_space_mean = np.zeros((initial_number_of_states,3))
    initial_state_space_mean[:,(1,2)] = hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[0],
                                                                           hill_coefficient=model_parameters[1],
                                                                           mRNA_degradation_rate=model_parameters[2],
                                                                           protein_degradation_rate=model_parameters[3],
                                                                           basal_transcription_rate=model_parameters[4],
                                                                           translation_rate=model_parameters[5])

    # assign time entries
    initial_state_space_mean[:,0] = np.linspace(-time_delay,0,initial_number_of_states)

    # initialise initial covariance matrix
    initial_state_space_variance = np.zeros((2*(initial_number_of_states),2*(initial_number_of_states)))

    # set the mRNA variance at nagative times to the LNA approximation
    LNA_mRNA_variance = np.power(hes5.calculate_approximate_mRNA_standard_deviation_at_parameter_point(),2)
    # the top left block of the matrix corresponds to the mRNA covariance, see docstring above
    np.fill_diagonal( initial_state_space_variance[:initial_number_of_states,:initial_number_of_states] ,
                      LNA_mRNA_variance)

    # set the protein variance at nagative times to the LNA approximation
    LNA_protein_variance = np.power(hes5.calculate_approximate_protein_standard_deviation_at_parameter_point(),2)
    # the bottom right block of the matrix corresponds to the mRNA covariance, see docstring above
    np.fill_diagonal( initial_state_space_variance[initial_number_of_states:,initial_number_of_states:] , LNA_protein_variance )

    observation_transform = np.array([0,1])

    predicted_observation_distributions = np.zeros((protein_at_observations.shape[0],3))
    predicted_observation_distributions[0,0] = 0
    predicted_observation_distributions[0,1] = observation_transform.dot(initial_state_space_mean[-1,(1,2)])

    current_number_of_states = initial_state_space_mean.shape[0]
    last_predicted_covariance_matrix = initial_state_space_variance[np.ix_([current_number_of_states-1,-1],
                                                                             [current_number_of_states-1,-1])]

    predicted_observation_distributions[0,2] = (observation_transform.dot(
                                                                     last_predicted_covariance_matrix).dot(observation_transform.transpose())
                                                                     +
                                                                     measurement_variance)
    # update the past ("negative time")
    state_space_mean, state_space_variance = kalman_update_step(initial_state_space_mean,
                                                                initial_state_space_variance,
                                                                protein_at_observations[0],
                                                                time_delay,
                                                                measurement_variance)
    ## loop through observations and at each observation apply the Kalman prediction step and then the update step
    for observation_index, current_observation in enumerate(protein_at_observations[1:]):
        predicted_state_space_mean, predicted_state_space_variance = kalman_prediction_step(state_space_mean,
                                                                                            state_space_variance,
                                                                                            model_parameters,
                                                                                            observation_time_step)
        predicted_observation_distributions[observation_index+1,0] = current_observation[0]
        predicted_observation_distributions[observation_index+1,1] = observation_transform.dot(predicted_state_space_mean[-1,(1,2)])

        current_number_of_states = state_space_mean.shape[0]
        last_predicted_covariance_matrix = predicted_state_space_variance[np.ix_([current_number_of_states-1,-1],
                                                                                 [current_number_of_states-1,-1])]

        predicted_observation_distributions[observation_index+1,2] = (observation_transform.dot(
                                                                         last_predicted_covariance_matrix).dot(observation_transform.transpose())
                                                                         +
                                                                         measurement_variance)

        state_space_mean, state_space_variance = kalman_update_step(predicted_state_space_mean,
                                                                    predicted_state_space_variance,
                                                                    current_observation,
                                                                    time_delay,
                                                                    measurement_variance)

    return state_space_mean, state_space_variance, predicted_observation_distributions

def kalman_prediction_step(state_space_mean,
                           state_space_variance,
                           model_parameters,
                           observation_time_step):
    """
    Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
    state space mean and variance. This gives rho_{t+\delta t-tau:t+\delta t} and P_{t+\delta t-tau:t+\delta t},
    using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
    approximated using a forward Euler scheme.

    Parameters
    ----------

    state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of states until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein. It
        represents the information based on observations we have already made.

    state_space_variance : numpy array.
        The dimension is 2n x 2n, where n is the number of states until the current time. The definition
        is identical to the one provided in the Kalman filter function, i.e.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    model_parameters : numpy array.
        An array containing the model parameters. The order is identical to the one provided in the
        Kalman filter function documentation, i.e.
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    observation_time_step : float.
        This gives the time between each experimental observation. This is required to know how far
        the function should predict.

    Returns
    -------
    predicted_state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of previous observations until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein.

    predicted_state_space_variance : numpy array.
    The dimension is 2n x 2n, where n is the number of previous observations until the current time.
        [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
          cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]
    """
    discretisation_time_step = 1.0
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))
    previous_number_of_states = state_space_mean.shape[0]
    new_number_of_states = previous_number_of_states + number_of_hidden_states

    ## extend the current mean and variance make space for predictions of hidden and observated states
    predicted_state_space_mean = np.zeros((new_number_of_states,3))
    predicted_state_space_mean[:previous_number_of_states] = state_space_mean

    current_time = state_space_mean[-1,0]
    predicted_state_space_mean[previous_number_of_states:,0] = np.linspace(current_time+discretisation_time_step,
                                                                           current_time+observation_time_step,
                                                                           number_of_hidden_states)

    predicted_state_space_variance = np.zeros((2*new_number_of_states,2*new_number_of_states))
    # filling in mRNA block
    predicted_state_space_variance[:previous_number_of_states,:previous_number_of_states] = state_space_variance[:previous_number_of_states,:previous_number_of_states]
    # filling in mRNA-Protein covariance block
    predicted_state_space_variance[:previous_number_of_states,
                                   new_number_of_states:
                                   (new_number_of_states + previous_number_of_states)] = state_space_variance[:previous_number_of_states,previous_number_of_states:]
    # filling in protein-mRNA covariance block
    predicted_state_space_variance[new_number_of_states:(new_number_of_states + previous_number_of_states),
                                   :previous_number_of_states] = state_space_variance[previous_number_of_states:,:previous_number_of_states]
    # filling in protein block
    predicted_state_space_variance[new_number_of_states:(new_number_of_states + previous_number_of_states),
                                   new_number_of_states:(new_number_of_states + previous_number_of_states)] = state_space_variance[previous_number_of_states:,
                                                                                                                                   previous_number_of_states:]

    ## name the model parameters
    repression_threshold = model_parameters[0]
    hill_coefficient = model_parameters[1]
    mRNA_degradation_rate = model_parameters[2]
    protein_degradation_rate = model_parameters[3]
    basal_transcription_rate = model_parameters[4]
    translation_rate = model_parameters[5]
    transcription_delay = model_parameters[6]

    discrete_delay = int(np.around(transcription_delay/discretisation_time_step))

    ## next_time_index corresponds to 't+Deltat' in the propagation equation on page 5 of the supplementary
    ## material in the calderazzo paper
    for next_time_index in range(previous_number_of_states, previous_number_of_states + number_of_hidden_states):
        current_time_index = next_time_index - 1 # this corresponds to t
        past_time_index = current_time_index - discrete_delay # this corresponds to t-tau
        current_mean = predicted_state_space_mean[current_time_index,(1,2)]
        past_protein = predicted_state_space_mean[past_time_index,2]

        hill_function_value = 1.0/(1.0+np.power(past_protein/repression_threshold,hill_coefficient))

        hill_function_derivative_value = - hill_coefficient*np.power(past_protein/repression_threshold,
                                                                     hill_coefficient - 1)/( repression_threshold*
                                           np.power(1.0+np.power( past_protein/repression_threshold,
                                                                hill_coefficient),2))

        ## derivative of mean is contributions from instant reactions + contributions from past reactions
        derivative_of_mean = ( np.array([[-mRNA_degradation_rate,0.0],
                                         [translation_rate,-protein_degradation_rate]]).dot(current_mean) +
                               np.array([basal_transcription_rate*hill_function_value,0]) )

        next_mean = current_mean + discretisation_time_step*derivative_of_mean
        # ensures the prediction is non negative
        next_mean = np.maximum(next_mean,0)
        predicted_state_space_mean[next_time_index,(1,2)] = next_mean

        current_covariance_matrix = predicted_state_space_variance[np.ix_([current_time_index,new_number_of_states + current_time_index],
                                                                          [current_time_index,new_number_of_states + current_time_index])]
        # this is P(t-\tau,t) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_past_to_now = predicted_state_space_variance[np.ix_([past_time_index,new_number_of_states + past_time_index],
                                                                              [current_time_index,new_number_of_states +current_time_index])]
        # this is P(t,t-\tau) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_now_to_past = predicted_state_space_variance[np.ix_([current_time_index,new_number_of_states + current_time_index],
                                                                              [past_time_index,new_number_of_states + past_time_index])]

        # derivations for the following are found in Calderazzo et. al. (2018)
        # g is [[-mRNA_degradation_rate,0],                  *[M(t),
        #       [translation_rate,-protein_degradation_rate]] [P(t)]
        # and its derivative will be called instant_jacobian
        # f is [[basal_transcription_rate*hill_function(past_protein)],0]
        # and its derivative with respect to the past state will be called delayed_jacobian
        # the matrix A in the paper will be called variance_of_noise
        instant_jacobian = np.array([[-mRNA_degradation_rate,0],[translation_rate,-protein_degradation_rate]])
        # jacobian of f is derivative of f with respect to past state ([past_mRNA, past_protein])
        delayed_jacobian = np.array([[0,basal_transcription_rate*hill_function_derivative_value],[0,0]])

        variance_change_current_contribution = ( instant_jacobian.dot(current_covariance_matrix) +
                                                 np.transpose(instant_jacobian.dot(current_covariance_matrix)) )

        variance_change_past_contribution = ( delayed_jacobian.dot(covariance_matrix_past_to_now) +
                                              covariance_matrix_now_to_past.dot(np.transpose(delayed_jacobian)) )

        variance_of_noise = np.array([[mRNA_degradation_rate*current_mean[0]+basal_transcription_rate*hill_function_value,0],
                                      [0,translation_rate*current_mean[0]+protein_degradation_rate*current_mean[1]]])

        derivative_of_variance = ( variance_change_current_contribution +
                                   variance_change_past_contribution +
                                   variance_of_noise )

        # P(t+Deltat,t+Deltat)
        next_covariance_matrix = current_covariance_matrix + discretisation_time_step*derivative_of_variance
        # ensure that the diagonal entries are non negative
        np.fill_diagonal(next_covariance_matrix,np.maximum(np.diag(next_covariance_matrix),0))

        predicted_state_space_variance[np.ix_([next_time_index, new_number_of_states + next_time_index],
                                              [next_time_index, new_number_of_states + next_time_index])] = next_covariance_matrix

        ## now we need to update the cross correlations, P(s,t) in the Calderazzo paper
        # the range needs to include t, since we want to propagate P(t,t) into P(t,t+Deltat)
        for intermediate_time_index in range(past_time_index,current_time_index+1):
            # This corresponds to P(s,t) in the Calderazzo paper
            covariance_matrix_intermediate_to_current = predicted_state_space_variance[np.ix_([intermediate_time_index, new_number_of_states + intermediate_time_index],
                                                                                              [current_time_index, new_number_of_states + current_time_index])]

            covariance_matrix_intermediate_to_past = predicted_state_space_variance[np.ix_([intermediate_time_index, new_number_of_states + intermediate_time_index],
                                                                                           [past_time_index, new_number_of_states + past_time_index])]

            covariance_derivative = ( covariance_matrix_intermediate_to_current.dot( np.transpose(instant_jacobian)) +
                                      covariance_matrix_intermediate_to_past.dot( np.transpose(delayed_jacobian)))

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next = covariance_matrix_intermediate_to_current + discretisation_time_step*covariance_derivative

            # Fill in the big matrix
            predicted_state_space_variance[np.ix_([intermediate_time_index,new_number_of_states + intermediate_time_index],
                                                  [next_time_index,new_number_of_states + next_time_index])] = covariance_matrix_intermediate_to_current

            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            predicted_state_space_variance[np.ix_([next_time_index,new_number_of_states + next_time_index],
                                                  [intermediate_time_index,new_number_of_states + intermediate_time_index])] = covariance_matrix_intermediate_to_current.transpose()
    #import pdb; pdb.set_trace()
    return predicted_state_space_mean, predicted_state_space_variance

def kalman_update_step(predicted_state_space_mean,predicted_state_space_variance,current_observation,time_delay,measurement_variance):
    """
    Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
    This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
    This assumes that the observations are collected at fixed time intervals.

    Parameters
    ----------

    predicted_state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of states until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein.

    predicted_state_space_variance : numpy array.
        The dimension is 2n x 2n, where n is the number of states until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    current_observation : numpy array.
        The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

    time_delay : float.
        The fixed transciptional time delay in the system. This tells us how far back we need to update our
        state space estimates.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    Returns
    -------

    state_space_mean : numpy array.
        The dimension is n x 3, where the first column is time, and the second and third columns are the mean
        mRNA and mean protein levels respectively. This corresponds to \rho* in
        Calderazzo et al., Bioinformatics (2018).

    state_space_variance : numpy array.
        This corresponds to P* in Calderazzo et al., Bioinformatics (2018).
        The dimension is 2n x 2n, where n is the number of states until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ].
    """
    discretisation_time_step = 1 #predicted_state_space_mean[1,0] - predicted_state_space_mean[0,0]
    discrete_delay = int(np.around(time_delay/discretisation_time_step))
    number_of_states = predicted_state_space_mean.shape[0]

    # predicted_state_space_mean until delay, corresponds to
    # rho(t+Deltat-delay:t+deltat). Includes current value and discrete_delay past values
    shortened_state_space_mean = predicted_state_space_mean[-(discrete_delay+1):,(1,2)]

    # put protein values underneath mRNA values, to make vector of means (rho)
    # consistent with variance (P)
    stacked_state_space_mean = np.hstack((shortened_state_space_mean[:,0],
                                          shortened_state_space_mean[:,1]))

    predicted_final_state_space_mean = predicted_state_space_mean[-1,(1,2)]

    # extract covariance matrix up to delay
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    mRNA_indices_to_keep = np.arange(number_of_states - discrete_delay - 1,number_of_states,1)
    protein_indices_to_keep = np.arange(2*number_of_states - discrete_delay - 1,2*number_of_states,1)
    all_indices_up_to_delay = np.hstack((mRNA_indices_to_keep, protein_indices_to_keep))
    shortened_covariance_matrix = predicted_state_space_variance[np.ix_(all_indices_up_to_delay,
                                                                        all_indices_up_to_delay)]

    # extract P(t+Deltat-delay:t+deltat,t+Deltat)
    shortened_covariance_matrix_past_to_final = shortened_covariance_matrix[:,((discrete_delay),-1)]

    # and P(t+Deltat,t+Deltat-delay:t+deltat)
    shortened_covariance_matrix_final_to_past = shortened_covariance_matrix[((discrete_delay),-1),:]

    # This is F in the paper
    observation_transform = np.array([0,1])

    # This is P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_matrix = predicted_state_space_variance[np.ix_([number_of_states-1,-1],
                                                                              [number_of_states-1,-1])]

    # This is (FP_{t+Deltat}F^T + Sigma_e)^-1
    helper_inverse = 1.0/(observation_transform.dot( predicted_final_covariance_matrix.dot(np.transpose(observation_transform)))
                                                     + measurement_variance )

    # This is C in the paper
    adaptation_coefficient = shortened_covariance_matrix_past_to_final.dot(
                                np.transpose(observation_transform) )*helper_inverse
    # This is rho*
    updated_stacked_state_space_mean = ( stacked_state_space_mean +
                                         adaptation_coefficient*(current_observation[1] -
                                                                 observation_transform.dot(
                                                                     predicted_final_state_space_mean)) )
    # ensures the the mean mRNA and Protein are non negative
    updated_stacked_state_space_mean = np.maximum(updated_stacked_state_space_mean,0)

    # unstack the rho into two columns, one with mRNA and one with protein
    updated_state_space_mean = np.column_stack((updated_stacked_state_space_mean[:(discrete_delay+1)],
                                                updated_stacked_state_space_mean[(discrete_delay+1):]))

    # initialise state space mean
    state_space_mean = np.copy(predicted_state_space_mean)

    # Fill in the updated values
    state_space_mean[-(discrete_delay+1):,(1,2)] = updated_state_space_mean

    # This is P*
    updated_shortened_covariance_matrix = ( shortened_covariance_matrix -
                                            np.dot(adaptation_coefficient.reshape((2*(discrete_delay+1),1)),observation_transform.reshape((1,2))).dot(
                                                shortened_covariance_matrix_final_to_past))
    # ensure that the diagonal entries are non negative
    np.fill_diagonal(updated_shortened_covariance_matrix,np.maximum(np.diag(updated_shortened_covariance_matrix),0))
    # initialise state space variance
    state_space_variance = np.copy(predicted_state_space_variance)

    # Fill in updated values
    state_space_variance[np.ix_(all_indices_up_to_delay,
                                all_indices_up_to_delay)] = updated_shortened_covariance_matrix

#     print('in update step, this is the shortened P before and after updating')
#     print(shortened_covariance_matrix)
#     print(updated_shortened_covariance_matrix)
# might try the following


    return state_space_mean, state_space_variance

def calculate_log_likelihood_at_parameter_point(protein_at_observations,model_parameters,measurement_variance = 10):
    """
    Calculates the log of the likelihood of our data given the paramters, using the Kalman filter. It uses the
    predicted_observation_distributions from the kalman_filter function. The entries of this array in the second and
    third columns represent the probability of the future observation of mRNA and Protein respectively, given our current knowledge.

    Parameters
    ----------

    protein_at_observations : numpy array.
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        that time.

    model_parameters : numpy array.
        An array containing the model parameters in the following order:
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    Returns
    -------

    log_likelihood : float.
        The log of the likelihood of the data.
    """
    from scipy.stats import norm

    _, _, predicted_observation_distributions = kalman_filter(protein_at_observations,
                                                              model_parameters,
                                                              measurement_variance)
    observations = protein_at_observations[:,1]
    mean = predicted_observation_distributions[:,1]
    sd = np.sqrt(predicted_observation_distributions[:,2])

    log_likelihood = np.sum(norm.logpdf(observations,mean,sd))

    return log_likelihood

def kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,acceptance_tuner,parameter_covariance,initial_state):
    """
    A basic random walk metropolis algorithm that infers parameters for a given
    set of protein observations. The likelihood is calculated using the
    calculate_likelihood_at_parameter_point function, and uninformative normal
    priors are assumed.

    Parameters
    ----------

    iterations : float.
        The number of iterations desired.

    protein_at_observations : numpy array.
        An array containing protein observations over a given length of time.

    hyper_parameters : numpy array.
        A 1x14 array containing the hyperparameters for the model parameter prior distributions. The
        distributions are chosen to be from the Gamma family, given that values are restricted to the
        postive reals. Thus there are two hyperparameters for each model parameter.
        The model parameters are given in the order:
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.
        Therefore the first two entries correspond to repression_threshold, the
        next two to hill_coefficient, etc.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    acceptance_tuner : float.
        A scalar value which is to be tuned to get an optimal level of acceptance (0.234) in the random walk
        algorithm. See Roberts et. al. (1997)

    parameter_covariance : numpy array.
        A 7x7 variance-covariance matrix of the state space parameters. It is obtained by first doing a run of the algorithm,
        and then computing the variance-covariance matrix of the output, after discarding burn-in.

    initial_state : numpy array.
        A 1x7 array containing the initial state in parameter space. This is obtained by first doing a run of the
        algorithm, and then computing the column means (after discarding burn-in) to get a good initial value.

    Returns
    -------

    random_walk : numpy array.
        An array with dimensions (iterations x 7). Each column contains the random walk for each parameter.

    acceptance_rate : float.
        An integer between 0 and 1 which tells us the proportion of accepted proposal parameters. This is 0.234.
    """
    from scipy.stats import gamma, multivariate_normal
    zero_row = np.zeros(7)
    identity = np.identity(7)
    cholesky_covariance = np.linalg.cholesky(parameter_covariance)
    shape = hyper_parameters[0:14:2]
    scale = hyper_parameters[1:14:2]
    current_state = initial_state

    random_walk = np.zeros((iterations,7))
    acceptance_count = 0
    for i in range(0,iterations):
        new_state = current_state + acceptance_tuner*cholesky_covariance.dot(multivariate_normal.rvs(zero_row,identity))

        if all(item > 0 for item in new_state) == True:
            log_prior_ratio = np.sum(gamma.logpdf(new_state,a=shape,scale=scale)) - np.sum(gamma.logpdf(current_state,a=shape,scale=scale))
            log_likelihood_ratio = (calculate_log_likelihood_at_parameter_point(protein_at_observations,new_state,measurement_variance) -
                                    calculate_log_likelihood_at_parameter_point(protein_at_observations,current_state,measurement_variance))
            print('log prior ratio:',log_prior_ratio)
            print('log likelihood ratio:', log_likelihood_ratio)
            acceptance_ratio = np.exp(log_prior_ratio+log_likelihood_ratio)

            if np.random.uniform() < acceptance_ratio:
                current_state = new_state
                acceptance_count += 1

        random_walk[i,:] = current_state

    acceptance_rate = acceptance_count/iterations
    return random_walk, acceptance_rate
