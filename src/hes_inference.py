import math
import numpy as np
import hes5
from numpy import number
from numba import jit
from scipy.stats import gamma, multivariate_normal, uniform
import multiprocessing as mp

#discretisation_time_step=1.0
# @jit(nopython=True)
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

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    predicted_observation_distributions : numpy array.
        An array of dimension n x 3 where n is the number of observation time points.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at each time point, respectively.
    """
    time_delay = model_parameters[6]
    number_of_observations = protein_at_observations.shape[0]
    observation_time_step = protein_at_observations[1,0]-protein_at_observations[0,0]
    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = int(np.around(time_delay/discretisation_time_step))
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))
    initial_number_of_states = discrete_delay + 1
    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states
    # scaling factors for mRNA and protein respectively. For example, observation might be fluorescence,
    # so the scaling would correspond to how light intensity relates to molecule number.
    observation_transform = np.array([0.0,1.0])

    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions = kalman_filter_state_space_initialisation(protein_at_observations,
                                                                                                                                                                                         model_parameters,
                                                                                                                                                                                         measurement_variance)
    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    # for observation_index, current_observation in enumerate(protein_at_observations[1:]):
    for observation_index in range(len(protein_at_observations)-1):
        current_observation = protein_at_observations[1+observation_index,:]
        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_prediction_step(state_space_mean,
                                                                                                                                      state_space_variance,
                                                                                                                                      state_space_mean_derivative,
                                                                                                                                      state_space_variance_derivative,
                                                                                                                                      current_observation,
                                                                                                                                      model_parameters,
                                                                                                                                      observation_time_step)
        current_number_of_states = int(np.around(current_observation[0]/observation_time_step))*number_of_hidden_states + initial_number_of_states

        predicted_observation_distributions[observation_index + 1] = kalman_observation_distribution_parameters(predicted_observation_distributions,
                                                                                                                current_observation,
                                                                                                                state_space_mean,
                                                                                                                state_space_variance,
                                                                                                                current_number_of_states,
                                                                                                                total_number_of_states,
                                                                                                                measurement_variance,
                                                                                                                observation_index)

        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                                  state_space_variance,
                                                                                                                                  state_space_mean_derivative,
                                                                                                                                  state_space_variance_derivative,
                                                                                                                                  protein_at_observations[0],
                                                                                                                                  time_delay,
                                                                                                                                  observation_time_step,
                                                                                                                                  measurement_variance)

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions

def kalman_filter_state_space_initialisation(protein_at_observations,model_parameters,measurement_variance = 10):
    """
    A function for initialisation of the state space mean and variance, and update for the "negative" times that
     are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
     and then updates them with kalman_update_step.

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

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    predicted_observation_distributions : numpy array.
        An array of dimension n x 3 where n is the number of observation time points.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at each time point, respectively.
    """
    time_delay = model_parameters[6]
    number_of_observations = protein_at_observations.shape[0]

    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = int(np.around(time_delay/discretisation_time_step))

    observation_time_step = protein_at_observations[1,0]-protein_at_observations[0,0]
    # 'synthetic' observations, which allow us to update backwards in time
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))

    ## initialise "negative time" with the mean and standard deviations of the LNA
    initial_number_of_states = discrete_delay + 1
    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states

    state_space_mean = np.zeros((total_number_of_states,3))
    # potential solution for numba:
#     state_space_mean = np.ones((total_number_of_states,3))
    state_space_mean[:initial_number_of_states,(1,2)] = hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[0],
                                                                                                   hill_coefficient=model_parameters[1],
                                                                                                   mRNA_degradation_rate=model_parameters[2],
                                                                                                   protein_degradation_rate=model_parameters[3],
                                                                                                   basal_transcription_rate=model_parameters[4],
                                                                                                   translation_rate=model_parameters[5])

    final_observation_time = protein_at_observations[-1,0]
    # assign time entries
    state_space_mean[:,0] = np.linspace(-time_delay,final_observation_time,total_number_of_states)

    # initialise initial covariance matrix
    state_space_variance = np.zeros((2*(total_number_of_states),2*(total_number_of_states)))

    # set the mRNA variance at nagative times to the LNA approximation
    LNA_mRNA_variance = np.power(hes5.calculate_approximate_mRNA_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                   hill_coefficient=model_parameters[1],
                                                                                                   mRNA_degradation_rate=model_parameters[2],
                                                                                                   protein_degradation_rate=model_parameters[3],
                                                                                                   basal_transcription_rate=model_parameters[4],
                                                                                                   translation_rate=model_parameters[5],
                                                                                                   transcription_delay=model_parameters[6]),2)
    # the top left block of the matrix corresponds to the mRNA covariance, see docstring above
    np.fill_diagonal( state_space_variance[:initial_number_of_states,:initial_number_of_states] ,
                    LNA_mRNA_variance)
    # potential solution for numba:
#     np.fill_diagonal( state_space_variance[:initial_number_of_states,:initial_number_of_states] ,
#                     1.0)

    # set the protein variance at nagative times to the LNA approximation
    LNA_protein_variance = np.power(hes5.calculate_approximate_protein_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                   hill_coefficient=model_parameters[1],
                                                                                                   mRNA_degradation_rate=model_parameters[2],
                                                                                                   protein_degradation_rate=model_parameters[3],
                                                                                                   basal_transcription_rate=model_parameters[4],
                                                                                                   translation_rate=model_parameters[5],
                                                                                                   transcription_delay=model_parameters[6]),2)
    # the bottom right block of the matrix corresponds to the mRNA covariance, see docstring above
    np.fill_diagonal( state_space_variance[total_number_of_states:total_number_of_states + initial_number_of_states,
                                            total_number_of_states:total_number_of_states + initial_number_of_states] , LNA_protein_variance )
    # potential solution for numba:
#     np.fill_diagonal( state_space_variance[total_number_of_states:total_number_of_states + initial_number_of_states,
#                                             total_number_of_states:total_number_of_states + initial_number_of_states] , 1.0 )

    observation_transform = np.array([0.0,1.0])

    predicted_observation_distributions = np.zeros((protein_at_observations.shape[0],3))
    predicted_observation_distributions[0,0] = 0
    predicted_observation_distributions[0,1] = observation_transform.dot(state_space_mean[initial_number_of_states-1,1:3])

    # making it numba-ready
    last_predicted_covariance_matrix = np.zeros((2,2))
    for short_row_index, long_row_index in enumerate([initial_number_of_states-1,
                                                      total_number_of_states+initial_number_of_states-1]):
        for short_column_index, long_column_index in enumerate([initial_number_of_states -1,
                                                                total_number_of_states+initial_number_of_states-1]):
            last_predicted_covariance_matrix[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                 long_column_index]

    predicted_observation_distributions[0,2] = (observation_transform.dot(
                                                                     last_predicted_covariance_matrix).dot(observation_transform.transpose())
                                                                     +
                                                                     measurement_variance)

    ####################################################################
    ####################################################################
    ##
    ## initialise derivative arrays
    ##
    ####################################################################
    ####################################################################

    state_space_mean_derivative = np.zeros((total_number_of_states,7,2))

    repression_threshold = model_parameters[0]
    hill_coefficient = model_parameters[1]
    mRNA_degradation_rate = model_parameters[2]
    protein_degradation_rate = model_parameters[3]
    basal_transcription_rate = model_parameters[4]
    translation_rate = model_parameters[5]
    transcription_delay = model_parameters[6]

    steady_state_protein = state_space_mean[0,2]

    hill_function_value = 1.0/(1.0+np.power(steady_state_protein/repression_threshold,hill_coefficient))

    hill_function_derivative_value_wrt_protein = - hill_coefficient*np.power(steady_state_protein/repression_threshold,
                                                                             hill_coefficient - 1)/( repression_threshold*
                                                   np.power(1.0+np.power( steady_state_protein/repression_threshold,
                                                                          hill_coefficient),2))

    protein_derivative_denominator_scalar = (basal_transcription_rate*translation_rate)/(mRNA_degradation_rate*protein_degradation_rate)
    initial_protein_derivative_denominator = (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_protein) - 1

    # assign protein derivative first, since mRNA derivative is given as a function of protein derivative

    hill_function_derivative_value_wrt_repression = hill_coefficient*np.power(steady_state_protein/repression_threshold,
                                                                             hill_coefficient)/( repression_threshold*
                                                   np.power(1.0+np.power( steady_state_protein/repression_threshold,
                                                                          hill_coefficient),2))

    hill_function_derivative_value_wrt_hill_coefficient = np.log(steady_state_protein/repression_threshold)*np.power(steady_state_protein/repression_threshold,
                                                                 hill_coefficient)/( np.power(1.0+np.power( steady_state_protein/repression_threshold,
                                                                 hill_coefficient),2))

    state_space_mean_derivative[:initial_number_of_states,0,1] = - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_repression/
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,0,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,0,1]

    state_space_mean_derivative[:initial_number_of_states,1,1] = - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_hill_coefficient/
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,1,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,1,1]

    state_space_mean_derivative[:initial_number_of_states,2,1] = (protein_derivative_denominator_scalar*hill_function_value/
                                                                  mRNA_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,2,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,2,1]

    state_space_mean_derivative[:initial_number_of_states,3,1] = (protein_derivative_denominator_scalar*hill_function_value/
                                                                  protein_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,3,0] = (steady_state_protein + protein_degradation_rate*state_space_mean_derivative[0,3,1])/translation_rate

    state_space_mean_derivative[:initial_number_of_states,4,1] = -(protein_derivative_denominator_scalar*hill_function_value/
                                                                   basal_transcription_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,4,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,4,1]

    state_space_mean_derivative[:initial_number_of_states,5,1] = -(protein_derivative_denominator_scalar*hill_function_value/
                                                                   translation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,5,0] = -(protein_degradation_rate/translation_rate)*(steady_state_protein/translation_rate -
                                                                                                               state_space_mean_derivative[0,5,1])

    state_space_mean_derivative[:initial_number_of_states,6,1] = 0
    state_space_mean_derivative[:initial_number_of_states,6,0] = 0

    state_space_variance_derivative = np.zeros((7,2*total_number_of_states,2*total_number_of_states))
    # update the past ("negative time")
    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                              state_space_variance,
                                                                                                                              state_space_mean_derivative,
                                                                                                                              state_space_variance_derivative,
                                                                                                                              protein_at_observations[0],
                                                                                                                              time_delay,
                                                                                                                              observation_time_step,
                                                                                                                              measurement_variance)

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions

# @jit(nopython = True)
def kalman_observation_distribution_parameters(predicted_observation_distributions,
                                               current_observation,
                                               state_space_mean,
                                               state_space_variance,
                                               current_number_of_states,
                                               total_number_of_states,
                                               measurement_variance,
                                               observation_index):
    """
    A function which updates the mean and variance for the distributions which describe the likelihood of
    our observations, given some model parameters.

    Parameters
    ----------

    predicted_observation_distributions : numpy array.
        An array of dimension n x 3 where n is the number of observation time points.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at each time point, respectively

    current_observation : int.
        Observed protein at the current time. The dimension is 1 x 2.
        The first column is the time, and the second column is the observed protein copy number at
        that time

    state_space_mean : numpy array
        An array of dimension n x 3, where n is the number of inferred time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein. Time points are generated every minute

    state_space_variance : numpy array.
        An array of dimension 2n x 2n.
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    current_number_of_states : float.
        The current number of (hidden and observed) states upto the current observation time point.
        This includes the initial states (with negative time).

    total_number_of_states : float.
        The total number of states that will be predicted by the kalman_filter function

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    observation_index : int.
        The index for the current observation time in the main kalman_filter loop

    Returns
    -------

    predicted_observation_distributions[observation_index + 1] : numpy array.
        An array of dimension 1 x 3.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at the current time point, respectively.
    """

    observation_transform = np.array([0.0,1.0])

    predicted_observation_distributions[observation_index+1,0] = current_observation[0]
    predicted_observation_distributions[observation_index+1,1] = observation_transform.dot(state_space_mean[current_number_of_states-1,1:3])

    # not using np.ix_-like indexing to make it numba-ready
    last_predicted_covariance_matrix = np.zeros((2,2))
    for short_row_index, long_row_index in enumerate([current_number_of_states-1,
                                                      total_number_of_states+current_number_of_states-1]):
        for short_column_index, long_column_index in enumerate([current_number_of_states -1,
                                                                total_number_of_states+current_number_of_states-1]):
            last_predicted_covariance_matrix[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                 long_column_index]

    predicted_observation_distributions[observation_index+1,2] = (observation_transform.dot(
                                                                     last_predicted_covariance_matrix).dot(observation_transform.transpose())
                                                                     +
                                                                     measurement_variance)

    return predicted_observation_distributions[observation_index + 1]

# @jit(nopython = True)
def kalman_prediction_step(state_space_mean,
                           state_space_variance,
                           state_space_mean_derivative,
                           state_space_variance_derivative,
                           current_observation,
                           model_parameters,
                           observation_time_step):
    """
    Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
    state space mean and variance. This gives rho_{t+\delta t-tau:t+\delta t} and P_{t+\delta t-tau:t+\delta t},
    using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
    approximated using a forward Euler scheme.

    TODO: update variable descriptions
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

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    current_observation : numpy array.
        The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

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

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]
    """
    discretisation_time_step = 1.0
    ## name the model parameters
    repression_threshold = model_parameters[0]
    hill_coefficient = model_parameters[1]
    mRNA_degradation_rate = model_parameters[2]
    protein_degradation_rate = model_parameters[3]
    basal_transcription_rate = model_parameters[4]
    translation_rate = model_parameters[5]
    transcription_delay = model_parameters[6]

    discrete_delay = int(np.around(transcription_delay/discretisation_time_step))

    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))

    # this is the number of states at t, i.e. before predicting towards t+observation_time_step
    current_number_of_states = (int(np.around(current_observation[0]/observation_time_step))-1)*number_of_hidden_states + discrete_delay+1
    total_number_of_states = state_space_mean.shape[0]

    ## next_time_index corresponds to 't+Deltat' in the propagation equation on page 5 of the supplementary
    ## material in the calderazzo paper
    for next_time_index in range(current_number_of_states, current_number_of_states + number_of_hidden_states):
        current_time_index = next_time_index - 1 # this corresponds to t
        past_time_index = current_time_index - discrete_delay # this corresponds to t-tau
        # indexing with 1:3 for numba
        current_mean = state_space_mean[current_time_index,1:3]
        past_protein = state_space_mean[past_time_index,2]

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
        # indexing with 1:3 for numba
        state_space_mean[next_time_index,1:3] = next_mean

        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        current_covariance_matrix = np.zeros((2,2))
        for short_row_index, long_row_index in enumerate([current_time_index,
                                                          total_number_of_states+current_time_index]):
            for short_column_index, long_column_index in enumerate([current_time_index,
                                                                    total_number_of_states+current_time_index]):
                current_covariance_matrix[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]

        # this is P(t-\tau,t) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_past_to_now = np.zeros((2,2))
        for short_row_index, long_row_index in enumerate([past_time_index,
                                                          total_number_of_states+past_time_index]):
            for short_column_index, long_column_index in enumerate([current_time_index,
                                                                    total_number_of_states+current_time_index]):
                covariance_matrix_past_to_now[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]

        # this is P(t,t-\tau) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_now_to_past = np.zeros((2,2))
        for short_row_index, long_row_index in enumerate([current_time_index,
                                                          total_number_of_states+current_time_index]):
            for short_column_index, long_column_index in enumerate([past_time_index,
                                                                    total_number_of_states+past_time_index]):
                covariance_matrix_now_to_past[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]
        # derivations for the following are found in Calderazzo et. al. (2018)
        # g is [[-mRNA_degradation_rate,0],                  *[M(t),
        #       [translation_rate,-protein_degradation_rate]] [P(t)]
        # and its derivative will be called instant_jacobian
        # f is [[basal_transcription_rate*hill_function(past_protein)],0]
        # and its derivative with respect to the past state will be called delayed_jacobian
        # the matrix A in the paper will be called variance_of_noise
        instant_jacobian = np.array([[-mRNA_degradation_rate,0.0],[translation_rate,-protein_degradation_rate]])
        # jacobian of f is derivative of f with respect to past state ([past_mRNA, past_protein])
        delayed_jacobian = np.array([[0.0,basal_transcription_rate*hill_function_derivative_value],[0.0,0.0]])

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

        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        for short_row_index, long_row_index in enumerate([next_time_index,
                                                          total_number_of_states+next_time_index]):
            for short_column_index, long_column_index in enumerate([next_time_index,
                                                                    total_number_of_states+next_time_index]):
                state_space_variance[long_row_index,long_column_index] = next_covariance_matrix[short_row_index,
                                                                                                short_column_index]

        ## now we need to update the cross correlations, P(s,t) in the Calderazzo paper
        # the range needs to include t, since we want to propagate P(t,t) into P(t,t+Deltat)
        for intermediate_time_index in range(past_time_index,current_time_index+1):
            # This corresponds to P(s,t) in the Calderazzo paper
            # for loops instead of np.ix_-like indexing
            covariance_matrix_intermediate_to_current = np.zeros((2,2))
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([current_time_index,
                                                                        total_number_of_states+current_time_index]):
                    covariance_matrix_intermediate_to_current[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                                         long_column_index]
            # This corresponds to P(s,t-tau)
            covariance_matrix_intermediate_to_past = np.zeros((2,2))
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([past_time_index,
                                                                        total_number_of_states+past_time_index]):
                    covariance_matrix_intermediate_to_past[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                                         long_column_index]

            covariance_derivative = ( covariance_matrix_intermediate_to_current.dot( np.transpose(instant_jacobian)) +
                                      covariance_matrix_intermediate_to_past.dot( np.transpose(delayed_jacobian)))

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next = covariance_matrix_intermediate_to_current + discretisation_time_step*covariance_derivative

            # Fill in the big matrix
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([next_time_index,
                                                                        total_number_of_states+next_time_index]):
                    state_space_variance[long_row_index,long_column_index] = covariance_matrix_intermediate_to_current[short_row_index,
                                                                                                                       short_column_index]
            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            for short_row_index, long_row_index in enumerate([next_time_index,
                                                              total_number_of_states+next_time_index]):
                for short_column_index, long_column_index in enumerate([intermediate_time_index,
                                                                        total_number_of_states+intermediate_time_index]):
                    state_space_variance[long_row_index,long_column_index] = covariance_matrix_intermediate_to_current[short_column_index,
                                                                                                                       short_row_index]

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative

# @jit(nopython = True)
def kalman_update_step(state_space_mean,
                       state_space_variance,
                       state_space_mean_derivative,
                       state_space_variance_derivative,
                       current_observation,
                       time_delay,
                       observation_time_step,
                       measurement_variance):
    """
    Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
    This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
    This assumes that the observations are collected at fixed time intervals.

    TODO: update variable descriptions
    Parameters
    ----------

    state_space_mean : numpy array.
        The dimension is n x 3, where n is the number of states until the current time.
        The first column is time, the second column is mean mRNA, and the third column is mean protein.

    state_space_variance : numpy array.
        The dimension is 2n x 2n, where n is the number of states until the current time.
            [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
              cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    current_observation : numpy array.
        The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

    time_delay : float.
        The fixed transciptional time delay in the system. This tells us how far back we need to update our
        state space estimates.

    observation_time_step : float.
        The fixed time interval between protein observations.

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
    discretisation_time_step = state_space_mean[1,0] - state_space_mean[0,0]

    discrete_delay = int(np.around(time_delay/discretisation_time_step))
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))

    # this is the number of states at t+Deltat, i.e. after predicting towards t+observation_time_step
    current_number_of_states = (int(np.around(current_observation[0]/observation_time_step)))*number_of_hidden_states + discrete_delay+1

    total_number_of_states = state_space_mean.shape[0]

    # predicted_state_space_mean until delay, corresponds to
    # rho(t+Deltat-delay:t+deltat). Includes current value and discrete_delay past values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    shortened_state_space_mean = state_space_mean[current_number_of_states-(discrete_delay+1):current_number_of_states,1:3]

    # put protein values underneath mRNA values, to make vector of means (rho)
    # consistent with variance (P)
    stacked_state_space_mean = np.hstack((shortened_state_space_mean[:,0],
                                          shortened_state_space_mean[:,1]))

    # funny indexing with 1:3 instead of (1,2) to make numba happy
    predicted_final_state_space_mean = state_space_mean[current_number_of_states-1,1:3]

    # extract covariance matrix up to delay
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    mRNA_indices_to_keep = np.arange(current_number_of_states - discrete_delay - 1,current_number_of_states,1)
    protein_indices_to_keep = np.arange(total_number_of_states + current_number_of_states - discrete_delay - 1,total_number_of_states + current_number_of_states,1)
    all_indices_up_to_delay = np.hstack((mRNA_indices_to_keep, protein_indices_to_keep))

    # using for loop indexing for numba
    shortened_covariance_matrix = np.zeros((all_indices_up_to_delay.shape[0],all_indices_up_to_delay.shape[0]))
    for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
        for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
            shortened_covariance_matrix[shortened_row_index,shortened_column_index] = state_space_variance[long_row_index,
                                                                                                           long_column_index]
    # extract P(t+Deltat-delay:t+deltat,t+Deltat), replacing ((discrete_delay),-1) with a splice for numba
    shortened_covariance_matrix_past_to_final = shortened_covariance_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)]

    # and P(t+Deltat,t+Deltat-delay:t+deltat), replacing ((discrete_delay),-1) with a splice for numba
    shortened_covariance_matrix_final_to_past = shortened_covariance_matrix[discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:]

    # This is F in the paper
    observation_transform = np.array([0.0,1.0])

    # This is P(t+Deltat,t+Deltat) in the paper
    # using np.ix_-like indexing
#     predicted_final_covariance_matrix = state_space_variance[[[current_number_of_states-1],[total_number_of_states+current_number_of_states-1]],
#                                                               [[current_number_of_states-1,total_number_of_states+current_number_of_states-1]]]
    # funny indexing to get numba to work properly
    predicted_final_covariance_matrix = np.zeros((2,2))
    for short_row_index, long_row_index in enumerate([current_number_of_states-1,
                                                      total_number_of_states+current_number_of_states-1]):
        for short_column_index, long_column_index in enumerate([current_number_of_states-1,
                                                                    total_number_of_states+current_number_of_states-1]):
            predicted_final_covariance_matrix[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                         long_column_index]

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
    # print(updated_stacked_state_space_mean.shape)
    # ensures the the mean mRNA and Protein are non negative
    updated_stacked_state_space_mean = np.maximum(updated_stacked_state_space_mean,0)

    # unstack the rho into two columns, one with mRNA and one with protein
    updated_state_space_mean = np.column_stack((updated_stacked_state_space_mean[:(discrete_delay+1)],
                                                updated_stacked_state_space_mean[(discrete_delay+1):]))
    # Fill in the updated values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    state_space_mean[current_number_of_states-(discrete_delay+1):current_number_of_states,1:3] = updated_state_space_mean

    # This is P*
    updated_shortened_covariance_matrix = ( shortened_covariance_matrix -
                                            np.dot(adaptation_coefficient.reshape((2*(discrete_delay+1),1)),observation_transform.reshape((1,2))).dot(
                                                shortened_covariance_matrix_final_to_past))

    # ensure that the diagonal entries are non negative
    np.fill_diagonal(updated_shortened_covariance_matrix,np.maximum(np.diag(updated_shortened_covariance_matrix),0))

    # Fill in updated values
    # replacing the following line with a loop for numba
    # state_space_variance[all_indices_up_to_delay,
    #                    all_indices_up_to_delay.transpose()] = updated_shortened_covariance_matrix
    for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
        for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
            state_space_variance[long_row_index,long_column_index] = updated_shortened_covariance_matrix[shortened_row_index,
                                                                                                         shortened_column_index]

    ##########################################
    ## derivative updates
    ##########################################

    # funny indexing with 0:2 instead of (0,1) to make numba happy
    shortened_state_space_mean_derivative = state_space_mean_derivative[current_number_of_states-(discrete_delay+1):current_number_of_states,:,0:2]

    # put protein values underneath mRNA values, to make vector of mean derivatives (d_rho/d_theta)
    # consistent with variance (P)
    stacked_state_space_mean_derivative = np.zeros((7,2*(discrete_delay+1)))

    # this gives us 7 rows (one for each parameter) of mRNA derivative values over time, followed by protein derivative values over time
    for parameter_index in range(7):
        stacked_state_space_mean_derivative[parameter_index] = np.hstack((shortened_state_space_mean_derivative[:,parameter_index,0],
                                                                          shortened_state_space_mean_derivative[:,parameter_index,1]))

    # funny indexing with 0:2 instead of (0,1) to make numba happy (this gives a 7 x 2 numpy array)
    predicted_final_state_space_mean_derivative = state_space_mean_derivative[current_number_of_states-1,:,0:2]

    # extract covariance derivative matrix up to delay
    # corresponds to d_P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)/d_theta
    mRNA_indices_to_keep = np.arange(current_number_of_states - discrete_delay - 1,current_number_of_states,1)
    protein_indices_to_keep = np.arange(total_number_of_states + current_number_of_states - discrete_delay - 1,total_number_of_states + current_number_of_states,1)
    all_indices_up_to_delay = np.hstack((mRNA_indices_to_keep, protein_indices_to_keep))

    # using for loop indexing for numba
    shortened_covariance_derivative_matrix = np.zeros((7,all_indices_up_to_delay.shape[0],all_indices_up_to_delay.shape[0]))
    for parameter_index in range(7):
        for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
            for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
                shortened_covariance_derivative_matrix[parameter_index,shortened_row_index,shortened_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                     long_row_index,
                                                                                                                                                     long_column_index]
    # extract d_P(t+Deltat-delay:t+deltat,t+Deltat)/d_theta, replacing ((discrete_delay),-1) with a splice for numba
    shortened_covariance_derivative_matrix_past_to_final = shortened_covariance_derivative_matrix[:,:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)]

    # and d_P(t+Deltat,t+Deltat-delay:t+deltat)/d_theta, replacing ((discrete_delay),-1) with a splice for numba
    shortened_covariance_derivative_matrix_final_to_past = shortened_covariance_derivative_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:]

    # This is the derivative of P(t+Deltat,t+Deltat) in the paper
    # using np.ix_-like indexing
    # predicted_final_covariance_derivative_matrix = np.zeros((7,2,2))
    # for parameter_index in range(7):
    #     predicted_final_covariance_derivative_matrix[parameter_index] = state_space_variance_derivative[parameter_index][[[current_number_of_states-1],
    #                                                                                                                      [total_number_of_states+current_number_of_states-1]],
    #                                                                                                                      [[current_number_of_states-1,total_number_of_states+current_number_of_states-1]]]
    # funny indexing to get numba to work properly
    predicted_final_covariance_derivative_matrix = np.zeros((7,2,2))
    for parameter_index in range(7):
        for short_row_index, long_row_index in enumerate([current_number_of_states-1,
                                                         total_number_of_states+current_number_of_states-1]):
            for short_column_index, long_column_index in enumerate([current_number_of_states-1,
                                                                    total_number_of_states+current_number_of_states-1]):
                predicted_final_covariance_derivative_matrix[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                   long_row_index,
                                                                                                                                                   long_column_index]

    # need derivative of the adaptation_coefficient
    # observation_transform = observation_transform.reshape((1,2))
    adaptation_coefficient_derivative = np.zeros((7,all_indices_up_to_delay.shape[0]))
    for parameter_index in range(7):
        adaptation_coefficient_derivative[parameter_index] = (shortened_covariance_derivative_matrix_past_to_final[parameter_index].dot(np.transpose(observation_transform))*helper_inverse -
                                                             (shortened_covariance_matrix_past_to_final[parameter_index].dot(np.transpose(observation_transform).dot(observation_transform.dot(
                                                              predicted_final_covariance_derivative_matrix[parameter_index].dot(np.transpose(observation_transform))))))*np.power(helper_inverse,2) )

    # This is d_rho*
    updated_stacked_state_space_mean_derivative = np.zeros((7,2*(discrete_delay+1)))
    for parameter_index in range(7):
        updated_stacked_state_space_mean_derivative[parameter_index] = ( stacked_state_space_mean_derivative[parameter_index] +
                                                                         adaptation_coefficient_derivative[parameter_index]*(current_observation[1] -
                                                                         observation_transform.dot(predicted_final_state_space_mean)) -
                                                                         adaptation_coefficient.dot(observation_transform.dot(
                                                                         predicted_final_state_space_mean_derivative[parameter_index])) )

    # unstack the rho into two columns, one with mRNA and one with protein

    updated_state_space_mean_derivative = np.zeros(((discrete_delay+1),7,2))
    for parameter_index in range(7):
        updated_state_space_mean_derivative[:,parameter_index,:] = np.column_stack((updated_stacked_state_space_mean_derivative[parameter_index,:(discrete_delay+1)],
                                                                                    updated_stacked_state_space_mean_derivative[parameter_index,(discrete_delay+1):]))

    # Fill in the updated values
    # funny indexing with 0:2 instead of (0,1) to make numba happy
    state_space_mean_derivative[current_number_of_states-(discrete_delay+1):current_number_of_states,:,0:2] = updated_state_space_mean_derivative

    # This is d_P*/d_theta
    updated_shortened_covariance_derivative_matrix = np.zeros((7,all_indices_up_to_delay.shape[0],all_indices_up_to_delay.shape[0]))
    for parameter_index in range(7):
        updated_shortened_covariance_derivative_matrix[parameter_index] = ( shortened_covariance_derivative_matrix[parameter_index] -
                                                                            np.dot(adaptation_coefficient_derivative[parameter_index].reshape((2*(discrete_delay+1),1)),
                                                                                   observation_transform.reshape((1,2))).dot(shortened_covariance_matrix_final_to_past) -
                                                                            np.dot(adaptation_coefficient.reshape((2*(discrete_delay+1),1)),
                                                                                   observation_transform.reshape((1,2))).dot(shortened_covariance_derivative_matrix_final_to_past[parameter_index]))
    # Fill in updated values
    # replacing the following line with a loop for numba
    # state_space_variance[all_indices_up_to_delay,
    #                    all_indices_up_to_delay.transpose()] = updated_shortened_covariance_matrix
    for parameter_index in range(7):
        for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
            for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
                state_space_variance_derivative[parameter_index,long_row_index,long_column_index] = updated_shortened_covariance_derivative_matrix[parameter_index,
                                                                                                                                                   shortened_row_index,
                                                                                                                                                   shortened_column_index]

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative

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
        An array containing the moderowl parameters in the following order:
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

    _, _, _, _, predicted_observation_distributions = kalman_filter(protein_at_observations,
                                                              model_parameters,
                                                              measurement_variance)
    observations = protein_at_observations[:,1]
    mean = predicted_observation_distributions[:,1]
    sd = np.sqrt(predicted_observation_distributions[:,2])

    log_likelihood = np.sum(norm.logpdf(observations,mean,sd))

    return log_likelihood


def kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,acceptance_tuner,parameter_covariance,initial_state,**kwargs):
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
        A real number between 0 and 1 which tells us the proportion of accepted proposal parameters. Optimal theoretical
        value is 0.234.

    acceptance_tuner : float.
        A positive real number which determines the length of our step size in the random walk.
    """
    np.random.seed()
    # define set of valid optional inputs, and raise error if not valid
    valid_kwargs = ['adaptive']
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs:" " %s" % unknown_kwargs)

    identity = np.identity(5)
    cholesky_covariance = np.linalg.cholesky(parameter_covariance+0.000001*identity)
    #print(cholesky_covariance)

    number_of_hyper_parameters = hyper_parameters.shape[0]
    shape = hyper_parameters[0:number_of_hyper_parameters:2]
    scale = hyper_parameters[1:number_of_hyper_parameters:2]
    current_state = initial_state

    # We perform likelihood calculations in a separate process which is managed by a process pool
    # this is necessary to prevent memory overflow due to memory fragmentation
    likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)

    random_walk = np.zeros((iterations,7))
    random_walk[0,:] = current_state
    reparameterised_current_state = np.copy(current_state)
    reparameterised_current_state[[4,5]] = np.power(10,current_state[[4,5]])
    current_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,reparameterised_current_state,measurement_variance)
    current_log_prior = np.sum(uniform.logpdf(current_state,loc=shape,scale=scale))
    acceptance_count = 0

    # if user chooses adaptive mcmc, the following will execute.
    if kwargs.get("adaptive") == "true":
        for step_index in range(1,iterations):
            # after 50000 iterations, update the proposal covariance every 1000 iterations using previous samples
            if step_index >= 50000:
                if np.mod(step_index,1000) == 0:
                    parameter_covariance = np.cov(random_walk[25000:step_index,(0,1,4,5,6)].T) + 0.0000000001*identity
                    cholesky_covariance  = np.linalg.cholesky(parameter_covariance)

            new_state = np.zeros(7)
            new_state[[2,3]] = np.array([np.log(2)/30,np.log(2)/90])
            new_state[[0,1,4,5,6]] = current_state[[0,1,4,5,6]] + (0.95*acceptance_tuner*cholesky_covariance.dot(multivariate_normal.rvs(size=5)) +
            (0.05*0.1*0.2)*multivariate_normal.rvs(size=5))

            new_log_prior = np.sum(uniform.logpdf(new_state,loc=shape,scale=scale))
            positive_new_parameters = new_state[[0,1,2,3,6]]
            if all(item > 0 for item in positive_new_parameters) == True and not new_log_prior == -np.inf:
                # reparameterise
                reparameterised_new_state            = np.copy(new_state)
                reparameterised_current_state        = np.copy(current_state)
                reparameterised_new_state[[4,5]]     = np.power(10,new_state[[4,5]])
                reparameterised_current_state[[4,5]] = np.power(10,current_state[[4,5]])

                try:
                    # in this line the pool returns an object of type mp.AsyncResult, which is not directly the likelihood,
                    # but which can be interrogated about the status of the calculation and so on
                    new_likelihood_result = likelihood_calculations_pool.apply_async(calculate_log_likelihood_at_parameter_point,
                                                                              args = (protein_at_observations,
                                                                                      reparameterised_new_state,
                                                                                      measurement_variance))

                    # ask the async result from above to return the new likelihood when it is ready
                    new_log_likelihood = new_likelihood_result.get(30)
                except ValueError:
                    print('value error!')
                    new_log_likelihood = -np.inf
                except mp.TimeoutError:
                    # import pdb; pdb.set_trace()
                    print('I have found a TimeoutError!')
                    likelihood_calculations_pool.close()
                    likelihood_calculations_pool.terminate()
                    likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)
                    new_likelihood_result = likelihood_calculations_pool.apply_async(calculate_log_likelihood_at_parameter_point,
                                                                              args = (protein_at_observations,
                                                                                      reparameterised_new_state,
                                                                                      measurement_variance))
                    new_log_likelihood = new_likelihood_result.get(30)

                if np.mod(step_index,500) == 0:
                    print('iteration number:',step_index)
                    print('current state:\n',current_state)
                    print('new log lik:', new_log_likelihood)
                    print('cur log lik:', current_log_likelihood)
                    print(float(acceptance_count)/step_index)

                acceptance_ratio = np.exp(new_log_prior + new_log_likelihood - current_log_prior - current_log_likelihood)

                if np.random.uniform() < acceptance_ratio:
                    current_state = new_state
                    current_log_likelihood = new_log_likelihood
                    current_log_prior = new_log_prior
                    acceptance_count += 1

            random_walk[step_index,:] = current_state
        acceptance_rate = float(acceptance_count)/iterations
#####################################################################################################################
    else:
        for step_index in range(1,iterations):
            new_state = np.zeros(7)
            new_state[[2,3]] = np.array([np.log(2)/30,np.log(2)/90])
            new_state[[0,1,4,5,6]] = current_state[[0,1,4,5,6]] + acceptance_tuner*cholesky_covariance.dot(multivariate_normal.rvs(size=5))

            positive_new_parameters = new_state[[0,1,2,3,6]]
            if all(item > 0 for item in positive_new_parameters) == True:
                new_log_prior = np.sum(uniform.logpdf(new_state,loc=shape,scale=scale))

                # reparameterise
                reparameterised_new_state            = np.copy(new_state)
                reparameterised_current_state        = np.copy(current_state)
                reparameterised_new_state[[4,5]]     = np.power(10,new_state[[4,5]])
                reparameterised_current_state[[4,5]] = np.power(10,current_state[[4,5]])

                try:
                    # in this line the pool returns an object of type mp.AsyncResult, which is not directly the likelihood,
                    # but which can be interrogated about the status of the calculation and so on
                    new_likelihood_result = likelihood_calculations_pool.apply_async(calculate_log_likelihood_at_parameter_point,
                                                                              args = (protein_at_observations,
                                                                                      reparameterised_new_state,
                                                                                      measurement_variance))

                    # ask the async result from above to return the new likelihood when it is ready
                    new_log_likelihood = new_likelihood_result.get(30)
                except ValueError:
                    new_log_likelihood = -np.inf
                except mp.TimeoutError:
                    likelihood_calculations_pool.close()
                    likelihood_calculations_pool.terminate()
                    likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)

                acceptance_ratio = np.exp(new_log_prior + new_log_likelihood - current_log_prior - current_log_likelihood)

                if np.mod(step_index,500) == 0:
                    print('iteration number:',step_index)
                    print('current state:\n',current_state)
                    print('new log lik:', new_log_likelihood)
                    print('cur log lik:', current_log_likelihood)
                    print(float(acceptance_count)/step_index)

                if np.random.uniform() < acceptance_ratio:
                    current_state = new_state
                    current_log_likelihood = new_log_likelihood
                    current_log_prior = new_log_prior
                    acceptance_count += 1

            random_walk[step_index,:] = current_state
        acceptance_rate = float(acceptance_count)/iterations
    return random_walk, acceptance_rate, acceptance_tuner

def calculate_langevin_summary_statistics_at_parameter_point(parameter_values, number_of_traces = 100):
    '''Calculate the mean, relative standard deviation, period, coherence and mean mRNA
    of protein traces at one parameter point using the langevin equation.
    Will assume the arguments to be of the order described in
    generate_prior_samples. This function is necessary to ensure garbage collection of
    unnecessary traces.

    Parameters
    ----------

    parameter_values : ndarray
        each row contains one model parameter set in the order
        (repression_threshold, hill_coefficient, mRNA_degradation_rate, protein_degradation_rate,
         basal_transcription_rate, translation_rate, transcriptional_delay)

    number_of_traces_per_sample : int
        number of traces that should be run per sample to calculate the summary statistics

    Returns
    -------

    summary_statistics : ndarray
        One dimension, five entries. Contains the summary statistics (mean, std, period, coherence, mean_mRNA) for the parameters
        in parameter_values
    '''
    these_mrna_traces, these_protein_traces = hes5.generate_multiple_langevin_trajectories(number_of_traces, # number_of_trajectories
                                                                                           1500*5, #duration
                                                                                           parameter_values[0], #repression_threshold,
                                                                                           parameter_values[1], #hill_coefficient,
                                                                                           parameter_values[2], #mRNA_degradation_rate,
                                                                                           parameter_values[3], #protein_degradation_rate,
                                                                                           parameter_values[4], #basal_transcription_rate,
                                                                                           parameter_values[5], #translation_rate,
                                                                                           parameter_values[6], #transcription_delay,
                                                                                           0, #initial_mRNA,
                                                                                           0, #initial_protein,
                                                                                           1000) #equilibration_time

    this_deterministic_trace = hes5.generate_deterministic_trajectory(1500*5+1000, #duration
                                                                parameter_values[0], #repression_threshold,
                                                                parameter_values[1], #hill_coefficient,
                                                                parameter_values[2], #mRNA_degradation_rate,
                                                                parameter_values[3], #protein_degradation_rate,
                                                                parameter_values[4], #basal_transcription_rate,
                                                                parameter_values[5], #translation_rate,
                                                                parameter_values[6], #transcription_delay,
                                                                0,
                                                                0,
                                                                for_negative_times = 'no_negative')

    this_deterministic_trace = this_deterministic_trace[this_deterministic_trace[:,0]>1000] # remove equilibration time
#     this_deterministic_trace = np.vstack((these_protein_traces[:,0],
#                                           these_mrna_traces[:,1],
#                                           these_protein_traces[:,1])).transpose()
    summary_statistics = np.zeros(9)
    _,this_coherence, this_period = hes5.calculate_power_spectrum_of_trajectories(these_protein_traces)
    this_mean = np.mean(these_protein_traces[:,1:])
    this_std = np.std(these_protein_traces[:,1:])/this_mean
    this_mean_mRNA = np.mean(these_mrna_traces[:,1:])
    this_deterministic_mean = np.mean(this_deterministic_trace[:,2])
    this_deterministic_std = np.std(this_deterministic_trace[:,2])/this_deterministic_mean
    deterministic_protein_trace = np.vstack((this_deterministic_trace[:,0] - 1000,
                                            this_deterministic_trace[:,2])).transpose()
    _,this_deterministic_coherence, this_deterministic_period = hes5.calculate_power_spectrum_of_trajectories(deterministic_protein_trace)
    summary_statistics[0] = this_mean
    summary_statistics[1] = this_std
    summary_statistics[2] = this_period
    summary_statistics[3] = this_coherence
    summary_statistics[4] = this_mean_mRNA
    summary_statistics[5] = this_deterministic_mean
    summary_statistics[6] = this_deterministic_std
    summary_statistics[7] = this_deterministic_period
    summary_statistics[8] = this_deterministic_coherence

    return summary_statistics

def calculate_langevin_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 100,
                                                         number_of_cpus = 12):
    '''Calculate the mean, relative standard deviation, period, coherence, and mean mrna
    of protein traces at each parameter point in parameter_values.

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
    summary_statistics = np.zeros((parameter_values.shape[0], 9))

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
