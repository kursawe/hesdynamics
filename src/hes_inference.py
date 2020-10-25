import math
import numpy as np
import hes5
from numpy import number

from numba import jit
# suppresses annoying performance warnings about np.dot() being
# faster on contiguous arrays. should look at fixing it but this
# is good for now
from numba.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from scipy.stats import gamma, multivariate_normal, uniform
import multiprocessing as mp

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

    predicted_observation_mean_derivatives : numpy array.
        An array of dimension n x m x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space mean at each observation time point, wrt each parameter

    predicted_observation_variance_derivatives : numpy array.
        An array of dimension n x m x 2 x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space variance at each observation time point, wrt each parameter
    """
    time_delay = model_parameters[6]

    if protein_at_observations.reshape(-1,2).shape[0] == 1:
        number_of_observations = 1.0
        observation_time_step = 10.0
    else:
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

    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = kalman_filter_state_space_initialisation(protein_at_observations,
                                                                                                                                                                                                                                                                             model_parameters,
                                                                                                                                                                                                                                                                             measurement_variance)
    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    # for observation_index, current_observation in enumerate(protein_at_observations[1:]):
    for observation_index in range(len(protein_at_observations)-1):
        if number_of_observations != 1:
            current_observation = protein_at_observations[1+observation_index,:]
            state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_prediction_step(state_space_mean,
                                                                                                                                      state_space_variance,
                                                                                                                                      state_space_mean_derivative,
                                                                                                                                      state_space_variance_derivative,
                                                                                                                                      current_observation,
                                                                                                                                      model_parameters,
                                                                                                                                      observation_time_step)

            current_number_of_states = int(np.around(current_observation[0]/observation_time_step))*number_of_hidden_states + initial_number_of_states

        # between the prediction and update steps we record the mean and sd for our likelihood, and the derivatives of the mean and variance for the
        # derivative of the likelihood wrt the parameters
            predicted_observation_distributions[observation_index + 1] = kalman_observation_distribution_parameters(predicted_observation_distributions,
                                                                                                                    current_observation,
                                                                                                                    state_space_mean,
                                                                                                                    state_space_variance,
                                                                                                                    current_number_of_states,
                                                                                                                    total_number_of_states,
                                                                                                                    measurement_variance,
                                                                                                                    observation_index)

            predicted_observation_mean_derivatives[observation_index + 1], predicted_observation_variance_derivatives[observation_index + 1] = kalman_observation_derivatives(predicted_observation_mean_derivatives,
                                                                                                                                                                          predicted_observation_variance_derivatives,
                                                                                                                                                                          current_observation,
                                                                                                                                                                          state_space_mean_derivative,
                                                                                                                                                                          state_space_variance_derivative,
                                                                                                                                                                          current_number_of_states,
                                                                                                                                                                          total_number_of_states,
                                                                                                                                                                          observation_index)

            state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                                  state_space_variance,
                                                                                                                                  state_space_mean_derivative,
                                                                                                                                  state_space_variance_derivative,
                                                                                                                                  current_observation,
                                                                                                                                  time_delay,
                                                                                                                                  observation_time_step,
                                                                                                                                  measurement_variance)
    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives

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
        of the distribution of the expected observations at each time point, respectively

    predicted_observation_mean_derivatives : numpy array.
        An array of dimension n x m x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space mean at each observation time point, wrt each parameter

    predicted_observation_variance_derivatives : numpy array.
        An array of dimension n x m x 2 x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space variance at each observation time point, wrt each parameter
    """
    time_delay = model_parameters[6]

    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = int(np.around(time_delay/discretisation_time_step))

    if protein_at_observations.reshape(-1,2).shape[0] == 1:
        observation_time_step = 10.0
        number_of_observations = 1
    else:
        observation_time_step = protein_at_observations[1,0]-protein_at_observations[0,0]
        number_of_observations = protein_at_observations.shape[0]

    # 'synthetic' observations, which allow us to update backwards in time
    number_of_hidden_states = int(np.around(observation_time_step/discretisation_time_step))

    ## initialise "negative time" with the mean and standard deviations of the LNA
    initial_number_of_states = discrete_delay + 1
    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states

    state_space_mean = np.zeros((total_number_of_states,3))
    state_space_mean[:initial_number_of_states,(1,2)] = hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[0],
                                                                                                   hill_coefficient=model_parameters[1],
                                                                                                   mRNA_degradation_rate=model_parameters[2],
                                                                                                   protein_degradation_rate=model_parameters[3],
                                                                                                   basal_transcription_rate=model_parameters[4],
                                                                                                   translation_rate=model_parameters[5])

    if protein_at_observations.reshape(-1,2).shape[0] == 1:
        final_observation_time = 0
    else:
        final_observation_time = protein_at_observations[-1,0]
    # assign time entries
    state_space_mean[:,0] = np.linspace(-time_delay,final_observation_time,total_number_of_states)

    # initialise initial covariance matrix
    state_space_variance = np.zeros((2*(total_number_of_states),2*(total_number_of_states)))

    # set the mRNA variance at nagative times to the LNA approximation
    # LNA_mRNA_variance = np.power(hes5.calculate_approximate_mRNA_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
    #                                                                                                hill_coefficient=model_parameters[1],
    #                                                                                                mRNA_degradation_rate=model_parameters[2],
    #                                                                                                protein_degradation_rate=model_parameters[3],
    #                                                                                                basal_transcription_rate=model_parameters[4],
    #                                                                                                translation_rate=model_parameters[5],
    #                                                                                                transcription_delay=model_parameters[6]),2)
    # the top left block of the matrix corresponds to the mRNA covariance, see docstring above
    initial_mRNA_scaling = 20.0
    initial_mRNA_variance = state_space_mean[0,1]*initial_mRNA_scaling
    np.fill_diagonal( state_space_variance[:initial_number_of_states,:initial_number_of_states] , initial_mRNA_variance)

    # set the protein variance at nagative times to the LNA approximation
    # LNA_protein_variance = np.power(hes5.calculate_approximate_protein_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
    #                                                                                                hill_coefficient=model_parameters[1],
    #                                                                                                mRNA_degradation_rate=model_parameters[2],
    #                                                                                                protein_degradation_rate=model_parameters[3],
    #                                                                                                basal_transcription_rate=model_parameters[4],
    #                                                                                                translation_rate=model_parameters[5],
    #                                                                                                transcription_delay=model_parameters[6]),2)
    # # the bottom right block of the matrix corresponds to the mRNA covariance, see docstring above
    initial_protein_scaling = 100.0
    initial_protein_variance = state_space_mean[0,2]*initial_protein_scaling
    np.fill_diagonal( state_space_variance[total_number_of_states:total_number_of_states + initial_number_of_states,
                                           total_number_of_states:total_number_of_states + initial_number_of_states] , initial_protein_variance )
    # potential solution for numba:
#     np.fill_diagonal( state_space_variance[total_number_of_states:total_number_of_states + initial_number_of_states,
#                                             total_number_of_states:total_number_of_states + initial_number_of_states] , 1.0 )

    observation_transform = np.array([0.0,1.0])

    predicted_observation_distributions = np.zeros((number_of_observations,3))
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
    #
    state_space_mean_derivative = np.zeros((total_number_of_states,7,2))
    #
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

    hill_function_derivative_value_wrt_hill_coefficient = - np.log(steady_state_protein/repression_threshold)*np.power(steady_state_protein/repression_threshold,
                                                                 hill_coefficient)/( np.power(1.0+np.power( steady_state_protein/repression_threshold,
                                                                 hill_coefficient),2))
    # repression threshold
    state_space_mean_derivative[:initial_number_of_states,0,1] = - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_repression)/(
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,0,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,0,1]

    # hill coefficient
    state_space_mean_derivative[:initial_number_of_states,1,1] = - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_hill_coefficient)/(
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,1,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,1,1]

    # mRNA degradation
    state_space_mean_derivative[:initial_number_of_states,2,1] = (protein_derivative_denominator_scalar*hill_function_value)/(
                                                                  mRNA_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,2,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,2,1]

    # protein degradation
    state_space_mean_derivative[:initial_number_of_states,3,1] = (protein_derivative_denominator_scalar*hill_function_value)/(
                                                                  protein_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,3,0] = (steady_state_protein + protein_degradation_rate*state_space_mean_derivative[0,3,1])/translation_rate

    # basal transcription
    state_space_mean_derivative[:initial_number_of_states,4,1] = -(protein_derivative_denominator_scalar*hill_function_value)/(
                                                                   basal_transcription_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,4,0] = (protein_degradation_rate/translation_rate)*state_space_mean_derivative[0,4,1]

    # translation
    state_space_mean_derivative[:initial_number_of_states,5,1] = -(protein_derivative_denominator_scalar*hill_function_value)/(
                                                                   translation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[:initial_number_of_states,5,0] = -(protein_degradation_rate/translation_rate)*((steady_state_protein/translation_rate) -
                                                                                                               state_space_mean_derivative[0,5,1])
    # transcriptional delay
    state_space_mean_derivative[:initial_number_of_states,6,1] = 0
    state_space_mean_derivative[:initial_number_of_states,6,0] = 0

    state_space_variance_derivative = np.zeros((7,2*total_number_of_states,2*total_number_of_states))
    for parameter_index in range(7):
        np.fill_diagonal(state_space_variance_derivative[parameter_index,:initial_number_of_states,:initial_number_of_states],
                         initial_mRNA_scaling*state_space_mean_derivative[0,parameter_index,0])
        np.fill_diagonal(state_space_variance_derivative[parameter_index,
                                                         total_number_of_states:total_number_of_states + initial_number_of_states,
                                                         total_number_of_states:total_number_of_states + initial_number_of_states],
                         initial_protein_scaling*state_space_mean_derivative[0,parameter_index,1])

    predicted_observation_mean_derivatives = np.zeros((number_of_observations,7,2))
    predicted_observation_mean_derivatives[0] = state_space_mean_derivative[initial_number_of_states-1]

    predicted_observation_variance_derivatives = np.zeros((number_of_observations,7,2,2))
    for parameter_index in range(7):
        for short_row_index, long_row_index in enumerate([initial_number_of_states-1,
                                                          total_number_of_states+initial_number_of_states-1]):
            for short_column_index, long_column_index in enumerate([initial_number_of_states -1,
                                                                    total_number_of_states+initial_number_of_states-1]):
                predicted_observation_variance_derivatives[0,parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                   long_row_index,
                                                                                                                                                   long_column_index]

    # update the past ("negative time")
    if protein_at_observations.reshape(-1,2).shape[0] == 1:
        current_observation = protein_at_observations
    else:
        current_observation = protein_at_observations[0]
    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                                  state_space_variance,
                                                                                                                                  state_space_mean_derivative,
                                                                                                                                  state_space_variance_derivative,
                                                                                                                                  current_observation,
                                                                                                                                  time_delay,
                                                                                                                                  observation_time_step,
                                                                                                                                  measurement_variance)

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives

@jit(nopython = True)
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

@jit(nopython = True)
def kalman_observation_derivatives(predicted_observation_mean_derivatives,
                                   predicted_observation_variance_derivatives,
                                   current_observation,
                                   state_space_mean_derivative,
                                   state_space_variance_derivative,
                                   current_number_of_states,
                                   total_number_of_states,
                                   observation_index):
    """

    Parameters
    ----------

    predicted_observation_mean_derivatives : numpy array.
        An array of dimension n x m x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space mean at each observation time point, wrt each parameter

    predicted_observation_variance_derivatives : numpy array.
        An array of dimension n x m x 2 x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space variance at each observation time point, wrt each parameter

    current_observation : numpy array.
        A 1 x 2 array which describes the observation of protein at the current time point. The first
        column is time, and the second column is the protein level

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    current_number_of_states : float.
        The current number of (hidden and observed) states upto the current observation time point.
        This includes the initial states (with negative time).

    total_number_of_states : float.
        The total number of (observed and hidden) states, used to index the variance matrix

    observation_index : int.
        The index for the current observation time in the main kalman_filter loop

    Returns
    -------

        predicted_observation_mean_derivatives[observation_index + 1] : numpy array.
            An array of dimension 7 x 2, which contains the derivative of the mean mRNA
            and protein wrt each parameter at the current observation time point


        predicted_observation_variance_derivatives[observation_index + 1] : numpy array.
            An array of dimension 7 x 2 x 2, which describes the derivative of the state
            space variance wrt each parameter for the current time point
    """

    predicted_observation_mean_derivatives[observation_index+1] = state_space_mean_derivative[current_number_of_states-1]

    for parameter_index in range(7):
        for short_row_index, long_row_index in enumerate([current_number_of_states-1,
                                                          total_number_of_states+current_number_of_states-1]):
            for short_column_index, long_column_index in enumerate([current_number_of_states-1,
                                                                    total_number_of_states+current_number_of_states-1]):
                predicted_observation_variance_derivatives[observation_index+1,parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                   long_row_index,
                                                                                                                                                   long_column_index]

    return predicted_observation_mean_derivatives[observation_index + 1], predicted_observation_variance_derivatives[observation_index + 1]

@jit(nopython = True)
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
    # This is the time step dt in the forward euler scheme
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

    # we initialise all our matrices outside of the main for loop for improved performance
    # this is P(t,t)
    current_covariance_matrix = np.zeros((2,2))
    # this is P(t-\tau,t) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_past_to_now = np.zeros((2,2))
    # this is P(t,t-\tau) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_now_to_past = np.zeros((2,2))
    # This corresponds to P(s,t) in the Calderazzo paper
    covariance_matrix_intermediate_to_current = np.zeros((2,2))
    # This corresponds to P(s,t-tau)
    covariance_matrix_intermediate_to_past = np.zeros((2,2))

    # this is d_rho(t)/d_theta
    next_mean_derivative = np.zeros((7,2))
    # this is d_P(t,t)/d_theta
    current_covariance_derivative_matrix = np.zeros((7,2,2))
    # this is d_P(t-\tau,t)/d_theta
    covariance_derivative_matrix_past_to_now = np.zeros((7,2,2))
    # this is d_P(t,t-\tau)/d_theta
    covariance_derivative_matrix_now_to_past = np.zeros((7,2,2))
    # d_P(t+Deltat,t+Deltat)/d_theta
    next_covariance_derivative_matrix = np.zeros((7,2,2))
    # initialisation for the common part of the derivative of P(t,t) for each parameter
    common_state_space_variance_derivative_element = np.zeros((7,2,2))
    # This corresponds to d_P(s,t)/d_theta in the Calderazzo paper
    covariance_matrix_derivative_intermediate_to_current = np.zeros((7,2,2))
    # This corresponds to d_P(s,t-tau)/d_theta
    covariance_matrix_derivative_intermediate_to_past = np.zeros((7,2,2))
    # This corresponds to d_P(s,t+Deltat)/d_theta in the Calderazzo paper
    covariance_matrix_derivative_intermediate_to_next = np.zeros((7,2,2))
    # initialisation for the common part of the derivative of P(s,t) for each parameter
    common_intermediate_state_space_variance_derivative_element = np.zeros((7,2,2))

    # derivations for the following are found in Calderazzo et. al. (2018)
    # g is [[-mRNA_degradation_rate,0],                  *[M(t),
    #       [translation_rate,-protein_degradation_rate]] [P(t)]
    # and its derivative will be called instant_jacobian
    # f is [[basal_transcription_rate*hill_function(past_protein)],0]
    # and its derivative with respect to the past state will be called delayed_jacobian
    # the matrix A in the paper will be called variance_of_noise
    instant_jacobian = np.array([[-mRNA_degradation_rate,0.0],[translation_rate,-protein_degradation_rate]])
    instant_jacobian_transpose = np.transpose(instant_jacobian)

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

        # jacobian of f is derivative of f with respect to past state ([past_mRNA, past_protein])
        delayed_jacobian = np.array([[0.0,basal_transcription_rate*hill_function_derivative_value],[0.0,0.0]])
        delayed_jacobian_transpose = np.transpose(delayed_jacobian)

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
        for short_row_index, long_row_index in enumerate([current_time_index,
                                                          total_number_of_states+current_time_index]):
            for short_column_index, long_column_index in enumerate([current_time_index,
                                                                    total_number_of_states+current_time_index]):
                current_covariance_matrix[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]

        # this is P(t-\tau,t) in page 5 of the supplementary material of Calderazzo et. al
        for short_row_index, long_row_index in enumerate([past_time_index,
                                                          total_number_of_states+past_time_index]):
            for short_column_index, long_column_index in enumerate([current_time_index,
                                                                    total_number_of_states+current_time_index]):
                covariance_matrix_past_to_now[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]

        # this is P(t,t-\tau) in page 5 of the supplementary material of Calderazzo et. al.
        for short_row_index, long_row_index in enumerate([current_time_index,
                                                          total_number_of_states+current_time_index]):
            for short_column_index, long_column_index in enumerate([past_time_index,
                                                                    total_number_of_states+past_time_index]):
                covariance_matrix_now_to_past[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                     long_column_index]

        variance_change_current_contribution = ( instant_jacobian.dot(current_covariance_matrix) +
                                                 current_covariance_matrix.dot(instant_jacobian_transpose) )

        variance_change_past_contribution = ( delayed_jacobian.dot(covariance_matrix_past_to_now) +
                                              covariance_matrix_now_to_past.dot(delayed_jacobian_transpose) )

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
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([current_time_index,
                                                                        total_number_of_states+current_time_index]):
                    covariance_matrix_intermediate_to_current[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                                         long_column_index]
            # This corresponds to P(s,t-tau)
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([past_time_index,
                                                                        total_number_of_states+past_time_index]):
                    covariance_matrix_intermediate_to_past[short_row_index,short_column_index] = state_space_variance[long_row_index,
                                                                                                                         long_column_index]

            covariance_derivative = ( covariance_matrix_intermediate_to_current.dot( instant_jacobian_transpose) +
                                      covariance_matrix_intermediate_to_past.dot(delayed_jacobian_transpose))

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next = covariance_matrix_intermediate_to_current + discretisation_time_step*covariance_derivative

            # Fill in the big matrix
            for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                              total_number_of_states+intermediate_time_index]):
                for short_column_index, long_column_index in enumerate([next_time_index,
                                                                        total_number_of_states+next_time_index]):
                    state_space_variance[long_row_index,long_column_index] = covariance_matrix_intermediate_to_next[short_row_index,
                                                                                                                    short_column_index]
            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            for short_row_index, long_row_index in enumerate([next_time_index,
                                                              total_number_of_states+next_time_index]):
                for short_column_index, long_column_index in enumerate([intermediate_time_index,
                                                                        total_number_of_states+intermediate_time_index]):
                    state_space_variance[long_row_index,long_column_index] = covariance_matrix_intermediate_to_next[short_column_index,
                                                                                                                    short_row_index]

    #################################
    ####
    #### prediction step for the derivatives of the state space mean and variance wrt each parameter
    ####
    #################################

    ###
    ### state space mean derivatives
    ###

        # indexing with 1:3 for numba
        current_mean_derivative = state_space_mean_derivative[current_time_index,:,0:2]
        past_mean_derivative = state_space_mean_derivative[past_time_index,:,0:2]
        past_protein_derivative = state_space_mean_derivative[past_time_index,:,1]

        # calculate predictions for derivative of mean wrt each parameter
        # repression threshold
        hill_function_derivative_value_wrt_repression = hill_coefficient*np.power(past_protein/repression_threshold,
                                                                                  hill_coefficient)/( repression_threshold*
                                                                                  np.power(1.0+np.power( past_protein/repression_threshold,
                                                                                                         hill_coefficient),
                                                                                           2))

        repression_derivative = ( instant_jacobian.dot(current_mean_derivative[0]).reshape((2,1)) +
                                  delayed_jacobian.dot(past_mean_derivative[0]).reshape((2,1)) +
                                  np.array([[basal_transcription_rate*hill_function_derivative_value_wrt_repression],[0.0]]) )

        next_mean_derivative[0] = current_mean_derivative[0] + discretisation_time_step*(repression_derivative.reshape((1,2)))

        # hill coefficient
        hill_function_derivative_value_wrt_hill_coefficient = - np.log(past_protein/repression_threshold)*np.power(past_protein/repression_threshold,
                                                                     hill_coefficient)/( np.power(1.0+np.power( past_protein/repression_threshold,
                                                                     hill_coefficient),2))

        hill_coefficient_derivative = ( instant_jacobian.dot(current_mean_derivative[1]).reshape((2,1)) +
                                        delayed_jacobian.dot(past_mean_derivative[1]).reshape((2,1)) +
                                        np.array(([[basal_transcription_rate*hill_function_derivative_value_wrt_hill_coefficient],[0.0]])) )

        next_mean_derivative[1] = current_mean_derivative[1] + discretisation_time_step*(hill_coefficient_derivative.reshape((1,2)))

        # mRNA degradation rate
        mRNA_degradation_rate_derivative = ( instant_jacobian.dot(current_mean_derivative[2]).reshape((2,1)) +
                                             delayed_jacobian.dot(past_mean_derivative[2]).reshape((2,1)) +
                                             np.array(([[-current_mean[0]],[0.0]])) )

        next_mean_derivative[2] = current_mean_derivative[2] + discretisation_time_step*(mRNA_degradation_rate_derivative.reshape((1,2)))

        # protein degradation rate
        protein_degradation_rate_derivative = ( instant_jacobian.dot(current_mean_derivative[3]).reshape((2,1)) +
                                                delayed_jacobian.dot(past_mean_derivative[3]).reshape((2,1)) +
                                                np.array(([[0.0],[-current_mean[1]]])) )

        next_mean_derivative[3] = current_mean_derivative[3] + discretisation_time_step*(protein_degradation_rate_derivative.reshape((1,2)))

        # basal transcription rate
        basal_transcription_rate_derivative = ( instant_jacobian.dot(current_mean_derivative[4]).reshape((2,1)) +
                                                delayed_jacobian.dot(past_mean_derivative[4]).reshape((2,1)) +
                                                np.array(([[hill_function_value],[0.0]])) )

        next_mean_derivative[4] = current_mean_derivative[4] + discretisation_time_step*(basal_transcription_rate_derivative.reshape((1,2)))

        # translation rate
        translation_rate_derivative = ( instant_jacobian.dot(current_mean_derivative[5]).reshape((2,1)) +
                                        delayed_jacobian.dot(past_mean_derivative[5]).reshape((2,1)) +
                                        np.array(([[0.0],[current_mean[0]]])) )

        next_mean_derivative[5] = current_mean_derivative[5] + discretisation_time_step*(translation_rate_derivative.reshape((1,2)))

        # transcriptional delay
        transcription_delay_derivative = ( instant_jacobian.dot(current_mean_derivative[6]).reshape((2,1)) +
                                           delayed_jacobian.dot(past_mean_derivative[6]).reshape((2,1)) )

        next_mean_derivative[6] = current_mean_derivative[6] + discretisation_time_step*(transcription_delay_derivative.reshape((1,2)))

        # assign the predicted derivatives to our state_space_mean_derivative array
        state_space_mean_derivative[next_time_index] = next_mean_derivative

        ###
        ### state space variance derivatives
        ###

        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        # this is d_P(t,t)/d_theta
        for parameter_index in range(7):
            for short_row_index, long_row_index in enumerate([current_time_index,
                                                              total_number_of_states+current_time_index]):
                for short_column_index, long_column_index in enumerate([current_time_index,
                                                                        total_number_of_states+current_time_index]):
                    current_covariance_derivative_matrix[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                               long_row_index,
                                                                                                                                               long_column_index]

        # this is d_P(t-\tau,t)/d_theta
        for parameter_index in range(7):
            for short_row_index, long_row_index in enumerate([past_time_index,
                                                              total_number_of_states+past_time_index]):
                for short_column_index, long_column_index in enumerate([current_time_index,
                                                                        total_number_of_states+current_time_index]):
                    covariance_derivative_matrix_past_to_now[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                   long_row_index,
                                                                                                                                                   long_column_index]

        # this is d_P(t,t-\tau)/d_theta
        for parameter_index in range(7):
            for short_row_index, long_row_index in enumerate([current_time_index,
                                                              total_number_of_states+current_time_index]):
                for short_column_index, long_column_index in enumerate([past_time_index,
                                                                        total_number_of_states+past_time_index]):
                    covariance_derivative_matrix_now_to_past[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                   long_row_index,
                                                                                                                                                   long_column_index]
        ## d_P(t+Deltat,t+Deltat)/d_theta

        # the derivative is quite long and slightly different for each parameter, meaning it's difficult to
        # code this part with a loop. For each parameter we divide it in to it's constituent parts. There is one
        # main part in common for every derivative which is defined here as common_state_space_variance_derivative_element
        for parameter_index in range(7):
            common_state_space_variance_derivative_element[parameter_index] = ( np.dot(instant_jacobian,
                                                                                       current_covariance_derivative_matrix[parameter_index]) +
                                                                                np.dot(current_covariance_derivative_matrix[parameter_index],
                                                                                       instant_jacobian_transpose) +
                                                                                np.dot(delayed_jacobian,
                                                                                       covariance_derivative_matrix_past_to_now[parameter_index]) +
                                                                                np.dot(covariance_derivative_matrix_now_to_past[parameter_index],
                                                                                       delayed_jacobian_transpose) )

        hill_function_second_derivative_value = hill_coefficient*np.power(past_protein/repression_threshold,
                                                                          hill_coefficient)*(
                                                np.power(past_protein/repression_threshold,
                                                         hill_coefficient) +
                                                hill_coefficient*(np.power(past_protein/repression_threshold,
                                                                           hill_coefficient)-1)+1)/( np.power(past_protein,2)*
                                                np.power(1.0+np.power( past_protein/repression_threshold,
                                                                       hill_coefficient),3))
        # repression threshold
        # this refers to d(f'(p(t-\tau)))/dp_0
        hill_function_second_derivative_value_wrt_repression = -np.power(hill_coefficient,2)*(np.power(past_protein/repression_threshold,
                                                                          hill_coefficient)-1)*np.power(past_protein/repression_threshold,
                                                                                                hill_coefficient-1)/( np.power(repression_threshold,2)*
                                                                (np.power(1.0+np.power( past_protein/repression_threshold,
                                                                       hill_coefficient),3)))

        # instant_jacobian_derivative_wrt_repression = 0
        delayed_jacobian_derivative_wrt_repression = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[0,1]],[0.0,0.0]]) +
                                                      np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value_wrt_repression],[0.0,0.0]]) )
        delayed_jacobian_derivative_wrt_repression_transpose = np.transpose(delayed_jacobian_derivative_wrt_repression)

        instant_noise_derivative_wrt_repression = (np.array([[mRNA_degradation_rate*current_mean_derivative[0,0],0.0],
                                                             [0.0,translation_rate*current_mean_derivative[0,0] + protein_degradation_rate*current_mean_derivative[0,1]]]))

        delayed_noise_derivative_wrt_repression = (np.array([[basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[0,1] + hill_function_derivative_value_wrt_repression),0.0],
                                                             [0.0,0.0]]))

        derivative_of_variance_wrt_repression_threshold = ( common_state_space_variance_derivative_element[0] +
                                                            np.dot(delayed_jacobian_derivative_wrt_repression,covariance_matrix_past_to_now) +
                                                            np.dot(covariance_matrix_now_to_past,delayed_jacobian_derivative_wrt_repression_transpose) +
                                                            instant_noise_derivative_wrt_repression + delayed_noise_derivative_wrt_repression )

        next_covariance_derivative_matrix[0] = current_covariance_derivative_matrix[0] + discretisation_time_step*(derivative_of_variance_wrt_repression_threshold)

        # hill coefficient
        # this refers to d(f'(p(t-\tau)))/dh
        hill_function_second_derivative_value_wrt_hill_coefficient = np.power(past_protein/repression_threshold,hill_coefficient)*(-np.power(past_protein/repression_threshold,hill_coefficient) +
                                                                     hill_coefficient*(np.power(past_protein/repression_threshold,hill_coefficient)-1)*np.log(past_protein/repression_threshold)-1)/(
                                                                        past_protein*np.power(1.0+np.power(past_protein/repression_threshold,hill_coefficient),3))

        # instant_jacobian_derivative_wrt_hill_coefficient = 0
        delayed_jacobian_derivative_wrt_hill_coefficient = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[1,1]],[0.0,0.0]]) +
                                                            np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value_wrt_hill_coefficient],[0.0,0.0]]) )

        instant_noise_derivative_wrt_hill_coefficient = (np.array([[mRNA_degradation_rate*current_mean_derivative[1,0],0.0],
                                                                   [0.0,translation_rate*current_mean_derivative[1,0] + protein_degradation_rate*current_mean_derivative[1,1]]]))

        delayed_noise_derivative_wrt_hill_coefficient = (np.array([[basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[1,1] + hill_function_derivative_value_wrt_hill_coefficient),0.0],
                                                                   [0.0,0.0]]))

        derivative_of_variance_wrt_hill_coefficient = ( common_state_space_variance_derivative_element[1] +
                                                        np.dot(delayed_jacobian_derivative_wrt_hill_coefficient,covariance_matrix_past_to_now) +
                                                        np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_hill_coefficient)) +
                                                        instant_noise_derivative_wrt_hill_coefficient + delayed_noise_derivative_wrt_hill_coefficient )

        next_covariance_derivative_matrix[1] = current_covariance_derivative_matrix[1] + discretisation_time_step*(derivative_of_variance_wrt_hill_coefficient)

        # mRNA degradation rate
        instant_jacobian_derivative_wrt_mRNA_degradation = np.array([[-1.0,0.0],[0.0,0.0]])
        delayed_jacobian_derivative_wrt_mRNA_degradation = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[2,1]],[0.0,0.0]]) )
        instant_noise_derivative_wrt_mRNA_degradation = (np.array([[mRNA_degradation_rate*current_mean_derivative[2,0] + current_mean[0],0.0],
                                                                   [0.0,translation_rate*current_mean_derivative[2,0] + protein_degradation_rate*current_mean_derivative[2,1]]]))

        delayed_noise_derivative_wrt_mRNA_degradation = (np.array([[basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[2,1]),0.0],
                                                                   [0.0,0.0]]))

        derivative_of_variance_wrt_mRNA_degradation = ( common_state_space_variance_derivative_element[2] +
                                                        np.dot(instant_jacobian_derivative_wrt_mRNA_degradation,current_covariance_matrix) +
                                                        np.dot(current_covariance_matrix,np.transpose(instant_jacobian_derivative_wrt_mRNA_degradation)) +
                                                        np.dot(delayed_jacobian_derivative_wrt_mRNA_degradation,covariance_matrix_past_to_now) +
                                                        np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_mRNA_degradation)) +
                                                        instant_noise_derivative_wrt_mRNA_degradation + delayed_noise_derivative_wrt_mRNA_degradation )

        next_covariance_derivative_matrix[2] = current_covariance_derivative_matrix[2] + discretisation_time_step*(derivative_of_variance_wrt_mRNA_degradation)

        # protein degradation rate
        instant_jacobian_derivative_wrt_protein_degradation = np.array([[0.0,0.0],[0.0,-1.0]])
        delayed_jacobian_derivative_wrt_protein_degradation = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[3,1]],[0.0,0.0]]) )
        instant_noise_derivative_wrt_protein_degradation = (np.array([[mRNA_degradation_rate*current_mean_derivative[3,0],0.0],
                                                                      [0.0,translation_rate*current_mean_derivative[3,0] + protein_degradation_rate*current_mean_derivative[3,1] + current_mean[1]]]))

        delayed_noise_derivative_wrt_protein_degradation = (np.array([[basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[3,1]),0.0],
                                                                      [0.0,0.0]]))

        derivative_of_variance_wrt_protein_degradation = ( common_state_space_variance_derivative_element[3] +
                                                           np.dot(instant_jacobian_derivative_wrt_protein_degradation,current_covariance_matrix) +
                                                           np.dot(current_covariance_matrix,np.transpose(instant_jacobian_derivative_wrt_protein_degradation)) +
                                                           np.dot(delayed_jacobian_derivative_wrt_protein_degradation,covariance_matrix_past_to_now) +
                                                           np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_protein_degradation)) +
                                                           instant_noise_derivative_wrt_protein_degradation + delayed_noise_derivative_wrt_protein_degradation )

        next_covariance_derivative_matrix[3] = current_covariance_derivative_matrix[3] + discretisation_time_step*(derivative_of_variance_wrt_protein_degradation)

        # basal transcription rate
        # instant_jacobian_derivative_wrt_basal_transcription = 0
        delayed_jacobian_derivative_wrt_basal_transcription = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[4,1]],[0.0,0.0]]) +
                                                               np.array([[0.0,hill_function_derivative_value],[0.0,0.0]]) )
        instant_noise_derivative_wrt_basal_transcription = (np.array([[mRNA_degradation_rate*current_mean_derivative[4,0],0.0],
                                                                      [0.0,translation_rate*current_mean_derivative[4,0] + protein_degradation_rate*current_mean_derivative[4,1]]]))

        delayed_noise_derivative_wrt_basal_transcription = (np.array([[basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[4,1] + hill_function_value,0.0],
                                                                      [0.0,0.0]]))

        derivative_of_variance_wrt_basal_transcription = ( common_state_space_variance_derivative_element[4] +
                                                           np.dot(delayed_jacobian_derivative_wrt_basal_transcription,covariance_matrix_past_to_now) +
                                                           np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_basal_transcription)) +
                                                           instant_noise_derivative_wrt_basal_transcription + delayed_noise_derivative_wrt_basal_transcription )

        next_covariance_derivative_matrix[4] = current_covariance_derivative_matrix[4] + discretisation_time_step*(derivative_of_variance_wrt_basal_transcription)

        # translation rate
        instant_jacobian_derivative_wrt_translation_rate = np.array([[0.0,0.0],[1.0,0.0]])
        delayed_jacobian_derivative_wrt_translation_rate = (np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[5,1]],[0.0,0.0]]))
        instant_noise_derivative_wrt_translation_rate = (np.array([[mRNA_degradation_rate*current_mean_derivative[5,0],0.0],
                                                                   [0.0,translation_rate*current_mean_derivative[5,0] + protein_degradation_rate*current_mean_derivative[5,1] + current_mean[0]]]))

        delayed_noise_derivative_wrt_translation_rate = (np.array([[basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[5,1],0.0],
                                                                      [0.0,0.0]]))

        derivative_of_variance_wrt_translation_rate = ( common_state_space_variance_derivative_element[5] +
                                                        np.dot(instant_jacobian_derivative_wrt_translation_rate,current_covariance_matrix) +
                                                        np.dot(current_covariance_matrix,np.transpose(instant_jacobian_derivative_wrt_translation_rate)) +
                                                        np.dot(delayed_jacobian_derivative_wrt_translation_rate,covariance_matrix_past_to_now) +
                                                        np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_translation_rate)) +
                                                        instant_noise_derivative_wrt_translation_rate + delayed_noise_derivative_wrt_translation_rate )

        next_covariance_derivative_matrix[5] = current_covariance_derivative_matrix[5] + discretisation_time_step*(derivative_of_variance_wrt_translation_rate)

        # transcriptional delay
        # instant_jacobian_derivative_wrt_transcription_delay = 0
        delayed_jacobian_derivative_wrt_transcription_delay = np.array([[0.0,basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[6,1]],[0.0,0.0]])
        instant_noise_derivative_wrt_transcription_delay = (np.array([[mRNA_degradation_rate*current_mean_derivative[6,0],0.0],
                                                                      [0.0,translation_rate*current_mean_derivative[6,0] + protein_degradation_rate*current_mean_derivative[6,1]]]))

        delayed_noise_derivative_wrt_transcription_delay = np.array([[basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[6,1],0.0],
                                                                      [0.0,0.0]])

        derivative_of_variance_wrt_transcription_delay = ( common_state_space_variance_derivative_element[6] +
                                                        np.dot(delayed_jacobian_derivative_wrt_transcription_delay,covariance_matrix_past_to_now) +
                                                        np.dot(covariance_matrix_now_to_past,np.transpose(delayed_jacobian_derivative_wrt_transcription_delay)) +
                                                        instant_noise_derivative_wrt_transcription_delay + delayed_noise_derivative_wrt_transcription_delay )

        next_covariance_derivative_matrix[6] = current_covariance_derivative_matrix[6] + discretisation_time_step*(derivative_of_variance_wrt_transcription_delay)

        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        for parameter_index in range(7):
            for short_row_index, long_row_index in enumerate([next_time_index,
                                                              total_number_of_states+next_time_index]):
                for short_column_index, long_column_index in enumerate([next_time_index,
                                                                        total_number_of_states+next_time_index]):
                    state_space_variance_derivative[parameter_index,long_row_index,long_column_index] = next_covariance_derivative_matrix[parameter_index,
                                                                                                                                          short_row_index,
                                                                                                                                          short_column_index]

        ## now we need to update the cross correlations, d_P(s,t)/d_theta in the Calderazzo paper
        # the range needs to include t, since we want to propagate d_P(t,t)/d_theta into d_P(t,t+Deltat)/d_theta
        for intermediate_time_index in range(past_time_index,current_time_index+1):
            # This corresponds to d_P(s,t)/d_theta in the Calderazzo paper
            # for loops instead of np.ix_-like indexing
            for parameter_index in range(7):
                for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                                  total_number_of_states+intermediate_time_index]):
                    for short_column_index, long_column_index in enumerate([current_time_index,
                                                                            total_number_of_states+current_time_index]):
                        covariance_matrix_derivative_intermediate_to_current[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                                   long_row_index,
                                                                                                                                                                   long_column_index]
            # This corresponds to d_P(s,t-tau)/d_theta
            for parameter_index in range(7):
                for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                                  total_number_of_states+intermediate_time_index]):
                    for short_column_index, long_column_index in enumerate([past_time_index,
                                                                            total_number_of_states+past_time_index]):
                        covariance_matrix_derivative_intermediate_to_past[parameter_index,short_row_index,short_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                                long_row_index,
                                                                                                                                                                long_column_index]

            # Again, this derivative is slightly different for each parameter, meaning it's difficult to
            # code this part with a loop. For each parameter we divide it in to it's constituent parts. There is one
            # main part in common for every derivative which is defined here as common_intermediate_state_space_variance_derivative_element
            for parameter_index in range(7):
                common_intermediate_state_space_variance_derivative_element[parameter_index] = ( np.dot(covariance_matrix_derivative_intermediate_to_current[parameter_index],
                                                                                                        instant_jacobian_transpose) +
                                                                                                 np.dot(covariance_matrix_derivative_intermediate_to_past[parameter_index],
                                                                                                        delayed_jacobian_transpose) )
            # repression threshold
            derivative_of_intermediate_variance_wrt_repression_threshold = ( common_intermediate_state_space_variance_derivative_element[0] +
                                                                             np.dot(covariance_matrix_intermediate_to_past,delayed_jacobian_derivative_wrt_repression_transpose) )

            covariance_matrix_derivative_intermediate_to_next[0] = covariance_matrix_derivative_intermediate_to_current[0] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_repression_threshold)

            # hill coefficient
            derivative_of_intermediate_variance_wrt_hill_coefficient = ( common_intermediate_state_space_variance_derivative_element[1] +
                                                                         np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_hill_coefficient)))

            covariance_matrix_derivative_intermediate_to_next[1] = covariance_matrix_derivative_intermediate_to_current[1] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_hill_coefficient)

            # mRNA degradation rate
            derivative_of_intermediate_variance_wrt_mRNA_degradation = ( common_intermediate_state_space_variance_derivative_element[2] +
                                                                         np.dot(covariance_matrix_intermediate_to_current,np.transpose(instant_jacobian_derivative_wrt_mRNA_degradation)) +
                                                                         np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_mRNA_degradation)) )

            covariance_matrix_derivative_intermediate_to_next[2] = covariance_matrix_derivative_intermediate_to_current[2] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_mRNA_degradation)

            # protein degradation rate
            derivative_of_intermediate_variance_wrt_protein_degradation = ( common_intermediate_state_space_variance_derivative_element[3] +
                                                                            np.dot(covariance_matrix_intermediate_to_current,np.transpose(instant_jacobian_derivative_wrt_protein_degradation)) +
                                                                            np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_protein_degradation)) )

            covariance_matrix_derivative_intermediate_to_next[3] = covariance_matrix_derivative_intermediate_to_current[3] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_protein_degradation)

            # basal transcription rate
            derivative_of_intermediate_variance_wrt_basal_transcription = ( common_intermediate_state_space_variance_derivative_element[4] +
                                                                            np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_basal_transcription)) )

            covariance_matrix_derivative_intermediate_to_next[4] = covariance_matrix_derivative_intermediate_to_current[4] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_basal_transcription)

            # translation rate
            derivative_of_intermediate_variance_wrt_translation_rate = ( common_intermediate_state_space_variance_derivative_element[5] +
                                                                         np.dot(covariance_matrix_intermediate_to_current,np.transpose(instant_jacobian_derivative_wrt_translation_rate)) +
                                                                         np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_translation_rate)) )

            covariance_matrix_derivative_intermediate_to_next[5] = covariance_matrix_derivative_intermediate_to_current[5] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_translation_rate)

            # transcriptional delay
            derivative_of_intermediate_variance_wrt_transcription_delay = ( common_intermediate_state_space_variance_derivative_element[6] +
                                                                            np.dot(covariance_matrix_intermediate_to_past,np.transpose(delayed_jacobian_derivative_wrt_transcription_delay)) )

            covariance_matrix_derivative_intermediate_to_next[6] = covariance_matrix_derivative_intermediate_to_current[6] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_transcription_delay)

            # Fill in the big matrix
            for parameter_index in range(7):
                for short_row_index, long_row_index in enumerate([intermediate_time_index,
                                                                  total_number_of_states+intermediate_time_index]):
                    for short_column_index, long_column_index in enumerate([next_time_index,
                                                                        total_number_of_states+next_time_index]):
                        state_space_variance_derivative[parameter_index,long_row_index,long_column_index] = covariance_matrix_derivative_intermediate_to_next[parameter_index,
                                                                                                                                                              short_row_index,
                                                                                                                                                              short_column_index]

            # Fill in the big matrix with transpose arguments, i.e. d_P(t+Deltat, s)/d_theta - works if initialised symmetrically
            for parameter_index in range(7):
                for short_row_index, long_row_index in enumerate([next_time_index,
                                                                  total_number_of_states+next_time_index]):
                    for short_column_index, long_column_index in enumerate([intermediate_time_index,
                                                                            total_number_of_states+intermediate_time_index]):
                        state_space_variance_derivative[parameter_index,long_row_index,long_column_index] = covariance_matrix_derivative_intermediate_to_next[parameter_index,
                                                                                                                                                              short_column_index,
                                                                                                                                                              short_row_index]

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative

@jit(nopython = True)
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
    if state_space_mean[-1,0] == 0:
        discretisation_time_step = 1.0
    else:
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
    predicted_final_state_space_mean = np.copy(state_space_mean[current_number_of_states-1,1:3])
    # print('predicted_final_state_space_mean 1',predicted_final_state_space_mean)

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
    # shortened_covariance_matrix_past_to_final = np.ascontiguousarray(shortened_covariance_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)])
    shortened_covariance_matrix_past_to_final = shortened_covariance_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)]
    # print(shortened_covariance_matrix_past_to_final.flags)

    # and P(t+Deltat,t+Deltat-delay:t+deltat), replacing ((discrete_delay),-1) with a splice for numba
    # shortened_covariance_matrix_final_to_past = np.ascontiguousarray(shortened_covariance_matrix[discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:])
    shortened_covariance_matrix_final_to_past = shortened_covariance_matrix[discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:]

    # This is F in the paper
    observation_transform = np.array([0.0,1.0])

    # This is P(t+Deltat,t+Deltat) in the paper
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
                                np.transpose(observation_transform.reshape((1,2))) )*helper_inverse

    # This is rho*
    updated_stacked_state_space_mean = ( stacked_state_space_mean +
                                         (adaptation_coefficient*(current_observation[1] -
                                                                 observation_transform.reshape((1,2)).dot(
                                                                     predicted_final_state_space_mean.reshape((2,1))))[0][0]).reshape(all_indices_up_to_delay.shape[0]) )
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
    # using for loop indexing for numba
    shortened_covariance_derivative_matrix = np.zeros((7,all_indices_up_to_delay.shape[0],all_indices_up_to_delay.shape[0]))
    for parameter_index in range(7):
        for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
            for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
                shortened_covariance_derivative_matrix[parameter_index,shortened_row_index,shortened_column_index] = state_space_variance_derivative[parameter_index,
                                                                                                                                                     long_row_index,
                                                                                                                                                     long_column_index]
    # extract d_P(t+Deltat-delay:t+deltat,t+Deltat)/d_theta, replacing ((discrete_delay),-1) with a splice for numba
    # shortened_covariance_derivative_matrix_past_to_final = np.ascontiguousarray(shortened_covariance_derivative_matrix[:,:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)])
    shortened_covariance_derivative_matrix_past_to_final = shortened_covariance_derivative_matrix[:,:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1)]

    # and d_P(t+Deltat,t+Deltat-delay:t+deltat)/d_theta, replacing ((discrete_delay),-1) with a splice for numba
    # shortened_covariance_derivative_matrix_final_to_past = np.ascontiguousarray(shortened_covariance_derivative_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:])
    shortened_covariance_derivative_matrix_final_to_past = shortened_covariance_derivative_matrix[:,discrete_delay:2*(discrete_delay+1):(discrete_delay+1),:]

    # This is the derivative of P(t+Deltat,t+Deltat) in the paper
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
        adaptation_coefficient_derivative[parameter_index] = (shortened_covariance_derivative_matrix_past_to_final[parameter_index].dot(np.transpose(observation_transform.reshape(1,2)))*helper_inverse -
                                                             (shortened_covariance_matrix_past_to_final.dot(np.transpose(observation_transform.reshape((1,2))).dot(observation_transform.reshape((1,2)).dot(
                                                             predicted_final_covariance_derivative_matrix[parameter_index].dot(np.transpose(observation_transform.reshape((1,2))))))))*np.power(helper_inverse,2) ).reshape(all_indices_up_to_delay.shape[0])

    # This is d_rho*/d_theta
    updated_stacked_state_space_mean_derivative = np.zeros((7,2*(discrete_delay+1)))
    # print('predicted_final_state_space_mean 2',predicted_final_state_space_mean)
    for parameter_index in range(7):
        updated_stacked_state_space_mean_derivative[parameter_index] = ( stacked_state_space_mean_derivative[parameter_index] +
                                                                         adaptation_coefficient_derivative[parameter_index]*(current_observation[1] -
                                                                         observation_transform.reshape((1,2)).dot(predicted_final_state_space_mean.reshape((2,1))))[0][0] -
                                                                         adaptation_coefficient.dot(observation_transform.reshape((1,2)).dot(
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

    _, _, _, _, predicted_observation_distributions, _, _ = kalman_filter(protein_at_observations,
                                                                          model_parameters,
                                                                          measurement_variance)
    observations = protein_at_observations[:,1]
    mean = predicted_observation_distributions[:,1]
    sd = np.sqrt(predicted_observation_distributions[:,2])

    log_likelihood = np.sum(norm.logpdf(observations,mean,sd))

    return log_likelihood

def calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,model_parameters,measurement_variance = 10):
    """
    Calculates the log of the likelihood, and the derivative of the negative log likelihood wrt each parameter, of our data given the
    paramters, using the Kalman filter. It uses the predicted_observation_distributions, predicted_observation_mean_derivatives, and
    predicted_observation_variance_derivatives from the kalman_filter function. It returns the log likelihood as in the
    calculate_log_likelihood_at_parameter_point function, and also returns an array of the derivative wrt each parameter.

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

    log_likelihood_derivative : numpy array.
        The derivative of the log likelihood of the data, wrt each model parameter
    """
    from scipy.stats import norm, gamma, uniform

    mean_protein = np.mean(protein_at_observations[:,1])
    number_of_parameters = model_parameters.shape[0]

    if ((uniform(100,2*mean_protein-100).pdf(model_parameters[0]) == 0) or
        (uniform(2,6-2).pdf(model_parameters[1]) == 0) or
        (uniform(0.01,60-0.01).pdf(model_parameters[4]) == 0) or
        (uniform(0.01,40-0.01).pdf(model_parameters[5]) == 0) or
        (uniform(5,40-5).pdf(model_parameters[6]) == 0) ):
        return -np.inf, np.zeros(number_of_parameters)

    _, _, _, _, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = kalman_filter(protein_at_observations,
                                                                                                                                                        model_parameters,
                                                                                                                                                        measurement_variance)
    # calculate log likelihood as before
    if protein_at_observations.reshape(-1,2).shape[0] == 1:
        number_of_observations = 1
        observations = [protein_at_observations[1]]
    else:
        number_of_observations = protein_at_observations.shape[0]
        observations = protein_at_observations[:,1]

    mean = predicted_observation_distributions[:,1]
    sd = np.sqrt(predicted_observation_distributions[:,2])

    log_likelihood = np.sum(norm.logpdf(observations,mean,sd))
    # now for the computation of the derivative of the negative log likelihood. An expression of this can be found
    # at equation (28) in Mbalawata, Särkkä, Haario (2013)
    observation_transform = np.array([[0.0,1.0]])
    helper_inverse = 1.0/predicted_observation_distributions[:,2]
    log_likelihood_derivative = np.zeros(number_of_parameters)

    for parameter_index in range(number_of_parameters):
        for time_index in range(number_of_observations):
            log_likelihood_derivative[parameter_index] -= 0.5*(helper_inverse[time_index]*np.trace(observation_transform.dot(
                                                                                                            predicted_observation_variance_derivatives[time_index,parameter_index].dot(
                                                                                                            np.transpose(observation_transform))))
                                                                         -
                                                                         helper_inverse[time_index]*np.transpose(observation_transform.dot(
                                                                                                                 predicted_observation_mean_derivatives[time_index,parameter_index]))[0]*
                                                                                                     (observations[time_index] - mean[time_index])
                                                                         -
                                                                         np.power(helper_inverse[time_index],2)*np.power(observations[time_index] - mean[time_index],2)*
                                                                         observation_transform.dot(
                                                                         predicted_observation_variance_derivatives[time_index,parameter_index].dot(
                                                                         np.transpose(observation_transform)))
                                                                         -
                                                                         helper_inverse[time_index]*(observations[time_index] - mean[time_index])*
                                                                         observation_transform.dot(predicted_observation_mean_derivatives[time_index,parameter_index])[0])

    return log_likelihood, log_likelihood_derivative


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
    # likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)

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
            new_state[[0,1,4,5,6]] = current_state[[0,1,4,5,6]] + acceptance_tuner*cholesky_covariance.dot(multivariate_normal.rvs(size=5))
            # fix certain parameters
            new_state[[2,3,4,5,6]] = np.copy(initial_state[[2,3,4,5,6]])

            positive_new_parameters = new_state[[0,1,2,3,6]]
            if all(item > 0 for item in positive_new_parameters) == True:
                new_log_prior = np.sum(uniform.logpdf(new_state,loc=shape,scale=scale))

                # reparameterise
                reparameterised_new_state            = np.copy(new_state)
                reparameterised_current_state        = np.copy(current_state)
                reparameterised_new_state[[4,5]]     = np.power(10,new_state[[4,5]])
                reparameterised_current_state[[4,5]] = np.power(10,current_state[[4,5]])

                # try:
                #     # in this line the pool returns an object of type mp.AsyncResult, which is not directly the likelihood,
                #     # but which can be interrogated about the status of the calculation and so on
                #     new_likelihood_result = likelihood_calculations_pool.apply_async(calculate_log_likelihood_at_parameter_point,
                #                                                               args = (protein_at_observations,
                #                                                                       reparameterised_new_state,
                #                                                                       measurement_variance))
                #
                #     # ask the async result from above to return the new likelihood when it is ready
                #     new_log_likelihood = new_likelihood_result.get(30)
                # except ValueError:
                #     new_log_likelihood = -np.inf
                # except mp.TimeoutError:
                #     likelihood_calculations_pool.close()
                #     likelihood_calculations_pool.terminate()
                #     likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)
                new_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,
                                                                                 reparameterised_new_state,
                                                                                 measurement_variance)

                acceptance_ratio = np.exp(new_log_prior + new_log_likelihood - current_log_prior - current_log_likelihood)

                if np.mod(step_index,5) == 0:
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

def kalman_hmc(iterations,protein_at_observations,measurement_variance,initial_epsilon,number_of_leapfrog_steps,initial_parameters):
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

    Returns
    -------

    """
    mass_matrix = np.identity(7)
    current_position = initial_parameters
    cholesky_mass_matrix = np.linalg.cholesky(mass_matrix)
    inverse_mass_matrix = mass_matrix
    # print(inverse_mass_matrix)

    output = np.zeros((iterations,7))
    for step_index in range(iterations):
        epsilon = np.random.normal(initial_epsilon,np.sqrt(0.2*initial_epsilon))
        position = current_position
        position[[2,3]] = np.array([np.log(2)/30,np.log(2)/90])
        momentum = cholesky_mass_matrix.dot(np.random.normal(0,1,len(position)))
        current_momentum = momentum

        print('momentum',momentum)
        print('position',position)

        # simulate hamiltonian dynamics
        _, negative_log_likelihood_derivative = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,
                                                                                                           position,
                                                                                                           measurement_variance)

        momentum = momentum - epsilon*negative_log_likelihood_derivative/2 # half step for momentum

        for leapfrog_step_index in range(number_of_leapfrog_steps):
            position = position + epsilon*inverse_mass_matrix.dot(momentum)
            position[[2,3]] = np.array([np.log(2)/30,np.log(2)/90])
            # handling constraint, all parameters must be positive
            for i in np.where(position < 0):
                position[i] *= -1
                momentum[i] *= -1

            if leapfrog_step_index != (number_of_leapfrog_steps - 1):
                _, negative_log_likelihood_derivative = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,
                                                                                                                   position,
                                                                                                                   measurement_variance)
                momentum = momentum - epsilon*negative_log_likelihood_derivative/2

        _, negative_log_likelihood_derivative = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,
                                                                                                           position,
                                                                                                           measurement_variance)
        momentum = momentum - epsilon*negative_log_likelihood_derivative/2 # half step for momentum

        # acceptance/rejection Metropolis step
        current_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,
                                                                             current_position,
                                                                             measurement_variance)

        current_kinetic_energy = current_momentum.dot(inverse_mass_matrix.dot(current_momentum))/2

        proposed_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,
                                                                              position,
                                                                              measurement_variance)

        # print(momentum)
        proposed_kinetic_energy = momentum.dot(inverse_mass_matrix.dot(momentum))/2

        print('current log likelihood',current_log_likelihood)
        print('proposed log likelihood',proposed_log_likelihood)
        # print('current kinetic energy',current_kinetic_energy)
        # print('proposed kinetic energy',proposed_kinetic_energy)

        if (np.random.uniform() < np.exp(-current_log_likelihood -
                                         -proposed_log_likelihood +
                                         -current_kinetic_energy -
                                         -proposed_kinetic_energy)):
            current_position = position

        output[step_index] = current_position

    return output

def generic_mala(likelihood_and_derivative_calculator,
                 number_of_samples,
                 initial_position,
                 step_size,
                 proposal_covariance=np.eye(1),
                 thinning_rate=1,
                 known_parameter_dict=None,
                 *specific_args):
    '''Metropolis adjusted Langevin algorithm which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h/2*proposal_covariance*log_likelihood_gradient + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters:
    -----------

    likelihood_and_derivative_calculator : function
        a function which takes in a parameter and returns a log likelihood and its derivative, in that order

    number_of_samples : integer
        the number of samples the random walk proposes

    initial_position : numpy array
        starting value of the Markov chain

    proposal_covariance: numpy array
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.5

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    known_parameter_dict : dict
        a dict which contains values for parameters where the ground truth is known. The key is the name of the parameter,
        the value is a 2d array, where the first entry is its parameter index in the likelihood function, and the second
        entry is the ground truth.

    Returns:
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space
    '''
    likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)

    # initialise the covariance proposal matrix
    number_of_parameters = len(initial_position) - len(known_parameter_dict.values())
    known_parameters = [list(known_parameter_dict.values())[i][1] for i in [j for j in range(len(known_parameter_dict.values()))]]
    known_parameter_indices = [list(known_parameter_dict.values())[i][0] for i in [j for j in range(len(known_parameter_dict.values()))]]
    unknown_parameter_indices = [i for i in range(len(initial_position)) if i not in known_parameter_indices]

    # check if default value is used, and set to q x q identity
    if np.array_equal(proposal_covariance, np.eye(1)):
        proposal_covariance = np.eye(number_of_parameters)

    if np.array_equal(proposal_covariance, np.eye(number_of_parameters)):
        identity = True
    else:
        identity = False
        proposal_cholesky = np.linalg.cholesky(proposal_covariance + 1e-8*np.eye(number_of_parameters))

    proposal_covariance_inverse = np.linalg.inv(proposal_covariance)

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = np.zeros((number_of_samples,number_of_parameters))
    mcmc_samples[0] = initial_position[unknown_parameter_indices]
    number_of_iterations = number_of_samples*thinning_rate

    # set LAP parameters
    k = 1
    c0=1.0
    c1=0.4

    # initial markov chain
    current_position = np.copy(initial_position)
    current_log_likelihood, current_log_likelihood_gradient = likelihood_and_derivative_calculator(current_position,*specific_args)

    for iteration_index in range(1,number_of_iterations):
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        proposal = np.zeros(len(initial_position))
        if identity:
            proposal[unknown_parameter_indices] = ( current_position[unknown_parameter_indices] +
                                                    step_size*current_log_likelihood_gradient[unknown_parameter_indices]/2 +
                                                    np.sqrt(step_size)*np.random.normal(size=number_of_parameters) )
        else:
            proposal[unknown_parameter_indices] = ( current_position[unknown_parameter_indices] +
                                                    step_size*proposal_covariance.dot(current_log_likelihood_gradient[unknown_parameter_indices])/2 +
                                                    np.sqrt(step_size)*proposal_cholesky.dot(np.random.normal(size=number_of_parameters)) )

        # print('proposal: ',proposal)

        # compute transition probabilities for acceptance step
        # fix known parameters
        if known_parameter_dict != None:
            proposal[known_parameter_indices] = np.copy(known_parameters)

        ######################################################################

        try:
            # in this line the pool returns an object of type mp.AsyncResult, which is not directly the likelihood,
            # but which can be interrogated about the status of the calculation and so on
            new_likelihood_result = likelihood_calculations_pool.apply_async(likelihood_and_derivative_calculator,
                                                                             args = (proposal,
                                                                                     *specific_args))
            # ask the async result from above to return the new likelihood and gradient when it is ready
            proposal_log_likelihood, proposal_log_likelihood_gradient = new_likelihood_result.get(10)
        except ValueError:
            print('value error!')
            proposal_log_likelihood = -np.inf
        except mp.TimeoutError:
            print('I have found a TimeoutError!')
            likelihood_calculations_pool.close()
            likelihood_calculations_pool.terminate()
            likelihood_calculations_pool = mp.Pool(processes = 1, maxtasksperchild = 500)
            new_likelihood_result = likelihood_calculations_pool.apply_async(likelihood_and_derivative_calculator,
                                                                             args = (proposal,
                                                                                     *specific_args))
            # ask the async result from above to return the new likelihood and gradient when it is ready
            proposal_log_likelihood, proposal_log_likelihood_gradient = new_likelihood_result.get(10)

        # if any of the parameters were negative we get -inf for the log likelihood
        if proposal_log_likelihood == -np.inf:
            if iteration_index%thinning_rate == 0:
                mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position[unknown_parameter_indices]

            # LAP stuff also needed here
            if iteration_index%k == 0 and iteration_index > 1 and iteration_index < int(number_of_samples/2):
                r_hat = accepted_moves/iteration_index
                block_sample = mcmc_samples[:iteration_index]
                block_proposal_covariance = np.cov(block_sample.T)
                gamma_1 = 1/np.power(iteration_index,c1)
                gamma_2 = c0*gamma_1
                log_step_size_squared = np.log(np.power(step_size,2)) + gamma_2*(r_hat - 0.574)
                step_size = np.sqrt(np.exp(log_step_size_squared))
                proposal_covariance = proposal_covariance + gamma_1*(block_proposal_covariance - proposal_covariance) + 0.00001*np.eye(number_of_parameters)
                proposal_cholesky = np.linalg.cholesky(proposal_covariance)
                proposal_covariance_inverse = np.linalg.inv(proposal_covariance)
            continue

        forward_helper_variable = ( proposal[unknown_parameter_indices] - current_position[unknown_parameter_indices] -
                                    step_size*proposal_covariance.dot(current_log_likelihood_gradient[unknown_parameter_indices])/2 )
        backward_helper_variable = ( current_position[unknown_parameter_indices] - proposal[unknown_parameter_indices] -
                                     step_size*proposal_covariance.dot(proposal_log_likelihood_gradient[unknown_parameter_indices])/2 )
        transition_kernel_pdf_forward = -np.transpose(forward_helper_variable).dot(proposal_covariance_inverse).dot(forward_helper_variable)/(2*step_size)
        transition_kernel_pdf_backward = -np.transpose(backward_helper_variable).dot(proposal_covariance_inverse).dot(backward_helper_variable)/(2*step_size)

        # accept-reject step
        if(np.random.uniform() < np.exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            current_log_likelihood_gradient = proposal_log_likelihood_gradient
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position[unknown_parameter_indices]

        # LAP stuff
        if iteration_index%k == 0 and iteration_index > 1 and iteration_index < int(number_of_samples/2):
            r_hat = accepted_moves/iteration_index
            block_sample = mcmc_samples[:iteration_index]
            block_proposal_covariance = np.cov(block_sample.T)
            gamma_1 = 1/np.power(iteration_index,c1)
            gamma_2 = c0*gamma_1
            log_step_size_squared = np.log(np.power(step_size,2)) + gamma_2*(r_hat - 0.574)
            step_size = np.sqrt(np.exp(log_step_size_squared))
            proposal_covariance = proposal_covariance + gamma_1*(block_proposal_covariance - proposal_covariance) + 0.00001*np.eye(number_of_parameters)
            proposal_cholesky = np.linalg.cholesky(proposal_covariance)
            proposal_covariance_inverse = np.linalg.inv(proposal_covariance)
    print("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples

def kalman_specific_likelihood_function(proposed_position,*args):
    """
    Likelihood function called by the generic_mala function inside the kalman_mala function. It takes the
    proposed position and computes the likelihood and its gradient at that point.

    Parameters
    ----------

    proposed_position : numpy array
        Proposed position in parameter space in the MALA function.

    protein_at_observations : numpy array
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        that time. The filter assumes that observations are generated with a fixed, regular time interval.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    Returns
    -------

    log_likelihood : float
        the likelihood evaluated by the Kalman filter at the given proposed position in parameter space.

    log_likelihood_derivative : numpy array
        the derivative of the likelihood with respect to each of the model parameters of the negative feedback
        loop, at the given proposed position in parameter space.

    """
    reparameterised_proposed_position = np.copy(proposed_position)
    reparameterised_proposed_position[[4,5]] = np.exp(reparameterised_proposed_position[[4,5]])
    log_likelihood, log_likelihood_derivative = calculate_log_likelihood_and_derivative_at_parameter_point(args[0],
                                                                                                           reparameterised_proposed_position,
                                                                                                           args[1])
    log_likelihood_derivative[4] = reparameterised_proposed_position[4]*log_likelihood_derivative[4]
    log_likelihood_derivative[5] = reparameterised_proposed_position[5]*log_likelihood_derivative[5]
    return log_likelihood, log_likelihood_derivative

def kalman_mala(protein_at_observations,
                measurement_variance,
                number_of_samples,
                initial_position,
                step_size,
                proposal_covariance=np.eye(1),
                thinning_rate=1,
                known_parameter_dict=None):
    """
    Metropolis adjusted Langevin algorithm which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h/2*proposal_covariance*log_likelihood_gradient + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    protein_at_observations : numpy array
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        that time. The filter assumes that observations are generated with a fixed, regular time interval.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    number_of_samples : integer
        the number of samples the random walk proposes

    initial_position : numpy array
        starting value of the Markov chain

    proposal_covariance: numpy array
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.5

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    known_parameter_dict : dict
        a dict which contains values for parameters where the ground truth is known. The key is the name of the parameter,
        the value is a 2d array, where the first entry is its parameter index in the likelihood function, and the second
        entry is the ground truth.

    Returns
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """
    kalman_args = (protein_at_observations,measurement_variance)
    mcmc_samples = generic_mala(kalman_specific_likelihood_function,
                                number_of_samples,
                                initial_position,
                                step_size,
                                proposal_covariance,
                                thinning_rate,
                                known_parameter_dict,
                                *kalman_args)

    return mcmc_samples

def gamma_mala(shape,
               scale,
               number_of_samples,
               initial_position,
               step_size,
               proposal_covariance=np.eye(1),
               thinning_rate=1):
    """
    Metropolis adjusted Langevin algorithm to sample a Gamma distribution, which takes as input a model and
    returns a N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters.
    Proposals, x', are drawn centered from the current position, x, by
    x + h/2*proposal_covariance*log_likelihood_gradient + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    data : numpy array
        Collection of samples from a Normal distribution with unknown mean and variance

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    number_of_samples : integer
        the number of samples the random walk proposes

    initial_position : numpy array
        starting value of the Markov chain

    proposal_covariance: numpy array
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.5

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    Returns
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """

    def gamma_likelihood_function(proposed_position):
        reparameterised_proposed_position = np.copy(proposed_position)
        reparameterised_proposed_position = np.exp(proposed_position)

        log_likelihood = (shape-1)*np.log(reparameterised_proposed_position) - reparameterised_proposed_position/scale
        log_likelihood_derivative = (shape-1)/reparameterised_proposed_position - 1/scale

        log_likelihood += proposed_position
        log_likelihood_derivative = 1 + reparameterised_proposed_position*log_likelihood_derivative

        return log_likelihood, log_likelihood_derivative

    mcmc_samples = generic_mala(gamma_likelihood_function,
                                number_of_samples,
                                initial_position,
                                step_size,
                                proposal_covariance,
                                thinning_rate)

    return mcmc_samples

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
