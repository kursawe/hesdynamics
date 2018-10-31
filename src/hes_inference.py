## need to write code here
def kalman_filter(protein_at_observation,model_parameters):
    """
    Perform Kalman-Bucy filter based on observation of protein
    copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).
    
    Parameters
    ----------
    
    protein_at_observation : numpy array.
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
    
    updated_state_space_mean : numpy array.
        An array of length n, which gives the number of observation time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein.
    
    updated_state_space_variance : numpy array.
        An array of dimension 2n x 2n. 
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]
    
    """
    
    ## first we need \rho_{t+\delta t-\tau:t+\delta t} and P_{t+\delta t-\tau:t+\delta t},
    ## which can be obtained using the differential equations in supplementary section 4.
    ## For the time being we will call these 'state_space_mean' and 'state_space_variance',
    ## and they will be updated in the following way.
    
    # n is the number of observation time points.
    n = len(protein_at_observation[:,0])
    
    # initialise updated mean and variance arrays.
    updated_state_space_mean = state_space_mean
    updated_state_space_variance = state_space_variance
    
    # need to define C (the coefficient of adaptation) somewhere
    # also there are a few things wrong with this. I think the right hand side should also
    # use the updated mean and variance, and also the observation y_{t+\delta t} in the first
    # equation is wrong.
    for i in range(1,n+1):
    	for j in range(i-1,-1,-1):
    		updated_state_space_mean[j,(1,2)] = (state_space_mean[j,(1,2)] +
    		*(protein_at_observation[j,1]-state_space_mean[i,(1,2)]))
			
			updated_state_space_variance[j,(1,2)] = (state_space_variance[j,(1,2)] - C*state_space_variance[j,(1,2)])
	
	return updated_state_space_mean, updated_state_space_variance
	
	
	
	
	
	
	
    
    
    
    
    
