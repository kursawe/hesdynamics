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
    
    state_space_mean : numpy array.
        An array of length n, which gives the number of observation time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein.
    
    state_space_variance : numpy array.
        An array of dimension 2n x 2n. 
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]
    
    """