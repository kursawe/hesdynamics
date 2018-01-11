import PyDDE
import numpy as np
import scipy.signal


def generate_single_trajectory(duration, 
                               repression_threshold = 10000,
                               hill_coefficient = 5,
                               mRNA_degradation_rate = np.log(2)/30,
                               protein_degradation_rate = np.log(2)/90, 
                               basal_transcription_rate = 1,
                               translation_rate = 1,
                               repression_delay = 29,
                               initial_mRNA = 0,
                               initial_protein = 0):
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
        
    repression_delay : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.

    Returns
    -------
    
    trace : ndarray
        2 dimenstional array, first column is time, second column mRNA number,
        third column is Hes5 protein copy number
    '''
    
    hes5_dde = PyDDE.dde()
    initial_condition = np.array([initial_mRNA,initial_protein]) 
    # The coefficients (constants) in the equations 
    parameters = np.array([repression_threshold,  
                           hill_coefficient, 
                           mRNA_degradation_rate,
                           protein_degradation_rate, 
                           basal_transcription_rate, 
                           translation_rate, 
                           repression_delay]) 

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
                            repression_delay]
        containing the value of these parameters.
    
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

    mRNA = float(y[0])
    protein = float(y[1])
    
    if (time>time_delay):
        past_protein = PyDDE.pastvalue(1,time-time_delay,0)
    elif time>0.0:
        past_protein = PyDDE.pastvalue(1,0.0,0)
    else:
        past_protein = protein

    hill_function_value = 1.0/(1.0+pow(past_protein/repression_threshold,hill_coefficient))
    dmRNA = basal_transcription_rate*hill_function_value-mRNA_degradation_rate*mRNA;
    dprotein = translation_rate*mRNA - protein_degradation_rate*protein;
    
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

# plt.show()
#     P0=par(1);
#     NP=par(2);
#     MUM = par(3);
#     MUP= par(4);
#     ALPHAP=par(6);
#     ALPHAM=par(5);
#     tau = par(7);
# 
# 
# 
# mout = zeros(totalreps,length(Output_Times));
# pout = zeros(totalreps,length(Output_Times));
# 
# parfor reps=1:totalreps %parfor
# % disp(reps)    
# j_t_next = 1;
# 
# moutcurr = zeros(1,length(Output_Times));
# poutcurr = zeros(1,length(Output_Times));
#     
# iter = 1;
# rlist = [];
# 
# % sets initial values
# m=mstart;
# mn=0;
# p=pstart;
# pn=0;
# t=0;tn=0;
# rlist=[];
# 
# %When rlist is empty, there is a different process:
# %We follow this process until rlist is no longer empty
# a1 = MUM*m;
# a2 = MUP*p; 
# a3 = ALPHAP*m;
# a4 = N*ALPHAM/(1 + ((p/N)/P0)^NP);
# while t<Output_Times(end) % runs until p hits a threshold            
#             a0=a1+a2+a3+a4;
#             r1=rand(1);
#             r2=rand(1);
#             dt=(1/a0)*log(1/r1);
#             if numel(rlist)>0 && t<=rlist(1) && rlist(1)<=(t+dt)
#             %if le(t(iter),rlist(1))*le(rlist(1),(t(iter)+dt))
#                 mn= m+1;
#                 pn = p;
#                 tn = rlist(1);
#                 rlist(1)=[];
#                 a1=(MUM)*(mn);
#                 a3=ALPHAP*mn;                
#             else
#                 if r2*a0<=a1
#                     mn= m-1;
#                     pn = p;
#                     a1=MUM*(mn);
#                     a3=ALPHAP*mn;                    
#                 %elseif le(a1,r2*a0)*le(r2*a0,(a1+a2))
#                 elseif a1<=r2*a0 && r2*a0<=(a1+a2)
#                     mn = m;
#                     pn = p-1;
#                     a2=MUP*pn;                    
#                     a4=N*ALPHAM/(1 + ((pn/N)/P0)^NP);
#                 elseif (a1+a2)<=r2*a0 && r2*a0<=(a1+a2+a3)
#                     mn = m;
#                     pn = p+1;                    
#                     a2=MUP*pn;                    
#                     a4=  N*ALPHAM/(1 + ((pn/N)/P0)^NP);                
#                 else
#                     rlist = [rlist (t+tau)];
#                     mn = m;
#                     pn = p;                    
#                 end
#                 tn = t+dt;
#             end
#         iter = iter + 1;
#         t=tn;
#         m =mn;
#         p =pn;        
#         while j_t_next<=length(Output_Times)&&t>Output_Times(j_t_next)
#             moutcurr(1,j_t_next) = m;
#             poutcurr(1,j_t_next) = p;            
#             j_t_next=j_t_next+1;
#         end
# end 
# 
# mout(reps,:) =moutcurr(1,:); 
# pout(reps,:) =poutcurr(1,:);
#    
# end                   
#                                      
# end
 