import hes_inference
import numpy as np

protein_at_observations = np.zeros((50,2))
protein_at_observations[:,0] = np.arange(0,15*50,15)
protein_at_observations[:,1] = np.arange(0,500*50,500)

model_parameters = np.array([10000,4,np.log(2)/30,np.log(2)/90,1,1,15])
measurement_variance = 100**2

hes_inference.kalman_filter(protein_at_observations,model_parameters,measurement_variance)
