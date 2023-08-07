import numpy as np
from tensorflow.keras import metrics


logc = np.log(2 * np.pi).astype(np.float32)

def X_normal_logpdf(x, mu, lsgms,backend):
    lsgms = backend.flatten(lsgms)   
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((x - mu)**2 / backend.exp(lsgms)), axis=-1)

def Y_normal_logpdf(y, mu, lsgms,backend):  
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((y - mu)**2 / backend.exp(lsgms)), axis=-1)
   
def obj(X, X_mu,Y, Y_mu, Y_lsgms,Z_mu,Z_lsgms,backend):
    X = backend.flatten(X)
    X_mu = backend.flatten(X_mu)
    
    Lp = 0.5 * backend.mean( 1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
    
    Lx =  - metrics.binary_crossentropy(X, X_mu) # Pixels have a Bernoulli distribution  
               
    Ly =  Y_normal_logpdf(Y, Y_mu, Y_lsgms) # Voxels have a Gaussian distribution
        
    lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
    
    cost = - lower_bound
              
    return  cost 