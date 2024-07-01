def Function_f_A_ERROR(f, fE, A, AE):
    import numpy as np
    #used to calculate error of fA= f1A1+f2A2+...
    f = np.array(f)
    fE = np.array(fE)
    A = np.array(A)
    AE = np.array(AE)
    
    error_terms = (f * AE)**2 + (A * fE)**2
    return np.sqrt(np.sum(error_terms))
def Function_f_ERROR(errors):
    import numpy as np
    #used to calculate error of f=f1+f2+f3
    errors = np.array(errors)
    return np.sqrt(np.sum(errors**2))
def Function_WEIGHTEDAVERAGEAPHYS(A, C, f, n, P_b, P_n, P_t, sigma_A, sigma_C, sigma_f, sigma_n, sigma_P_b, sigma_P_n, sigma_P_t):
    import numpy as np
    #A is asymmetry, C is sum(f*A), f is fraction
    #n is nitrogen fraction, Pb Pn Pt are beam, neutron, target polarizations
     
    W = (A - C) / ((1 - f) * P_b * P_n * P_t)
    
    partial_A = 1 / ((1 - f)  * P_b * P_n * P_t)
    partial_C = -1 / ((1 - f)  * P_b * P_n * P_t)
    partial_f = (A - C) / (((1 - f)**2)  * P_b * P_n * P_t)
    partial_P_b = (A - C) / ((1 - f)  * (P_b**2) * P_n * P_t)
    partial_P_n = (A - C) / ((1 - f)  * P_b * (P_n**2) * P_t)
    partial_P_t = (A - C) / ((1 - f)  * P_b * P_n * (P_t**2))
    
    sigma_W = np.sqrt((partial_A * sigma_A)**2 + 
                      (partial_C * sigma_C)**2 + 
                      (partial_f * sigma_f)**2 + 
                      (partial_P_b * sigma_P_b)**2 + 
                      (partial_P_n * sigma_P_n)**2 + 
                      (partial_P_t * sigma_P_t)**2)
    
    return W, sigma_W