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

def Function_STATERROR(Araw,fN,Pb,Pt,PN,ArawSig,fNSig,PbSig,PtSig,PNSig):
    import numpy as np
    S=Araw/(fN*Pb*Pt*PN)
    partial_Araw=1/(fN*Pb*Pt*PN)
    partial_fN=-Araw/(fN**2*Pb*Pt*PN)
    partial_Pb=-Araw/(fN*Pb**2*Pt*PN)
    partial_Pt=-Araw/(fN*Pt**2*Pb*PN)
    partial_PN=-Araw/(fN*PN**2*Pt*Pb)
    S_sig=np.sqrt((partial_Araw*ArawSig)**2+
                 (partial_fN*fNSig)**2+
                  (partial_Pb*PbSig)**2+
                  (partial_Pt*PtSig)**2+
                  (partial_PN*PNSig)**2)
    return S_sig
def Function_SYSERROR(fA,fN,Pb,Pt,PN,fASig,fNSig,PbSig,PtSig,PNSig):
    import numpy as np
    S=fA/(fN*Pb*Pt*PN)
    partial_fA=1/(fN*Pb*Pt*PN)
    partial_fN=-fA/(fN**2*Pb*Pt*PN)
    partial_Pb=-fA/(fN*Pb**2*Pt*PN)
    partial_Pt=-fA/(fN*Pt**2*Pb*PN)
    partial_PN=-fA/(fN*PN**2*Pt*Pb)
    S_sig=np.sqrt((partial_fA*fASig)**2+
                 (partial_fN*fNSig)**2+
                  (partial_Pb*PbSig)**2+
                  (partial_Pt*PtSig)**2+
                  (partial_PN*PNSig)**2)
    return S_sig