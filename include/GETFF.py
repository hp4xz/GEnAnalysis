############################/
############################/
## Parameterized Form Factor Central Value and Error
############################/
## ID = 1 for GEp, 2 for GMp, 3 for GEn, 4 for GMn,
## Q2 in GeV^2
##
# The parameterization formula returns the uncertainty devided by G(0)*GD, where
#  GD(Q2) = 1./(1+Q2/0.71)^2
# and GEp(0) = 1, GMp(0) = 2.79284356, GEn(0) = 1, GMn(0) = -1.91304272,
#
# The parameterization formula for the Form Factor value is:
#  $$ GN(z) = sum_{i=0}^{N=12}(a_i * z^i)
# Note that the return value has been divided by (G(Q2=0)*G_Dip)
#
# The parameterization formula for the Form Factor error is:
# $$ log_{10}\frac{\delta G}{G_D} = (L+c_0)\Theta_a(L_1-L)
#                                 +\sum_{i=1}^{N}(c_i+d_i L)[\Theta_a(L_i-L)-\Theta_a(L_{i+1}-L)]
#                                 +log_{10}(E_{\inf})\Theta_a(L-L_{N+1})$$
# where $L=log_{10}(Q^2)$, $\Theta_{a}(x)=[1+10^{-ax}]^{-1}$. $a=1$.

def Function_GETFF(kID, kQ2):
    import numpy as np
    from math import sqrt,log10
# {{{
    ### GEp->kID=1, GMp->kID=2, GEn->kID=3, GMn->kID=4
    if kID<1 or kID>4:
        print ('*** ERROR***, kID is not any of [1->GEp, 2->GMp, 3->GEn, 4->GMn]')
        return -1000, -1000

    #################################################
    #### z-Expansion Parameters for Form Factor Values
    #################################################{{{
    GN_Coef_Fit = np.zeros((4,13), dtype=float)
    GN_Coef_Fit[0] = np.array([0.239163298067,  -1.10985857441,  1.44438081306,  0.479569465603,  -2.28689474187,  1.12663298498,  1.25061984354,  -3.63102047159,  4.08221702379,  0.504097346499,  -5.08512046051,  3.96774254395,  -0.981529071103]) #GEp
    GN_Coef_Fit[1] = np.array([0.264142994136, -1.09530612212, 1.21855378178, 0.661136493537, -1.40567892503, -1.35641843888, 1.44702915534, 4.2356697359, -5.33404565341, -2.91630052096, 8.70740306757, -5.70699994375, 1.28081437589]) #GMp
    GN_Coef_Fit[2] = np.array([0.048919981379,-0.064525053912,-0.240825897382,0.392108744873, 0.300445258602,-0.661888687179,-0.175639769687, 0.624691724461,-0.077684299367,-0.236003975259, 0.090401973470, 0.0, 0.0]) #GEn
    GN_Coef_Fit[3] = np.array([0.257758326959,-1.079540642058, 1.182183812195,0.711015085833,-1.348080936796,-1.662444025208, 2.624354426029, 1.751234494568,-4.922300878888, 3.197892727312,-0.712072389946, 0.0, 0.0]) #GMn
    #}}}

#################################################
#### Parameters for Form Factor Errors
#################################################{{{
    parL = np.zeros((4,2), dtype=float)
    parM = np.zeros((4,15), dtype=float)
    parH = np.zeros((4,3), dtype=float)
    ## GEp:
    parL[0] = np.array([-0.97775297,  0.99685273]) #Low-Q2
    parM[0] = np.array([ -1.97750308e+00,  -4.46566998e-01,   2.94508717e-01,   1.54467525e+00,
        9.05268347e-01,  -6.00008111e-01,  -1.10732394e+00,  -9.85982716e-02,
        4.63035988e-01,   1.37729116e-01,  -7.82991627e-02,  -3.63056932e-02,
        2.64219326e-03,   3.13261383e-03,   3.89593858e-04 ]) #Mid-Q2:
    parH[0] = np.array([ 0.78584754,  1.89052183, -0.4104746]) #High-Q2

    #GMp:
    parL[1] = np.array([-0.68452707,  0.99709151]) #Low-Q2
    parM[1] = np.array([ -1.76549673e+00,   1.67218457e-01,  -1.20542733e+00,  -4.72244127e-01,
        1.41548871e+00,   6.61320779e-01,  -8.16422909e-01,  -3.73804477e-01,
        2.62223992e-01,   1.28886639e-01,  -3.90901510e-02,  -2.44995181e-02,
        8.34270064e-04,   1.88226433e-03,   2.43073327e-04]) #Mid-Q2:
    parH[1] = np.array([  0.80374002,  1.98005828, -0.69700928]) #High-Q2
    
    #GEn:
    parL[2] = np.array([-2.02311829, 1.00066282]) #Low-Q2
    parM[2] = np.array([-2.07343771e+00,   1.13218347e+00,   1.03946682e+00,  -2.79708561e-01,
        -3.39166129e-01,   1.98498974e-01,  -1.45403679e-01,  -1.21705930e-01,
        1.14234312e-01,   5.69989513e-02,  -2.33664051e-02,  -1.35740738e-02,
        7.84044667e-04,   1.19890550e-03,   1.55012141e-04,]) #Mid-Q2:
    parH[2] = np.array([0.4553596, 1.95063341, 0.32421279]) #High-Q2:

    #GMn:
    parL[3] = np.array([-0.20765505, 0.99767103]) #Low-Q2:
    parM[3] = np.array([  -2.07087611e+00,   4.32385770e-02,  -3.28705077e-01,   5.08142662e-01,
        1.89103676e+00,   1.36784324e-01,  -1.47078994e+00,  -3.54336795e-01,
        4.98368396e-01,   1.77178596e-01,  -7.34859451e-02,  -3.72184066e-02,
        1.97024963e-03,   2.88676628e-03,   3.57964735e-04]) #Mid-Q2:
    parH[3] = np.array([ 0.50859057, 1.96863291, 0.2321395]) #High-Q2
    ##}}}

    ## Apply the z-expansion formula
    tcut = 0.0779191396
    t0 = -0.7
    z = (sqrt(tcut+kQ2)-sqrt(tcut-t0))/(sqrt(tcut+kQ2)+sqrt(tcut-t0)) 
    GNQ2 = np.array([GN_Coef_Fit[kID-1][i]*(z**i) for i in range(0, len(GN_Coef_Fit[kID-1]))]).sum() 
    GDip= 1./(1. + kQ2/0.71)**2
    GNGD_Fit = GNQ2 / GDip #Note that the GN_Coef_Fit has been divided by mu_p or mu_n for GMp and GMn

    ## Apply the parameterization formula for error
    lnQ2 = log10(kQ2)
    lnGNGD_Err=0.0
    if kQ2<1e-3:
        lnGNGD_Err = parL[kID-1][0] + parL[kID-1][1]*lnQ2
    elif kQ2>1e2:
        lnGNGD_Err = parH[kID-1][0]*np.sqrt(lnQ2 - parH[kID-1][1]) + parH[kID-1][2]
    else:
        lnGNGD_Err = np.array([parM[kID-1][i]*(lnQ2**i) for i in range(0, len(parM[kID-1]))]).sum() 
    GNGD_Err = 10.**(lnGNGD_Err)    ##LOG10(dG/G(0)/GD)

    return GNGD_Fit, GNGD_Err
# }}}
