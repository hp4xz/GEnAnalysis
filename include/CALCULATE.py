
def Function_CALCGEN(config,A,AE):
    from math import pi
    import numpy as np
    #in GeV^2
    
    m=.939565
    muN=-1.9103
    if config=="2":
        Q2=3
        tau=Q2/(4*m**2)
        theta=29.5*pi/180
    if config=="3":
        Q2=6.83
        #Q2=6.62 #seans number
        tau=Q2/(4*m**2)
        theta=36.5*pi/180
    if config=="4":
        Q2=9.82
        tau=Q2/(4*m**2)
        theta=35*pi/180
    a=A
    b=2*(np.sqrt(tau*(tau+1))*np.tan(theta/2))
    c=A*(tau+2*tau*(1+tau)*np.tan(theta/2)**2)
    aE=AE
    cE=AE*(tau+2*tau*(1+tau)*np.tan(theta/2)**2)
    
    R,RE=Rcalc(a, b, c, aE, cE)
    return R,RE

def Rcalc(a, b, c, aE, cE):
    import numpy as np
    R = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    
    # Calculate the partial derivatives
    p1=(-b**2)/np.sqrt(b**2-4*a*c)
    p2=(2*a*c)/np.sqrt(b**2-4*a*c)
    p3=b
    partial_a = (p1+p2+p3)/(2*a**2)
    partial_c = -1 / (np.sqrt(b**2 - 4 * a * c))
    
    # Calculate the propagated error
    RE = np.sqrt((partial_a * aE)**2 + (partial_c * cE)**2)
    
    return R,RE

def Function_GENWORLDFROMQ2(Q2):
    import pandas as pd
    import numpy as np
    row=main(Q2)
    R=np.round(row[1]/row[4],4)
    return R
def RcalcINCORRECT(tau, theta, A, AE):
    import numpy as np
    import pandas as pd
    n1 = -np.sqrt(tau * (tau + 1)) * np.tan(theta / 2)
    n2 = np.sqrt(tau * (tau + 1) * np.tan(theta / 2)**2 - A**2 * (tau + 2 * tau * (1 + tau) * np.tan(theta / 2)**2))
    
    R_plus = (n1 + n2) / A
    R_minus = (n1 - n2) / A
    
    partial_n2_A = (-A * (tau + 2 * tau * (1 + tau) * np.tan(theta / 2)**2)) / n2
    sigma_n2 = np.abs(partial_n2_A) * AE
    
    sigma_R_plus = R_plus * np.sqrt((sigma_n2 / (n1 + n2))**2 + (AE / A)**2)
    sigma_R_minus = R_minus * np.sqrt((sigma_n2 / (n1 - n2))**2 + (AE / A)**2)

       
    
    return R_plus, sigma_R_plus

def load_data(file_path):
    import pandas as pd
    """
    Load the data from the given file path into a pandas DataFrame.
    """
    columns = ["Q2", "GEp/GD", "dGEp/GD", "dGEp_Par/GD", "GMp/mu_p/GD", "dGMp/mu_p/GD", "dGMp_Par/mu_p/GD"]
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=columns)
    return data

def find_closest_row(data, input_Q2):
    import pandas as pd
    """
    Find the row in the data with the Q2 value closest to the input_Q2.
    """
    closest_row = data.iloc[(data['Q2'] - input_Q2).abs().argmin()]
    return closest_row

def main(input_Q2):
    file_path = '../DB/neutron_lookup.dat'
    data = load_data(file_path)
    closest_row = find_closest_row(data, input_Q2)
    return closest_row
