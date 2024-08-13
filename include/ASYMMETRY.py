def Function_ASYMMETRYSENS(config,cut,value,cutstyle=0):
    #imports
    import ROOT as r
    import math
    import array
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    #______________Add include directory_______________
    current_dir = os.getcwd()
    include_dir = os.path.join(current_dir, '../include')
    sys.path.insert(0, include_dir)
    #__________________________________________________

    import CONFIG
    import DBPARSE
    #cuts
    W2min=CONFIG.Function_JSON("W2min",f"../config/cuts{config}.cfg")
    W2max=CONFIG.Function_JSON("W2max",f"../config/cuts{config}.cfg")
    dxmin=CONFIG.Function_JSON("dxmin",f"../config/cuts{config}.cfg")
    dxmax=CONFIG.Function_JSON("dxmax",f"../config/cuts{config}.cfg")
    dymin=CONFIG.Function_JSON("dymin",f"../config/cuts{config}.cfg")
    dymax=CONFIG.Function_JSON("dymax",f"../config/cuts{config}.cfg")
    dybgmin=CONFIG.Function_JSON("dybgmin",f"../config/cuts{config}.cfg")
    dybgmax=CONFIG.Function_JSON("dybgmax",f"../config/cuts{config}.cfg")
    coinmin=CONFIG.Function_JSON("coinmin",f"../config/cuts{config}.cfg")
    coinmax=CONFIG.Function_JSON("coinmax",f"../config/cuts{config}.cfg")
    #____________Varying Cut__________________
    if cut=="w2":
        W2max=value[1]
        W2min=value[0]
    elif cut=="dy":
        dymax=value[1]
        dymin=value[0]
    elif cut=="dx":
        dxmax=value[1]
        dxmin=value[0]        
    elif cut=="coin":
        coinmin=value[1]
        coinmax=value[0]
    else:
        print(f"Invalid Cut ({cut}) Options are: w2,dy,dx,coin with value=[min,max]")
    #____________Varying Cut__________________
    
    
    r.gErrorIgnoreLevel = r.kError  # Suppress Info and Warning messages
    r.gStyle.SetOptStat(0)
    r.gStyle.SetOptFit(1)
    rootfilenp = f"../outfiles/Pass1/QE_data_GEN{config}_sbs100p_nucleon_np_model2.root"
    rootfilep = f"../outfiles/Pass1/QE_sim_GEN{config}_sbs100p_nucleon_np_model2.root"
    print(f'Config: {config}')
    if config == "4":
        config = "4b"

        # Load the TTrees
    B = r.TChain("Tout")
    C = r.TChain("Tout")

    dx_np = array.array('d', [0])
    dx_p = array.array('d', [0])
    dy_np = array.array('d', [0])
    dy_p = array.array('d', [0])
    W2_np = array.array('d', [0])
    W2_p = array.array('d', [0])
    coin_np = array.array('d', [0])
    weight = array.array('d', [0])
    helicity_p = array.array('i', [0])
    IHWP_p = array.array('i', [0])
    runnum_p = array.array('i', [0])
    helicity_np = array.array('i', [0])
    IHWP_np = array.array('i', [0])
    runnum_np = array.array('i', [0])

    # Load the TTrees

    C.Add(rootfilenp)
    B.Add(rootfilep)

    # Disable all branches initially
    C.SetBranchStatus("*", 0)
    B.SetBranchStatus("*", 0)

    # Enable specific branches
    branches = ["dx", "dy", "W2", "helicity", "IHWP", "runnum", "coinCut", "coin_time"]
    b2=["dx", "dy", "W2"]
    for branch in branches:
        C.SetBranchStatus(branch, 1)
    for branch in b2:
        B.SetBranchStatus(branch, 1)

    B.SetBranchStatus("weight", 1)

    # Set branch addresses
    C.SetBranchAddress("dx", dx_np)
    B.SetBranchAddress("dx", dx_p)
    C.SetBranchAddress("dy", dy_np)
    B.SetBranchAddress("dy", dy_p)
    C.SetBranchAddress("W2", W2_np)
    B.SetBranchAddress("W2", W2_p)
    C.SetBranchAddress("helicity", helicity_np)
    C.SetBranchAddress("IHWP", IHWP_np)
    C.SetBranchAddress("coin_time", coin_np)
    C.SetBranchAddress("runnum", runnum_np)
    B.SetBranchAddress("weight", weight)
    


    
    nEntries_np = C.GetEntries()
    nEntries_p = B.GetEntries()
    C.GetEntry(0)
    thisnum = runnum_np[0]
    nplus_np = 0
    pplus_np = 0
    nminus_np = 0
    pminus_np = 0
    pYield=[]
    nYield=[]
    runnumVec = []
    runnumA_p = []
    runnumA_n = []
    runnumA_targetpol = []
    runnumA_beampol = []
    Err_A_n = []
    Err_A_p = []
    QE = 0
    analyze=True
    # Loop over the entries
    
    rxn3 = (dxmax - dxmin) / 2.0
    ryn3 = (dymax - dymin) / 2.0
    x0_n3 = (dxmax + dxmin) / 2.0
    y0_3 = (dymax + dymin) / 2.0
    for i in range(nEntries_np):
        C.GetEntry(i)
        if runnum_np[0] > 2165 and runnum_np[0]<4470:
            if thisnum == runnum_np[0]:
                if IHWP_np[0] == 1:
                    helicity_np[0] *= 1

                elif IHWP_np[0] == -1:
                    helicity_np[0] *= -1
                else:
                    continue
                #____________CUTS_______________________________   
                ncut = (dx_np[0] - x0_n3)**2 / rxn3**2 + (dy_np[0] - y0_3)**2 / ryn3**2 <= 1
                ycut = dymin < dy_np[0] < dymax

                xcutn = dxmin < dx_np[0] < dxmax
                
                coin_cut = coinmin < coin_np[0] < coinmax

                W2cut=W2min<W2_np[0]<W2max
                #________________________________________________ 
                
               
                if cutstyle==0:
                
                    if coin_cut and W2cut and runnum_np[0] > 2165 and ycut and xcutn:
                        QE += 1

                        if helicity_np[0] == 1:
                            nplus_np += 1
                        if helicity_np[0] == -1:
                            nminus_np += 1
                if cutstyle==1:
                
                    if coin_cut and W2cut and runnum_np[0] > 2165 and ncut:
                        QE += 1

                        if helicity_np[0] == 1:
                            nplus_np += 1
                        if helicity_np[0] == -1:
                            nminus_np += 1
        
            else:
                analyze = True
                if nplus_np<1 or nminus_np<1:
                    analyze = False
                if analyze:
                    n_Asym = (nplus_np - nminus_np) * 1.0 / (nplus_np + nminus_np)
                    #print(f"Asymmetry for run number {thisnum}: {n_Asym} {p_Asym}")
                    if runnum_np[0] > 2165 and runnum_np[0]< 4470:
                        runnumVec.append(runnum_np[0])
                        runnumA_n.append(n_Asym)
                        nYield.append(nplus_np+nminus_np)
                        Err_A_n.append(2 * math.sqrt(nplus_np * nminus_np) / (nplus_np + nminus_np)**(3/2))
                        thisnum = runnum_np[0]
                QE = 0
                nminus_np = 0
                nplus_np = 0
            
                thisnum = runnum_np[0]
        else:
            thisnum = runnum_np[0]
            
    beamPol=np.empty(0);
    he3Pol=np.empty(0);
    for i in range(0,len(runnumVec)):
        beamPol=np.append(beamPol,DBPARSE.Function_RETURNPROCESSEDBEAMPOL(runnumVec[i]))
        he3Pol=np.append(he3Pol,DBPARSE.Function_RETURNPROCESSEDHE3POL(runnumVec[i]))
    return runnumVec,runnumA_n,Err_A_n,nYield,he3Pol,beamPol,cut,value

def Function_ASYMMETRY(config):
    #imports
    import ROOT as r
    import math
    import array
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    #______________Add include directory_______________
    current_dir = os.getcwd()
    include_dir = os.path.join(current_dir, '../include')
    sys.path.insert(0, include_dir)
    #__________________________________________________

    import CONFIG
    import DBPARSE
    #cuts
    W2min=CONFIG.Function_JSON("W2min",f"../config/cuts{config}.cfg")
    W2max=CONFIG.Function_JSON("W2max",f"../config/cuts{config}.cfg")
    dxmin=CONFIG.Function_JSON("dxmin",f"../config/cuts{config}.cfg")
    dxmax=CONFIG.Function_JSON("dxmax",f"../config/cuts{config}.cfg")
    dymin=CONFIG.Function_JSON("dymin",f"../config/cuts{config}.cfg")
    dymax=CONFIG.Function_JSON("dymax",f"../config/cuts{config}.cfg")
    dybgmin=CONFIG.Function_JSON("dybgmin",f"../config/cuts{config}.cfg")
    dybgmax=CONFIG.Function_JSON("dybgmax",f"../config/cuts{config}.cfg")
    coinmin=CONFIG.Function_JSON("coinmin",f"../config/cuts{config}.cfg")
    coinmax=CONFIG.Function_JSON("coinmax",f"../config/cuts{config}.cfg")

    
    
    r.gErrorIgnoreLevel = r.kError  # Suppress Info and Warning messages
    r.gStyle.SetOptStat(0)
    r.gStyle.SetOptFit(1)
    rootfilenp = f"../outfiles/Pass1/QE_data_GEN{config}_sbs100p_nucleon_np_model2.root"
    rootfilep = f"../outfiles/Pass1/QE_sim_GEN{config}_sbs100p_nucleon_np_model2.root"
    print(f'Config: {config}')
    if config == "4":
        config = "4b"

        # Load the TTrees
    B = r.TChain("Tout")
    C = r.TChain("Tout")

    dx_np = array.array('d', [0])
    dx_p = array.array('d', [0])
    dy_np = array.array('d', [0])
    dy_p = array.array('d', [0])
    W2_np = array.array('d', [0])
    W2_p = array.array('d', [0])
    coin_np = array.array('d', [0])
    weight = array.array('d', [0])
    helicity_p = array.array('i', [0])
    IHWP_p = array.array('i', [0])
    runnum_p = array.array('i', [0])
    helicity_np = array.array('i', [0])
    IHWP_np = array.array('i', [0])
    runnum_np = array.array('i', [0])

    # Load the TTrees

    C.Add(rootfilenp)
    B.Add(rootfilep)

    # Disable all branches initially
    C.SetBranchStatus("*", 0)
    B.SetBranchStatus("*", 0)

    # Enable specific branches
    branches = ["dx", "dy", "W2", "helicity", "IHWP", "runnum", "coinCut", "coin_time"]
    b2=["dx", "dy", "W2"]
    for branch in branches:
        C.SetBranchStatus(branch, 1)
    for branch in b2:
        B.SetBranchStatus(branch, 1)

    B.SetBranchStatus("weight", 1)

    # Set branch addresses
    C.SetBranchAddress("dx", dx_np)
    B.SetBranchAddress("dx", dx_p)
    C.SetBranchAddress("dy", dy_np)
    B.SetBranchAddress("dy", dy_p)
    C.SetBranchAddress("W2", W2_np)
    B.SetBranchAddress("W2", W2_p)
    C.SetBranchAddress("helicity", helicity_np)
    C.SetBranchAddress("IHWP", IHWP_np)
    C.SetBranchAddress("coin_time", coin_np)
    C.SetBranchAddress("runnum", runnum_np)
    B.SetBranchAddress("weight", weight)
    


    
    nEntries_np = C.GetEntries()
    nEntries_p = B.GetEntries()
    C.GetEntry(0)
    thisnum = runnum_np[0]
    nplus_np = 0
    pplus_np = 0
    nminus_np = 0
    pminus_np = 0
    pYield=[]
    nYield=[]
    runnumVec = []
    runnumA_p = []
    runnumA_n = []
    runnumA_targetpol = []
    runnumA_beampol = []
    Err_A_n = []
    Err_A_p = []
    QE = 0
    analyze=True
    # Loop over the entries
    
    

    for i in range(nEntries_np):
        C.GetEntry(i)
        if runnum_np[0] > 2165 and runnum_np[0]<4470:
            if thisnum == runnum_np[0]:
                if IHWP_np[0] == 1:
                    helicity_np[0] *= 1

                elif IHWP_np[0] == -1:
                    helicity_np[0] *= -1
                else:
                    continue
                #____________CUTS_______________________________      
                ycut = dymin < dy_np[0] < dymax
                #bgycut=dybgmin<dy_np[0]<dybgmax
                #coin_cut = coinmin < coin_np[0] < coinmax
                #W2cut=W2min < W2_np[0] < W2max
                xcutn = dxmin < dx_np[0] < dxmax
                
                coin_cut = coinmin < coin_np[0] < coinmax
                #ycut=abs(dy_np[0])<dymax
                #xcutn=abs(dx_np[0])<dxmax
                W2cut=W2min<W2_np[0]<W2max
                #________________________________________________ 
                
               
                
                
                if coin_cut and W2cut and runnum_np[0] > 2165 and ycut and xcutn:
                    QE += 1

                    if helicity_np[0] == 1:
                        nplus_np += 1
                    if helicity_np[0] == -1:
                        nminus_np += 1
          
        
            else:
                analyze = True
                if nplus_np<1 or nminus_np<1:
                    analyze = False
                if analyze:
                    n_Asym = (nplus_np - nminus_np) * 1.0 / (nplus_np + nminus_np)
                    #print(f"Asymmetry for run number {thisnum}: {n_Asym} {p_Asym}")
                    if runnum_np[0] > 2165 and runnum_np[0]< 4470:
                        runnumVec.append(runnum_np[0])
                        runnumA_n.append(n_Asym)
                        nYield.append(nplus_np+nminus_np)
                        Err_A_n.append(2 * math.sqrt(nplus_np * nminus_np) / (nplus_np + nminus_np)**(3/2))
                        thisnum = runnum_np[0]
                QE = 0
                nminus_np = 0
                nplus_np = 0
            
                thisnum = runnum_np[0]
        else:
            thisnum = runnum_np[0]
            
    beamPol=np.empty(0);
    he3Pol=np.empty(0);
    for i in range(0,len(runnumVec)):
        beamPol=np.append(beamPol,DBPARSE.Function_RETURNPROCESSEDBEAMPOL(runnumVec[i]))
        he3Pol=np.append(he3Pol,DBPARSE.Function_RETURNPROCESSEDHE3POL(runnumVec[i]))
    return runnumVec,runnumA_n,Err_A_n,nYield,he3Pol,beamPol
def Function_FITDXSENS(config,cut,value):
    import ROOT as r
    import math
    import array
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    #______________Add include directory_______________
    current_dir = os.getcwd()
    include_dir = os.path.join(current_dir, '../include')
    sys.path.insert(0, include_dir)
    #__________________________________________________

    import CONFIG
    import DBPARSE
    import UTILITIES
    from SIMFITS import DistributionFits
    from ROOT import gStyle, TChain, TH1F, TCanvas, TLegend
    W2min=CONFIG.Function_JSON("W2min",f"../config/cuts{config}.cfg")
    W2max=CONFIG.Function_JSON("W2max",f"../config/cuts{config}.cfg")
    dxmin=CONFIG.Function_JSON("dxmin",f"../config/cuts{config}.cfg")
    dxmax=CONFIG.Function_JSON("dxmax",f"../config/cuts{config}.cfg")
    dymin=CONFIG.Function_JSON("dymin",f"../config/cuts{config}.cfg")
    dymax=CONFIG.Function_JSON("dymax",f"../config/cuts{config}.cfg")
    dybgmin=CONFIG.Function_JSON("dybgmin",f"../config/cuts{config}.cfg")
    dybgmax=CONFIG.Function_JSON("dybgmax",f"../config/cuts{config}.cfg")
    coinmin=CONFIG.Function_JSON("coinmin",f"../config/cuts{config}.cfg")
    coinmax=CONFIG.Function_JSON("coinmax",f"../config/cuts{config}.cfg")
    nBins=CONFIG.Function_JSON("nBins",f"../config/cuts{config}.cfg")

    #____________Varying Cut__________________
    if cut=="w2":
        W2max=value[1]
        W2min=value[0]
    elif cut=="dy":
        dymax=value[1]
        dymin=value[0]
    elif cut=="dx":
        dxmax=value[1]
        dxmin=value[0]        
    elif cut=="coin":
        coinmin=value[1]
        coinmax=value[0]
    else:
        print(f"Invalid Cut ({cut}) Options are: w2,dy,dx,coin with value=[min,max]")
    #____________Varying Cut__________________
    
    r.gErrorIgnoreLevel = r.kError  # Suppress Info and Warning messages
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(1)

    #_____bring in config values______#
    coinVector=CONFIG.Function_JSON("GEN"+config,"../config/coin.cfg")
    he3spot=CONFIG.Function_JSON("GEN"+config+"He3","../config/spotsize.cfg")
    he3spotsim=CONFIG.Function_JSON("GEN"+config+"He3sim","../config/spotsize.cfg")
    
    #here, set coincidence cut and spot cut values
    coinSigma=2.5
    coin_low=coinVector[0]-coinSigma*coinVector[1]
    coin_high=coinVector[0]+coinSigma*coinVector[1]

    #spot imports for dx,dy high and low
    
    hiydata=he3spot[1];
    lowydata=he3spot[0];
    
    hixdatan=he3spotsim[7];
    lowxdatan=he3spotsim[6];
    hixdatap=he3spotsim[5];
    lowxdatap=he3spotsim[4];
    
    hix_n_3 = he3spotsim[7]
    lowx_n_3 = he3spotsim[6]
    hix_p_3 = he3spotsim[5]
    lowx_p_3 = he3spotsim[4]
    hiy_p3 = he3spotsim[1]
    lowy_p3 = he3spotsim[0]
    hiy_n3 = he3spotsim[3]
    lowy_n3 = he3spotsim[2]
    

    rxn3 = (hix_n_3 - lowx_n_3) / 2.0
    rxp3 = (hix_p_3 - lowx_p_3) / 2.0
    ryp3 = (hiy_p3 - lowy_p3) / 2.0
    ryn3 = (hiy_n3 - lowy_n3) / 2.0



    x0_n3 = (hix_n_3 + lowx_n_3) / 2.0
    x0_p3 = (hix_p_3 + lowx_p_3) / 2.0
    y0_3 = (hiy_p3 + lowy_p3) / 2.0


    
    #-------------------------------------
    rootfilenp = (f"../outfiles/Pass1/QE_data_GEN{config}_sbs100p_nucleon_np_model2.root")
    rootfilep = (f"../outfiles/Pass1/QE_sim_GEN{config}_sbs100p_nucleon_np_model2.root")
    
    
    C = TChain("Tout")
    B = TChain("Tout")
    
    C.Add(rootfilenp)
    B.Add(rootfilep)

    dx_p, dy_p, W2_p, coin_p,fnucl  = array.array('d', [0]),array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0])
    dx_np, dy_np, W2_np, coin_np, weight = array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0])
    helicity_p, IHWP_p, runnum_p = array.array('i', [0]), array.array('i', [0]), array.array('i', [0])
    helicity_np, IHWP_np, runnum_np= array.array('i', [0]), array.array('i', [0]), array.array('i', [0])
    
    # Disable all branches initially
    C.SetBranchStatus("*", 0)
    B.SetBranchStatus("*", 0)

    # Enable specific branches
    branches = ["dx", "dy", "W2", "helicity", "IHWP", "runnum", "coinCut", "coin_time"]
    b2=["dx", "dy", "W2"]
    for branch in branches:
        C.SetBranchStatus(branch, 1)
    for branch in b2:
        B.SetBranchStatus(branch, 1)

    B.SetBranchStatus("weight", 1)
    B.SetBranchStatus("fnucl",1)
    # Set branch addresses
    C.SetBranchAddress("dx", dx_np)
    B.SetBranchAddress("dx", dx_p)
    C.SetBranchAddress("dy", dy_np)
    B.SetBranchAddress("dy", dy_p)
    C.SetBranchAddress("W2", W2_np)
    B.SetBranchAddress("W2", W2_p)
    C.SetBranchAddress("helicity", helicity_np)
    #B.SetBranchAddress("helicity", helicity_p)
    C.SetBranchAddress("IHWP", IHWP_np)
    #B.SetBranchAddress("IHWP", IHWP_p)
    C.SetBranchAddress("coin_time", coin_np)
    #B.SetBranchAddress("coin_time", coin_pp)
    C.SetBranchAddress("runnum", runnum_np)
    #B.SetBranchAddress("runnum", runnum_p)
    B.SetBranchAddress("weight", weight)
    B.SetBranchAddress("fnucl", fnucl)
    
    # Assuming the variables are already defined or loaded from the ROOT file
    nbins=nBins
    
    xmin, xmax = -4, 2.5
    if config=="2":
        xmin=-5.5
        xmax=2.8
    
    hdx_total_data = TH1F("hdx_total_data", "#Deltax;#Deltax;Entries", nbins, xmin, xmax)
    hdx_total_sim = TH1F("hdx_total_sim", "#Deltax;#Deltax;Entries", 100, -6, 4)
    hdx_p = TH1F("hdx_p", "#Deltax for helicity +1;#Deltax;Entries", 100, -6, 4)
    hdx_m = TH1F("hdx_m", "#Deltax for helicity -1;#Deltax;Entries", 100, -6, 4)
    
    hdx_data_plus = TH1F("hdx_data_plus", "", nbins, xmin, xmax)
    hdx_data_minus = TH1F("hdx_data_minus", "", nbins, xmin, xmax)
    hdx_sim_p = TH1F("hdx_sim_p", "", nbins, xmin, xmax)
    hdx_sim_n = TH1F("hdx_sim_n", "", nbins, xmin, xmax)
    hdx_bg_data = TH1F("hdx_bg_data", "", nbins, xmin, xmax)
    hdx_bg_data_plus = TH1F("hdx_bg_data_plus", "", nbins, xmin, xmax)
    hdx_bg_data_minus = TH1F("hdx_bg_data_minus", "", nbins, xmin, xmax)
    

    nEntries_np = C.GetEntries()
    for i in range(nEntries_np):
        C.GetEntry(i)
        if IHWP_np[0] == 1:
            helicity_np[0] *= -1
        elif IHWP_np[0] == -1:
            helicity_np[0] *= 1
        else:
            continue
            
#____________CUTS_______________________________      
        ycut = dymin < dy_np[0] < dymax
        bgycut=dybgmin<dy_np[0]<dybgmax
        coin_cut = coinmin < coin_np[0] < coinmax
        W2cut=W2min < W2_np[0] < W2max
#________________________________________________     

        if coin_cut and W2cut and runnum_np[0] > 2165 and not bgycut:
            hdx_bg_data.Fill(dx_np[0])
            if helicity_np[0] == 1:
                hdx_bg_data_plus.Fill(dx_np[0])
            if helicity_np[0] == -1:
                hdx_bg_data_minus.Fill(dx_np[0])

        if coin_cut and W2cut and runnum_np[0] > 2165 and ycut:
            hdx_total_data.Fill(dx_np[0])
            if helicity_np[0] == 1:
                hdx_data_plus.Fill(dx_np[0])
            if helicity_np[0] == -1:
                hdx_data_minus.Fill(dx_np[0])
    
    # Simulation loop
    nEntries_p = B.GetEntries()
    
    for i in range(nEntries_p):
        B.GetEntry(i)
        

        #ncut = (dx_p[0] - x0_n3)**2 / rxn3**2 + (dy_p[0] - y0_3)**2 / ryn3**2 <= 1
        #pcut = (dx_p[0] - x0_p3)**2 / rxp3**2 + (dy_p[0] - y0_3)**2 / ryp3**2 <= 1
        ##temp testing-------------------------------
       # ycut = lowydata < dy_p[0] < hiydata
        #xcutn = lowxdatan < dx_p[0] < hixdatan
       # #xcutp = lowxdatap < dx_p[0] < hixdatap
        ######---------------------------------------
                
        #____________CUTS_______________________________      
        ycut = dymin < dy_p[0] < dymax
        W2cut=W2min < W2_p[0] < W2max
        #________________________________________________ 
        
        #temp replace pcut with ycut+xcut
        #print(fnucl)
        #if W2_p[0] < W2max and ycut and fnucl[0]==1:
        #    hdx_sim_p.Fill(dx_p[0], weight[0])
        #if W2_p[0] < W2max and ycut and fnucl[0]==0:
        #    hdx_sim_n.Fill(dx_p[0], weight[0])
        if W2cut and ycut and fnucl[0]==1:
            hdx_sim_p.Fill(dx_p[0], weight[0])
        if W2cut and ycut and fnucl[0]==0:
            hdx_sim_n.Fill(dx_p[0], weight[0])
    
    

    # Fit distributions
    cfg = f"GEN{config}"
    print(cfg)
    dists = DistributionFits(bg_shape_option="pol4" if cfg == "GEN2" else "from data")
    dists.hdx_data = (np.array([hdx_total_data.GetBinCenter(i) for i in range(1, hdx_total_data.GetNbinsX() + 1)]), 
                      np.array([hdx_total_data.GetBinContent(i) for i in range(1, hdx_total_data.GetNbinsX() + 1)]))
    dists.hdx_sim_p = (np.array([hdx_sim_p.GetBinCenter(i) for i in range(1, hdx_sim_p.GetNbinsX() + 1)]), 
                       np.array([hdx_sim_p.GetBinContent(i) for i in range(1, hdx_sim_p.GetNbinsX() + 1)]))
    dists.hdx_sim_n = (np.array([hdx_sim_n.GetBinCenter(i) for i in range(1, hdx_sim_n.GetNbinsX() + 1)]), 
                       np.array([hdx_sim_n.GetBinContent(i) for i in range(1, hdx_sim_n.GetNbinsX() + 1)]))
    dists.hdx_bg_data = (np.array([hdx_bg_data.GetBinCenter(i) for i in range(1, hdx_bg_data.GetNbinsX() + 1)]), 
                         np.array([hdx_bg_data.GetBinContent(i) for i in range(1, hdx_bg_data.GetNbinsX() + 1)]))

    hdx_bg_fit, hdx_total_fit, hdx_sim_p, hdx_sim_n = dists.He3_fit_dists()
    # Plot results
    hdx_data_plot = hdx_total_data.Clone("hdx_data_plot")
    #hdx_sim_p_plot = hdx_sim_p.Clone("hdx_sim_p_plot")
    #hdx_sim_n_plot = hdx_sim_n.Clone("hdx_sim_n_plot")
    hdx_sim_p_plot = TH1F("hdx_sim_p_plot", "", nbins, xmin, xmax)
    hdx_sim_n_plot = TH1F("hdx_sim_n_plot", "", nbins, xmin, xmax)
    hdx_bg_plot = TH1F("hdx_bg_plot", "", nbins, xmin, xmax)
    hdx_total_fit_plot = TH1F("hdx_total_fit_plot", "", nbins, xmin, xmax)

    for i in range(nbins):
        hdx_bg_plot.SetBinContent(i + 1, hdx_bg_fit[i])
        hdx_total_fit_plot.SetBinContent(i + 1, hdx_total_fit[i])
        hdx_sim_p_plot.SetBinContent(i + 1, hdx_sim_p[i])
        hdx_sim_n_plot.SetBinContent(i + 1, hdx_sim_n[i])

    gStyle.SetOptFit(0)
    
    hdx_data_plot.SetTitle(f"Data/Simulation Comparison {cfg};#Deltax (m);Entries")
    hdx_data_plot.SetMarkerStyle(r.kFullCircle)
    hdx_total_fit_plot.SetFillColorAlpha(30, 0.5)
    hdx_sim_p_plot.SetFillColorAlpha(r.kRed, 0.3)
    hdx_sim_n_plot.SetFillColorAlpha(r.kBlue, 0.3)
    hdx_bg_plot.SetFillColorAlpha(r.kMagenta, 0.3)
    
    hdx_total_fit_plot.SetLineStyle(7)
    hdx_sim_p_plot.SetLineStyle(7)
    hdx_sim_n_plot.SetLineStyle(7)
    hdx_bg_plot.SetLineStyle(7)
    
    hdx_total_fit_plot.SetLineColor(30)
    hdx_sim_p_plot.SetLineColor(r.kRed)
    hdx_sim_n_plot.SetLineColor(r.kBlue)
    hdx_bg_plot.SetLineColor(r.kMagenta)
    
    c = TCanvas("c", "", 800, 600)
    hdx_data_plot.Draw()
    hdx_total_fit_plot.Draw("same hist")
    hdx_sim_p_plot.Draw("same hist")
    hdx_sim_n_plot.Draw("same hist")
    hdx_bg_plot.Draw("same hist")
    
    legend = TLegend(0.65, 0.72, 0.89, 0.89)
    legend.AddEntry("hdx_data_plot", "Data", "p")
    legend.AddEntry("hdx_total_fit_plot", "MC Fit", "lf")
    legend.AddEntry("hdx_sim_p_plot", "MC p", "lf")
    legend.AddEntry("hdx_sim_n_plot", "MC n", "lf")
    legend.AddEntry("hdx_bg_plot", "Background", "lf")
    legend.SetLineColor(0)
    legend.Draw("same")
    
    output = f"Data_sim_total_{cfg}.pdf"
    #c.SaveAs(f"../plots/{output}")
    return UTILITIES.Function_HIST2NP(hdx_data_plot), UTILITIES.Function_HIST2NP(hdx_bg_plot),UTILITIES.Function_HIST2NP(hdx_total_fit_plot),UTILITIES.Function_HIST2NP(hdx_sim_p_plot),UTILITIES.Function_HIST2NP(hdx_sim_n_plot)


def Function_FITDXSENS2D(config,cut,value):
    import ROOT as r
    import math
    import array
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    #______________Add include directory_______________
    current_dir = os.getcwd()
    include_dir = os.path.join(current_dir, '../include')
    sys.path.insert(0, include_dir)
    #__________________________________________________

    import CONFIG
    import DBPARSE
    import UTILITIES
    from SIMFITS2D import DistributionFits2D
    from ROOT import gStyle, TChain, TH1F, TCanvas, TLegend
    W2min=CONFIG.Function_JSON("W2min",f"../config/cuts{config}.cfg")
    W2max=CONFIG.Function_JSON("W2max",f"../config/cuts{config}.cfg")
    dxmin=CONFIG.Function_JSON("dxmin",f"../config/cuts{config}.cfg")
    dxmax=CONFIG.Function_JSON("dxmax",f"../config/cuts{config}.cfg")
    dymin=CONFIG.Function_JSON("dymin",f"../config/cuts{config}.cfg")
    dymax=CONFIG.Function_JSON("dymax",f"../config/cuts{config}.cfg")
    dybgmin=CONFIG.Function_JSON("dybgmin",f"../config/cuts{config}.cfg")
    dybgmax=CONFIG.Function_JSON("dybgmax",f"../config/cuts{config}.cfg")
    coinmin=CONFIG.Function_JSON("coinmin",f"../config/cuts{config}.cfg")
    coinmax=CONFIG.Function_JSON("coinmax",f"../config/cuts{config}.cfg")
    nBins=CONFIG.Function_JSON("nBins",f"../config/cuts{config}.cfg")
    r.gErrorIgnoreLevel = r.kError  # Suppress Info and Warning messages
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(1)

    #_____bring in config values______#
    coinVector = CONFIG.Function_JSON("GEN" + config, "../config/coin.cfg")
    he3spot = CONFIG.Function_JSON("GEN" + config + "He3", "../config/spotsize.cfg")
    he3spotsim = CONFIG.Function_JSON("GEN" + config + "He3sim", "../config/spotsize.cfg")
    
    # Set coincidence cut and spot cut values
    coinSigma = 2.5
    coin_low = coinVector[0] - coinSigma * coinVector[1]
    coin_high = coinVector[0] + coinSigma * coinVector[1]

    # Spot imports for dx, dy high and low
    hiydata = he3spot[1]
    lowydata = he3spot[0]
    
    hixdatan = he3spotsim[7]
    lowxdatan = he3spotsim[6]
    hixdatap = he3spotsim[5]
    lowxdatap = he3spotsim[4]
    
    hix_n_3 = he3spotsim[7]
    lowx_n_3 = he3spotsim[6]
    hix_p_3 = he3spotsim[5]
    lowx_p_3 = he3spotsim[4]
    hiy_p3 = he3spotsim[1]
    lowy_p3 = he3spotsim[0]
    hiy_n3 = he3spotsim[3]
    lowy_n3 = he3spotsim[2]
    
    rxn3 = (hix_n_3 - lowx_n_3) / 2.0
    rxp3 = (hix_p_3 - lowx_p_3) / 2.0
    ryp3 = (hiy_p3 - lowy_p3) / 2.0
    ryn3 = (hiy_n3 - lowy_n3) / 2.0

    x0_n3 = (hix_n_3 + lowx_n_3) / 2.0
    x0_p3 = (hix_p_3 + lowx_p_3) / 2.0
    y0_3 = (hiy_p3 + lowy_p3) / 2.0
    
    #-------------------------------------
    rootfilenp = f"../outfiles/Pass1/QE_data_GEN{config}_sbs100p_nucleon_np_model2.root"
    rootfilep = f"../outfiles/Pass1/QE_sim_GEN{config}_sbs100p_nucleon_np_model2.root"
    
    C = TChain("Tout")
    B = TChain("Tout")
    
    C.Add(rootfilenp)
    B.Add(rootfilep)

    dx_p, dy_p, W2_p, coin_p, fnucl = array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0])
    dx_np, dy_np, W2_np, coin_np, weight = array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0]), array.array('d', [0])
    helicity_p, IHWP_p, runnum_p = array.array('i', [0]), array.array('i', [0]), array.array('i', [0])
    helicity_np, IHWP_np, runnum_np = array.array('i', [0]), array.array('i', [0]), array.array('i', [0])
    
    # Disable all branches initially
    C.SetBranchStatus("*", 0)
    B.SetBranchStatus("*", 0)

    # Enable specific branches
    branches = ["dx", "dy", "W2", "helicity", "IHWP", "runnum", "coinCut", "coin_time"]
    b2 = ["dx", "dy", "W2"]
    for branch in branches:
        C.SetBranchStatus(branch, 1)
    for branch in b2:
        B.SetBranchStatus(branch, 1)

    B.SetBranchStatus("weight", 1)
    B.SetBranchStatus("fnucl", 1)

    # Set branch addresses
    C.SetBranchAddress("dx", dx_np)
    B.SetBranchAddress("dx", dx_p)
    C.SetBranchAddress("dy", dy_np)
    B.SetBranchAddress("dy", dy_p)
    C.SetBranchAddress("W2", W2_np)
    B.SetBranchAddress("W2", W2_p)
    C.SetBranchAddress("helicity", helicity_np)
    C.SetBranchAddress("IHWP", IHWP_np)
    C.SetBranchAddress("coin_time", coin_np)
    C.SetBranchAddress("runnum", runnum_np)
    B.SetBranchAddress("weight", weight)
    B.SetBranchAddress("fnucl", fnucl)
    
    # Assuming the variables are already defined or loaded from the ROOT file
    nbins = 100   
    xmin, xmax = -4,2.5
    ymin, ymax = -1.5,1.5  # Add appropriate ymin and ymax

    if config == "2":
        xmin = -5.5
        xmax = 2.8
    
    hdx_total_data = r.TH2F("hdx_total_data", "#Deltax;#Deltax;Entries", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_total_sim = r.TH2F("hdx_total_sim", "#Deltax;#Deltax;Entries", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_sim_p = r.TH2F("hdx_sim_p", "#Deltax for helicity +1;#Deltax;Entries", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_sim_n = r.TH2F("hdx_sim_n", "#Deltax for helicity -1;#Deltax;Entries", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_data_plus = r.TH1F("hdx_data_plus", "", nbins, xmin, xmax)
    hdx_data_minus = r.TH1F("hdx_data_minus", "", nbins, xmin, xmax)
    hdx_bg_data = r.TH2F("hdx_bg_data", "", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_bg_data_plus = r.TH2F("hdx_bg_data_plus", "", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_bg_data_minus = r.TH2F("hdx_bg_data_minus", "", nbins, xmin, xmax, nbins, ymin, ymax)
    
    nEntries_np = C.GetEntries()
    for i in range(nEntries_np):
        C.GetEntry(i)
        if IHWP_np[0] == 1:
            helicity_np[0] *= -1
        elif IHWP_np[0] == -1:
            helicity_np[0] *= 1
        else:
            continue
            
        #____________CUTS_______________________________      
        ycut = dymin < dy_np[0] < dymax
        bgycut = dybgmin < dy_np[0] < dybgmax
        coin_cut = coinmin < coin_np[0] < coinmax
        W2cut = W2min < W2_np[0] < W2max
        #________________________________________________     

        if coin_cut and W2cut and runnum_np[0] > 2165 and not bgycut:
            hdx_bg_data.Fill(dx_np[0], dy_np[0])
            if helicity_np[0] == 1:
                hdx_bg_data_plus.Fill(dx_np[0], dy_np[0])
            if helicity_np[0] == -1:
                hdx_bg_data_minus.Fill(dx_np[0], dy_np[0])

        if coin_cut and W2cut and runnum_np[0] > 2165:
            hdx_total_data.Fill(dx_np[0], dy_np[0])
            if helicity_np[0] == 1:
                hdx_data_plus.Fill(dx_np[0], dy_np[0])
            if helicity_np[0] == -1:
                hdx_data_minus.Fill(dx_np[0], dy_np[0])
    
    # Simulation loop
    nEntries_p = B.GetEntries()
    
    for i in range(nEntries_p):
        B.GetEntry(i)
        
        #____________CUTS_______________________________      
        ycut = dymin < dy_p[0] < dymax
        W2cut = W2min < W2_p[0] < W2max
        #________________________________________________ 
        
        if W2cut and fnucl[0] == 1:
            hdx_sim_p.Fill(dx_p[0], dy_p[0], weight[0])
        if W2cut and fnucl[0] == 0:
            hdx_sim_n.Fill(dx_p[0], dy_p[0], weight[0])
    
    # Fit distributions
    cfg = f"GEN{config}"
    print(cfg)
    
    dists = DistributionFits2D(bg_shape_option="gaus" if cfg == "GEN2" or cfg=="GEN3" or cfg =="GEN4" else "from data")
    
    bin_centers_x, bin_centers_y, bin_contents_data = UTILITIES.Function_2DHIST2NP(hdx_total_data)
    bin_centers_x, bin_centers_y, bin_contents_sim_p = UTILITIES.Function_2DHIST2NP(hdx_sim_p)
    bin_centers_x, bin_centers_y, bin_contents_sim_n = UTILITIES.Function_2DHIST2NP(hdx_sim_n)
    bin_centers_x, bin_centers_y, bin_contents_bg_data = UTILITIES.Function_2DHIST2NP(hdx_bg_data)

    dists.hdx_data = bin_contents_data
    dists.hdx_sim_p = bin_contents_sim_p
    dists.hdx_sim_n = bin_contents_sim_n
    dists.hdx_bg_data = bin_contents_bg_data

    hdx_bg_fit, hdx_total_fit, hdx_sim_p, hdx_sim_n = dists.He3_fit_dists()
    
    # Plot results
    hdx_data_plot = hdx_total_data.Clone("hdx_data_plot")
    hdx_sim_p_plot = r.TH2F("hdx_sim_p_plot", "", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_sim_n_plot = r.TH2F("hdx_sim_n_plot", "", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_bg_plot = r.TH2F("hdx_bg_plot", "", nbins, xmin, xmax, nbins, ymin, ymax)
    hdx_total_fit_plot = r.TH2F("hdx_total_fit_plot", "", nbins, xmin, xmax, nbins, ymin, ymax)

    for i in range(nbins):
        for j in range(nbins):
            hdx_bg_plot.SetBinContent(i + 1, j + 1, hdx_bg_fit[i, j])
            hdx_total_fit_plot.SetBinContent(i + 1, j + 1, hdx_total_fit[i, j])
            hdx_sim_p_plot.SetBinContent(i + 1, j + 1, hdx_sim_p[i, j])
            hdx_sim_n_plot.SetBinContent(i + 1, j + 1, hdx_sim_n[i, j])

    gStyle.SetOptFit(0)
    
    hdx_data_plot.SetTitle(f"Data/Simulation Comparison {cfg};#Deltax (m);Entries")
    hdx_data_plot.SetMarkerStyle(r.kFullCircle)
    hdx_total_fit_plot.SetFillColorAlpha(30, 0.5)
    hdx_sim_p_plot.SetFillColorAlpha(r.kRed, 0.3)
    hdx_sim_n_plot.SetFillColorAlpha(r.kBlue, 0.3)
    hdx_bg_plot.SetFillColorAlpha(r.kMagenta, 0.3)
    
    hdx_total_fit_plot.SetLineStyle(7)
    hdx_sim_p_plot.SetLineStyle(7)
    hdx_sim_n_plot.SetLineStyle(7)
    hdx_bg_plot.SetLineStyle(7)
    
    hdx_total_fit_plot.SetLineColor(30)
    hdx_sim_p_plot.SetLineColor(r.kRed)
    hdx_sim_n_plot.SetLineColor(r.kBlue)
    hdx_bg_plot.SetLineColor(r.kMagenta)
    
    c = TCanvas("c", "", 800, 600)
    hdx_data_plot.Draw("COLZ")
    hdx_total_fit_plot.Draw("same COLZ")
    hdx_sim_p_plot.Draw("same COLZ")
    hdx_sim_n_plot.Draw("same COLZ")
    hdx_bg_plot.Draw("same COLZ")
    
    legend = TLegend(0.65, 0.72, 0.89, 0.89)
    legend.AddEntry("hdx_data_plot", "Data", "p")
    legend.AddEntry("hdx_total_fit_plot", "MC Fit", "lf")
    legend.AddEntry("hdx_sim_p_plot", "MC p", "lf")
    legend.AddEntry("hdx_sim_n_plot", "MC n", "lf")
    legend.AddEntry("hdx_bg_plot", "Background", "lf")
    legend.SetLineColor(0)
    legend.Draw("same")
    
    output = f"Data_sim_total_{cfg}.pdf"
    # c.SaveAs(f"../plots/{output}")
    
    return UTILITIES.Function_2DHIST2NP(hdx_data_plot), UTILITIES.Function_2DHIST2NP(hdx_bg_plot), UTILITIES.Function_2DHIST2NP(hdx_total_fit_plot), UTILITIES.Function_2DHIST2NP(hdx_sim_p_plot), UTILITIES.Function_2DHIST2NP(hdx_sim_n_plot)
                     
                         
                         
def Function_APHYS(config,pas,rawResults,accResult,bgResult,protonResult):
    runs,A,AE,Y,he3Pol,beamPol,cut,cutVal=rawResults
    import ROOT as r
    import math
    import array
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    #______________Add include directory_______________
    current_dir = os.getcwd()
    include_dir = os.path.join(current_dir, '../include')
    sys.path.insert(0, include_dir)
    #__________________________________________________
    import ERROR
    import CONFIG
    import DBPARSE
    import UTILITIES
    from SIMFITS import DistributionFits
    from ROOT import gStyle, TChain, TH1F, TCanvas, TLegend


    from joblib import Parallel, delayed
    Aacc,Eacc,facc,Efacc=accResult
    fP,fPE=protonResult
    
    

    fproton=fP
    Efproton=fPE
    Aproton=np.sum(np.load(f'CorrectionArrays/Pass{pas}/Aproton{config}.npy'))
    Eproton=.005*Aproton
    
    if config=="2":
        fpion=0
    else:
        fpion=0.01
    Efpion=.0005*fpion
    Apion=0
    Epion=.0005*Apion

    fFSI=0
    EfFSI=.0005*fFSI
    AFSI=0
    EFSI=.0005*AFSI

    fnitro=.0157
    Efnitro=.0005*fnitro

    Pneutron=.95
    Eneutron=.0005*Pneutron

    Abg,Ebg,fbackground,Efbackground=bgResult
    fbg=fbackground-fpion-facc-fFSI-fnitro
    Efbg=np.sqrt(Efbackground**2+Efpion**2+Efacc**2+Efnitro**2)
    
    
    #----------------------------------------------------------
    farray=[facc,fproton,fbg,fpion,fFSI]
    
    Efarray1=[Efacc,Efproton,Efbg,Efpion,EfFSI]
    Efarray2=[Efacc,Efproton,Efbg,Efpion,EfFSI,Efnitro]

    Aarray=[Aacc,Aproton,Abg,Apion,AFSI]
    AEarray=[Eacc,Eproton,Ebg,Epion,EFSI]

    fAE=ERROR.Function_f_A_ERROR(farray,Efarray1,Aarray,AEarray)
  
    fE=ERROR.Function_f_ERROR(Efarray2)
    

    #----------------------------------------------------------
    fA= facc*Aacc + fproton*Aproton + fbg*Abg + fpion*Apion + fFSI*AFSI

    f=facc + fproton + fbg + fpion + fFSI +fnitro
    fN=1-f
    
    print(f'fA Error:{fA}+-{fAE}')
    print(f'f Error:{f}+-{fE}')
    newA=np.array(A)
    newYield=np.array(Y)
    newAE=np.array(AE)
    weightedSum=0
    sumWeights=0
    sumSys=0
    sumErrors=0

    polsum=np.empty(0)

    precorrection=0
    precorrectionW=0
    for i in range(0,len(newA)):
        #temp
        #Efacc=AE[i]
        #Eproton=AE[i]
        #Epion=AE[i]
        #EFSI=AE[i]
        #Ebg=AE[i]
        #----------------
        PbE=.03*beamPol[i]/100
        PtE=.03*he3Pol[i]/100
        rawE=newAE[i]/(he3Pol[i]*beamPol[i]*Pneutron*(1-f)/10000)
        precorrection+=newA[i]/(AE[i]**2)
        precorrectionW+=1/(AE[i]**2)
    
        calculate=ERROR.Function_WEIGHTEDAVERAGEAPHYS(newA[i],fA,f,fnitro,beamPol[i]/100,Pneutron,he3Pol[i]/100,AE[i],fAE,fE,Efnitro,PbE,Eneutron,PtE)
        sysError=ERROR.Function_SYSERROR(fA,fN,beamPol[i]/100,he3Pol[i]/100,Pneutron,fAE,fE,PbE,PtE,Eneutron)
        statError=ERROR.Function_STATERROR(newA[i],fN,beamPol[i]/100,he3Pol[i]/100,Pneutron,AE[i],fE,PbE,PtE,Eneutron)
        #totError=np.sqrt(sysError**2+statError**2)
        w=calculate[0]
        w_sig= calculate[1] 
        weightedSum+=(w/(statError**2))
        sumWeights+=(1/(statError**2))
        sumSys+=(1/sysError**2)
        
    totalSys=1/math.sqrt(sumSys)
    totalStat=1/math.sqrt(sumWeights)
    
    totError=totalSys+totalStat

    rawAsymmetry=precorrection/precorrectionW
    rawAsymmetryE=math.sqrt(1/precorrectionW)
    weighted_A=weightedSum/sumWeights
    weighted_A_E=totError
    return weighted_A,weighted_A_E,fbg, rawAsymmetry,rawAsymmetryE,totalSys,totalStat