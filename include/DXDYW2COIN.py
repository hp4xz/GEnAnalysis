def Function_MAKEHIST(config):
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
    r.gErrorIgnoreLevel = r.kError  # Suppress Info and Warning messages
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(1)

    
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
    

    nbins = 150
    xmin=-5.5
    xmax=2.8

    
    hdx = TH1F("hdx", "#Deltax;#Deltax;Entries", nbins, xmin, xmax)
    hw2 = TH1F("hw2", "w2;w2;Entries", nbins, 0, 3)
    hdy = TH1F("hdy", "#Deltay;#Deltay;Entries", nbins, xmin, xmax)
  
    
    

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
        xcut=dxmin < dx_np[0] < dxmax
#________________________________________________     

        if coin_cut and W2cut and runnum_np[0] > 2165 and ycut:
            hdx.Fill(dx_np[0])

        if coin_cut and W2cut and runnum_np[0] > 2165 and xcut:
            hdy.Fill(dy_np[0])
            
        if coin_cut and xcut and runnum_np[0] > 2165 and ycut:
            hw2.Fill(W2_np[0])


 
    return UTILITIES.Function_HIST2NP(hdx), UTILITIES.Function_HIST2NP(hdy),UTILITIES.Function_HIST2NP(hw2)