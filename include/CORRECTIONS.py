def Function_ANALYZEROOTFILE(config,hbg,hproton,htotal):
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

   
    rootfilenp = (f"../outfiles/Pass1/QE_data_GEN{config}_sbs100p_nucleon_np_model2.root")
    rootfilep = (f"../outfiles/Pass1/QE_sim_GEN{config}_sbs100p_nucleon_np_model2.root")
    if config == "4":
        config = "4b"
    
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
    
    nbins = 200
    xmin, xmax = 0, 200
    
    hcoin = TH1F("hcoin","Coincidence Time ;Time (ns);Entries", nbins, xmin, xmax)
    hcoin_minus = TH1F("hcoin_minus","Coincidence Time -;Time (ns);Entries", nbins, xmin, xmax)
    hcoin_plus = TH1F("hcoin_pluys","Coincidence Time + ;Time (ns);Entries", nbins, xmin, xmax)

    hbgtot = TH1F("hbgtot","Background ;dx;Entries", nbins, xmin, xmax)
    hbg_plus = TH1F("hbg_plus","Background -;Time (ns);Entries", nbins, xmin, xmax)
    hbg_minus = TH1F("hbg_minus","Background + ;dx;Entries", nbins, xmin, xmax)

    nEntries_np = C.GetEntries()
  
    


    
    for i in range(nEntries_np):
        C.GetEntry(i)
        #____________CUTS_______________________________      
        ycut = dymin < dy_np[0] < dymax
        bgycut=dybgmin<dy_np[0]<dybgmax
        coin_cut = coinmin < coin_np[0] < coinmax
        W2cut=W2min < W2_np[0] < W2max
        xcutn = dxmin < dx_np[0] < dxmax
        #________________________________________________ 

        if IHWP_np[0] == 1:
            helicity_np[0] *= -1
        elif IHWP_np[0] == -1:
            helicity_np[0] *= 1
        else:
            continue

        
        if W2cut and ycut and xcutn:
            hcoin.Fill(coin_np[0])
            
            if helicity_np[0] == 1:
                hcoin_plus.Fill(coin_np[0])
            if helicity_np[0] == -1:
                hcoin_minus.Fill(coin_np[0])
        if coin_cut and not W2cut and xcutn and not bgycut and runnum_np[0] > 2165:
            hbgtot.Fill(dx_np[0])
            if helicity_np[0] == 1:
                hbg_plus.Fill(dx_np[0])
            if helicity_np[0] == -1:
                hbg_minus.Fill(dx_np[0])
                
    Aacc,AEacc,facc=Function_ACCIDENTAL(config,hcoin,hcoin_plus,hcoin_minus,coinmin,coinmax)
    Abg,AEbg=Function_INELASTIC(config,hbgtot,hbg_plus,hbg_minus)
    
    lower_bound = dxmin
    upper_bound = dxmax

    bin_centers,bin_contents=hbg
    numBG=np.sum(bin_contents[(bin_centers >= lower_bound) & (bin_centers <= upper_bound)])

    bin_centers,bin_contents=hproton
    numProton=np.sum(bin_contents[(bin_centers >= lower_bound) & (bin_centers <= upper_bound)])

    bin_centers,bin_contents=htotal
    numTotal=np.sum(bin_contents[(bin_centers >= lower_bound) & (bin_centers <= upper_bound)])
    
    fproton=np.round(numProton/numTotal,4)
    fbg=np.round(numBG/numTotal,4)
    
    return [Aacc,AEacc,facc],[Abg,AEbg,fbg],fproton




def Function_ACCIDENTAL(config,hcoin,hcoinp,hcoinm,coinmin,coinmax):
    import UTILITIES
    import numpy as np
    import math
    p=0
    m=0
    bgextra=50
    cointot=UTILITIES.Function_HIST2NP(hcoin)
    coinplus=UTILITIES.Function_HIST2NP(hcoinp)
    coinminus=UTILITIES.Function_HIST2NP(hcoinm)
    
    bin_centers,bin_contents=coinplus

    count_ranges=[(0, coinmin), (coinmax, 200)]

    total_counts = 0

    # Sum the bin contents within the specified ranges
    for count_range_min, count_range_max in count_ranges:
        count_mask = (bin_centers >= count_range_min) & (bin_centers <= count_range_max)
        count_bin_contents = bin_contents[count_mask]
        total_counts += np.sum(count_bin_contents)

    p=total_counts
    bin_centers,bin_contents=coinminus

    count_ranges=[(0, coinmin), (coinmax, 200)]

    total_counts = 0

    # Sum the bin contents within the specified ranges
    for count_range_min, count_range_max in count_ranges:
        count_mask = (bin_centers >= count_range_min) & (bin_centers <= count_range_max)
        count_bin_contents = bin_contents[count_mask]
        total_counts += np.sum(count_bin_contents)

    m=total_counts
    
    bin_centers,bin_contents=cointot
    
    background_events=np.sum(bin_contents[(bin_centers >= coinmin+bgextra) & (bin_centers <= coinmax+bgextra)])
    total_events=np.sum(bin_contents[(bin_centers >= coinmin) & (bin_centers <= coinmax)])
    ratio = np.round(background_events / total_events,4)

    
    A=(p-m)/(p+m)
    AE=2*math.sqrt(p * m) / (p + m)**(3/2)
    return A,AE,ratio

def Function_INELASTIC(config,hbg,hbgp,hbgm):
    import numpy as np
    import UTILITIES
    import math
    bgtot=UTILITIES.Function_HIST2NP(hbg)
    bgplus=UTILITIES.Function_HIST2NP(hbgp)
    bgmin=UTILITIES.Function_HIST2NP(hbgm)
    p=0
    m=0
    
    bin_centers,bin_contents=bgplus
    p=np.sum(bin_contents)
    
    bin_centers,bin_contents=bgmin
    m=np.sum(bin_contents)
    
    A=(p-m)/(p+m)
    AE=2*math.sqrt(p * m)/(p + m)**(3/2)
    return A,AE
