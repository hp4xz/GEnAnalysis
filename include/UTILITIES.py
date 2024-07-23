def Function_HIST2NP(hist):
    import numpy as np
    nbins = hist.GetNbinsX()
    bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, nbins + 1)])
    bin_contents = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)])
    return bin_centers, bin_contents
def Function_2DHIST2NP(hist):
    import numpy as np
    nbins_x = hist.GetNbinsX()
    nbins_y = hist.GetNbinsY()
    
    bin_centers_x = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, nbins_x + 1)])
    bin_centers_y = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, nbins_y + 1)])
    
    bin_contents = np.zeros((nbins_x, nbins_y))
    for i in range(1, nbins_x + 1):
        for j in range(1, nbins_y + 1):
            bin_contents[i-1, j-1] = hist.GetBinContent(i, j)
    
    return bin_centers_x, bin_centers_y, bin_contents
