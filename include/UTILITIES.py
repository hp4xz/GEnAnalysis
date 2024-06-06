def Function_HIST2NP(hist):
    import numpy as np
    nbins = hist.GetNbinsX()
    bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, nbins + 1)])
    bin_contents = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)])
    return bin_centers, bin_contents