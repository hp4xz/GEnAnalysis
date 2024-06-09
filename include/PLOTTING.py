def Function_EXTRACTHIST(hist):
    from matplotlib.colors import LogNorm

    from IPython.display import Image, display
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import ROOT as r
    import array
    import os
    """Extract data from ROOT 2D histogram"""
    x_bins = hist.GetNbinsX()
    y_bins = hist.GetNbinsY()
    
    x_edges = [hist.GetXaxis().GetBinLowEdge(i) for i in range(1, x_bins + 2)]
    y_edges = [hist.GetYaxis().GetBinLowEdge(i) for i in range(1, y_bins + 2)]
    
    data = np.zeros((x_bins, y_bins))
    
    for i in range(1, x_bins + 1):
        for j in range(1, y_bins + 1):
            data[i-1, j-1] = hist.GetBinContent(i, j)
            
    return np.array(x_edges), np.array(y_edges), data

def Function_PLOT2DROOTHIST(hist, title, filename):
    from matplotlib.colors import LogNorm
    from matplotlib.colors import PowerNorm

    from IPython.display import Image, display
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import ROOT as r
    import array
    import os
    """Plot a ROOT 2D histogram using Matplotlib"""
    colors = ['dodgerblue', '#f2be44','red']
    colors = ['black', 'magenta', '#fc9662', 'orange']
    colors = ['white','black','purple', 'red','#fc9662','orange','yellow']
    colors = ['white', 'red','orange','yellow']
    colors = ['white', 'dodgerblue','mediumaquamarine','gold','yellow']
    #colors = ['#000004', '#b63679', '#fc9662', '#fcffa4']
    #colors = ['white', 'red','dodgerblue']

#norm=PowerNorm(gamma=2.4)


    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    x_edges, y_edges, data = Function_EXTRACTHIST(hist)
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_edges, y_edges, data.T, shading='auto', cmap=cmap)
    plt.colorbar()
    fs=15
    plt.title(title,fontsize=fs)
    plt.xlabel(r'$\Delta y$',fontsize=fs,fontweight='bold')
    plt.ylabel(r'$\Delta x$',fontsize=fs,fontweight='bold')
    plt.savefig(filename)
    plt.show()
    plt.close()