

class DistributionFits:
    import numpy as np
    from scipy.optimize import curve_fit
    def __init__(self, bg_shape_option='pol4'):
        import numpy as np
        from scipy.optimize import curve_fit
        self.bg_shape_option = bg_shape_option
        self.bg_shape_set = True
        self.hdx_data = None
        self.hdx_sim_p = None
        self.hdx_sim_n = None
        self.hdx_bg_data = None

    def fitbg_pol4(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit
        bg = np.polyval(par[::-1], x)
        return bg

    def fitbg_pol3(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit        
        bg = np.polyval(par[::-1], x)
        return bg

    def fitbg_pol2(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit 
        bg = np.polyval(par[::-1], x)
        return bg

    def fitbg_gaus(self, x, a, mu, sigma):
        import numpy as np
        from scipy.optimize import curve_fit 
        bg = a * np.exp(-0.5 * ((x - mu) / sigma)**2)
        return bg

    def fitsim(self, x, Norm_overall, R_pn, Bg_norm, *par):
        import numpy as np
        from scipy.optimize import curve_fit
        if self.bg_shape_option == "pol4":
            bg = self.fitbg_pol4(x, *par)
        elif self.bg_shape_option == "pol3":
            bg = self.fitbg_pol3(x, *par)
        elif self.bg_shape_option == "pol2":
            bg = self.fitbg_pol2(x, *par)
        elif self.bg_shape_option == "gaus":
            bg = self.fitbg_gaus(x, *par[:3])
        elif self.bg_shape_option == "from data":
            bg = np.interp(x, self.hdx_bg_data[0], self.hdx_bg_data[1])
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        simu = Norm_overall * (np.interp(x, self.hdx_sim_p[0], self.hdx_sim_p[1]) +
                               R_pn * np.interp(x, self.hdx_sim_n[0], self.hdx_sim_n[1]) +
                               Bg_norm * bg)
        return simu



    def He3_fit_dists(self):
        import numpy as np
        from scipy.optimize import curve_fit
        if not self.bg_shape_set:
            raise ValueError("bg shape has not been set!")

        if self.hdx_data is None or self.hdx_sim_p is None or self.hdx_sim_n is None or self.hdx_bg_data is None:
            raise ValueError("Histograms have not been set!")

        self.hdx_data = list(self.hdx_data)  # Convert to list for item assignment
        self.hdx_sim_p = list(self.hdx_sim_p)  # Convert to list for item assignment
        self.hdx_sim_n = list(self.hdx_sim_n)  # Convert to list for item assignment
        self.hdx_bg_data = list(self.hdx_bg_data)  # Convert to list for item assignment



        scale = np.sum(self.hdx_data[1])
        self.hdx_data[1] /= scale
        self.hdx_sim_p[1] /= np.sum(self.hdx_sim_p[1])
        self.hdx_sim_n[1] /= np.sum(self.hdx_sim_n[1])
        self.hdx_bg_data[1] /= np.sum(self.hdx_bg_data[1])

        xmin, xmax = self.hdx_data[0][0], self.hdx_data[0][-1]

        if self.bg_shape_option == "pol4":
            npar = 5
        elif self.bg_shape_option == "pol3":
            npar = 4
        elif self.bg_shape_option == "pol2":
            npar = 3
        elif self.bg_shape_option == "gaus":
            npar = 3
        elif self.bg_shape_option == "from data":
            npar = 0
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        p0 = [1.0] * (3 + npar)
        bounds = ([0.1, 0.1, 0] + [-np.inf] * npar, [100, 100, 100] + [np.inf] * npar)

        x_data = self.hdx_data[0]
        y_data = self.hdx_data[1]

        popt, _ = curve_fit(self.fitsim, x_data, y_data, p0=p0, bounds=bounds)



        self.hdx_sim_p[1] *= popt[0]
        self.hdx_sim_n[1] *= popt[0] * popt[1]

        if self.bg_shape_option == "from data":
            hdx_bg_fit = np.interp(x_data, self.hdx_bg_data[0], self.hdx_bg_data[1])
        elif self.bg_shape_option == "gaus":
            hdx_bg_fit = self.fitbg_gaus(x_data, popt[3], popt[4], popt[5])
        else:
            bg_fit = self.fitbg_pol4 if self.bg_shape_option == "pol4" else self.fitbg_pol3 if self.bg_shape_option == "pol3" else self.fitbg_pol2
            hdx_bg_fit = bg_fit(x_data, *popt[3:])

        # Apply normalization factors to bg_fit
        hdx_bg_fit *= (popt[0] * popt[2])


        hdx_total_fit = self.hdx_sim_p[1] + self.hdx_sim_n[1] + hdx_bg_fit

        self.hdx_data[1] *= scale
        self.hdx_sim_p[1] *= scale
        self.hdx_sim_n[1] *= scale
        hdx_bg_fit *= scale
        hdx_total_fit *= scale



        return hdx_bg_fit, hdx_total_fit, self.hdx_sim_p[1] ,self.hdx_sim_n[1]
