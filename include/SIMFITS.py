class DistributionFits:
    import numpy as np
    from scipy.optimize import curve_fit

    def __init__(self, bg_shape_option='pol4'):
        self.bg_shape_option = bg_shape_option
        self.bg_shape_set = True
        self.hdx_data = None
        self.hdx_sim_p = None
        self.hdx_sim_n = None
        self.hdx_bg_data = None
        self.hdx_acc_data = None  # For 'from data+acc'

    def fitbg_pol4(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit
        return np.polyval(par[::-1], x)

    def fitbg_pol3(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit
        return np.polyval(par[::-1], x)

    def fitbg_pol2(self, x, *par):
        import numpy as np
        from scipy.optimize import curve_fit
        return np.square(np.polyval(par[::-1], x))
    def fitbg_gaus(self, x, a, mu, sigma):
        import numpy as np
        from scipy.optimize import curve_fit
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def fitsim(self, x, Norm_overall, R_pn, Bg_norm, *args):
        import numpy as np
        from scipy.optimize import curve_fit
        Acc_norm = 0.0
        if self.bg_shape_option == "from data+acc":
            Acc_norm = args[0]
            par = args[1:]
        else:
            par = args

        if self.bg_shape_option == "pol4":
            bg = self.fitbg_pol4(x, *par)
        elif self.bg_shape_option == "pol3":
            bg = self.fitbg_pol3(x, *par)
        elif self.bg_shape_option == "pol2":
            bg = self.fitbg_pol2(x, *par)
        elif self.bg_shape_option == "gaus":
            bg = self.fitbg_gaus(x, *par[:3])
        elif self.bg_shape_option in ["from data", "from data+acc"]:
            bg = np.interp(x, self.hdx_bg_data[0], self.hdx_bg_data[1])
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        acc = 0
        if self.bg_shape_option == "from data+acc":
            acc = np.interp(x, self.hdx_acc_data[0], self.hdx_acc_data[1])

        simu = Norm_overall * (
            np.interp(x, self.hdx_sim_p[0], self.hdx_sim_p[1]) +
            R_pn * np.interp(x, self.hdx_sim_n[0], self.hdx_sim_n[1]) +
            Bg_norm * bg +
            Acc_norm * acc
        )
        return simu

    def He3_fit_dists(self, more=False):
        import numpy as np
        from scipy.optimize import curve_fit
        if not self.bg_shape_set:
            raise ValueError("bg shape has not been set!")

        if self.hdx_data is None or self.hdx_sim_p is None or self.hdx_sim_n is None or self.hdx_bg_data is None:
            raise ValueError("Histograms have not been set!")

        self.hdx_data = list(self.hdx_data)
        self.hdx_sim_p = list(self.hdx_sim_p)
        self.hdx_sim_n = list(self.hdx_sim_n)
        self.hdx_bg_data = list(self.hdx_bg_data)

        scale = np.sum(self.hdx_data[1])
        self.hdx_data[1] /= scale
        self.hdx_sim_p[1] /= np.sum(self.hdx_sim_p[1])
        self.hdx_sim_n[1] /= np.sum(self.hdx_sim_n[1])
        self.hdx_bg_data[1] /= np.sum(self.hdx_bg_data[1])

        if self.bg_shape_option in ["pol4", "pol3", "pol2", "gaus"]:
            npar = {"pol4": 5, "pol3": 4, "pol2": 3, "gaus": 3}[self.bg_shape_option]
            extra_norms = 0
        elif self.bg_shape_option == "from data":
            npar = 0
            extra_norms = 0
        elif self.bg_shape_option == "from data+acc":
            npar = 0
            extra_norms = 1
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        p0 = [1.0] * (3 + extra_norms + npar)
        bounds = (
            [0.1, 0.1, 0] + [0] * extra_norms + [-np.inf] * npar,
            [100, 100, 100] + [10] * extra_norms + [np.inf] * npar
        )

        x_data = self.hdx_data[0]
        y_data = self.hdx_data[1]

        popt, _ = curve_fit(self.fitsim, x_data, y_data, p0=p0, bounds=bounds)

        self.hdx_sim_p[1] *= popt[0]
        self.hdx_sim_n[1] *= popt[0] * popt[1]

        # Handle bg fit
        if self.bg_shape_option in ["from data", "from data+acc"]:
            hdx_bg_fit = np.interp(x_data, self.hdx_bg_data[0], self.hdx_bg_data[1])
        elif self.bg_shape_option == "gaus":
            hdx_bg_fit = self.fitbg_gaus(x_data, popt[3], popt[4], popt[5])
        else:
            bg_fit = {
                "pol4": self.fitbg_pol4,
                "pol3": self.fitbg_pol3,
                "pol2": self.fitbg_pol2
            }[self.bg_shape_option]
            hdx_bg_fit = bg_fit(x_data, *popt[3:])

        hdx_bg_fit *= (popt[0] * popt[2])

        hdx_total_fit = self.hdx_sim_p[1] + self.hdx_sim_n[1] + hdx_bg_fit

        if self.bg_shape_option == "from data+acc":
            print("Fitting BG with inelasticsim and accidentals")
            hdx_acc_fit = np.interp(x_data, self.hdx_acc_data[0], self.hdx_acc_data[1])
            hdx_acc_fit *= (popt[0] * popt[3])
            hdx_total_fit += hdx_acc_fit

        self.hdx_data[1] *= scale
        self.hdx_sim_p[1] *= scale
        self.hdx_sim_n[1] *= scale
        hdx_bg_fit *= scale
        hdx_total_fit *= scale

        if self.bg_shape_option == "from data+acc":
            hdx_acc_fit *= scale

        if more:
            if self.bg_shape_option == "from data+acc":
                return hdx_bg_fit, hdx_total_fit, self.hdx_sim_p[1], self.hdx_sim_n[1], popt[0], popt[1], popt[2], popt[3], scale, hdx_acc_fit
            else:
                return hdx_bg_fit, hdx_total_fit, self.hdx_sim_p[1], self.hdx_sim_n[1], popt[0], popt[1], popt[2], scale

        return hdx_bg_fit, hdx_total_fit, self.hdx_sim_p[1], self.hdx_sim_n[1]
