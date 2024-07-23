import numpy as np
from scipy.optimize import curve_fit

class DistributionFits2D:
    def __init__(self, bg_shape_option='pol4'):
        self.bg_shape_option = bg_shape_option
        self.bg_shape_set = True
        self.hdx_data = None
        self.hdx_sim_p = None
        self.hdx_sim_n = None
        self.hdx_bg_data = None

    def fitbg_pol4(self, xy, *par):
        print(f"Creating polynomial of 4th order with parameters: {par}")
        x, y = xy
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        bg = np.zeros_like(x)
        index = 0
        for i in range(5):
            for j in range(5 - i):
                bg += par[index] * (x ** i) * (y ** j)
                index += 1
        print(f"Generated 4th order polynomial shape: {bg.shape}")
        return bg

    def fitbg_pol3(self, xy, *par):
        print(f"Creating polynomial of 3rd order with parameters: {par}")
        x, y = xy
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        bg = np.zeros_like(x)
        index = 0
        for i in range(4):
            for j in range(4 - i):
                bg += par[index] * (x ** i) * (y ** j)
                index += 1
        print(f"Generated 3rd order polynomial shape: {bg.shape}")
        return bg

    def fitbg_pol2(self, xy, *par):
        print(f"Creating polynomial of 2nd order with parameters: {par}")
        x, y = xy
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        bg = np.zeros_like(x)
        index = 0
        for i in range(3):
            for j in range(3 - i):
                bg += par[index] * (x ** i) * (y ** j)
                index += 1
        print(f"Generated 2nd order polynomial shape: {bg.shape}")
        return bg

    def fitbg_gaus(self, xy, a, mux, muy, sigmax, sigmay):
        print(f"Creating Gaussian with parameters: a={a}, mux={mux}, muy={muy}, sigmax={sigmax}, sigmay={sigmay}")
        x, y = xy
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        bg = a * np.exp(-0.5 * (((x - mux) / sigmax) ** 2 + ((y - muy) / sigmay) ** 2))
        print(f"Generated Gaussian shape: {bg.shape}")
        return bg

    def fitsim(self, xy, Norm_overall, R_pn, Bg_norm, *par):
        print(f"Running fitsim with parameters: Norm_overall={Norm_overall}, R_pn={R_pn}, Bg_norm={Bg_norm}, par={par}")
        x, y = xy
        if self.bg_shape_option == "pol4":
            bg = self.fitbg_pol4((x, y), *par)
        elif self.bg_shape_option == "pol3":
            bg = self.fitbg_pol3((x, y), *par)
        elif self.bg_shape_option == "pol2":
            bg = self.fitbg_pol2((x, y), *par)
        elif self.bg_shape_option == "gaus":
            bg = self.fitbg_gaus((x, y), *par[:5])
        elif self.bg_shape_option == "from data":
            bg = self.hdx_bg_data.flatten()
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        print(f"Background shape after fitting: {bg.shape}")

        if self.bg_shape_option != "from data":
            bg = bg.reshape(self.hdx_sim_p.shape).flatten()
        print(f"Background shape after reshaping: {bg.shape}")

        simu = Norm_overall * (self.hdx_sim_p.flatten() + R_pn * self.hdx_sim_n.flatten() + Bg_norm * bg)
        print(f"Simulation shape: {simu.shape}")
        return simu

    def He3_fit_dists(self):
        if not self.bg_shape_set:
            raise ValueError("bg shape has not been set!")

        if self.hdx_data is None or self.hdx_sim_p is None or self.hdx_sim_n is None or self.hdx_bg_data is None:
            raise ValueError("Histograms have not been set!")

        print("Normalizing histograms...")
        scale = np.sum(self.hdx_data)
        self.hdx_data /= scale
        self.hdx_sim_p /= np.sum(self.hdx_sim_p)
        self.hdx_sim_n /= np.sum(self.hdx_sim_n)
        self.hdx_bg_data /= np.sum(self.hdx_bg_data)

        print(f"hdx_data shape: {self.hdx_data.shape}")
        print(f"hdx_sim_p shape: {self.hdx_sim_p.shape}")
        print(f"hdx_sim_n shape: {self.hdx_sim_n.shape}")
        print(f"hdx_bg_data shape: {self.hdx_bg_data.shape}")

        # Create meshgrid for x and y with correct dimensions
        y = np.arange(self.hdx_data.shape[0])
        x = np.arange(self.hdx_data.shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        xy = np.array([xv.flatten(), yv.flatten()])

        y_data = self.hdx_data.flatten()
        print(f"Flattened hdx_data shape: {y_data.shape}")

        if self.bg_shape_option == "pol4":
            npar = 15
        elif self.bg_shape_option == "pol3":
            npar = 10
        elif self.bg_shape_option == "pol2":
            npar = 6
        elif self.bg_shape_option == "gaus":
            npar = 5
        elif self.bg_shape_option == "from data":
            npar = 0
        else:
            raise ValueError(f"Unsupported bg_shape_option: {self.bg_shape_option}")

        p0 = [1.0] * (3 + npar)
        bounds = ([0.1, 0.1, 0] + [-np.inf] * npar, [100, 100, 100] + [np.inf] * npar)
        print(f"Initial parameters: {p0}")
        print(f"Bounds: {bounds}")

        popt, _ = curve_fit(self.fitsim, xy, y_data, p0=p0, bounds=bounds)

        print(f"Optimized parameters: {popt}")

        self.hdx_sim_p *= popt[0]
        self.hdx_sim_n *= popt[0] * popt[1]

        if self.bg_shape_option == "from data":
            hdx_bg_fit = self.hdx_bg_data
        elif self.bg_shape_option == "gaus":
            hdx_bg_fit = self.fitbg_gaus(xy, popt[3], popt[4], popt[5], popt[6], popt[7]).reshape(self.hdx_data.shape)
        else:
            bg_fit = self.fitbg_pol4 if self.bg_shape_option == "pol4" else self.fitbg_pol3 if self.bg_shape_option == "pol3" else self.fitbg_pol2
            hdx_bg_fit = bg_fit(xy, *popt[3:]).reshape(self.hdx_data.shape)

        hdx_bg_fit *= (popt[0] * popt[2])

        hdx_total_fit = self.hdx_sim_p + self.hdx_sim_n + hdx_bg_fit

        self.hdx_data *= scale
        self.hdx_sim_p *= scale
        self.hdx_sim_n *= scale
        hdx_bg_fit *= scale
        hdx_total_fit *= scale

        print(f"Final background fit shape: {hdx_bg_fit.shape}")
        print(f"Final total fit shape: {hdx_total_fit.shape}")

        return hdx_bg_fit, hdx_total_fit, self.hdx_sim_p, self.hdx_sim_n
