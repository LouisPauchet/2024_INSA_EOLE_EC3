import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, weibull_min
from scipy.interpolate import interp1d
import numpy as np
import os

class ProcessWind():
    def __init__(self, paths, cp_path = None):
        self.paths = paths
        self._load_data(paths)
        if cp_path is not None:
            self._load_cp(path=cp_path)
        else:
            self._load_cp()
        self.params = {}
        self.physics = {
            "air_density" : 1.225,
            "opex_per_MW_per_year" : 1e5,
            "life_time_year" : 25,
            "decommissioning_per_MW" : 2e5,
            "discount_rate" : 0.07

        }
        self.cp_interpolator = None
        self.turbines = [
            {
                "name" : "Small_Rotor_Small_Gen",
                "rotor_diam" : 130,
                "mass" : 250,
                "blade_tip_ground_clearance" : 30,
                "rated_power" : 5,
                "min_rot_speed" : 3.5,
                "max_rot_speed" : 9,
                "cut_in" : 3,
                "cut_out" : 25,
                "tower_eq_load" : 15,
                "efficiency" : 0.6,
                "cost" : 15e6
            },
            {
                "name": "Small_Rotor_Large_Gen",
                "rotor_diam": 130,
                "mass": 400,
                "blade_tip_ground_clearance": 30,
                "rated_power": 8,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 15,
                "efficiency" : 0.6,
                "cost" : 20e6

            },
            {
                "name": "Large_Rotor_Small_Gen",
                "rotor_diam": 160,
                "mass": 400,
                "blade_tip_ground_clearance": 30,
                "rated_power": 5,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 20,
                "efficiency" : 0.6,
                "cost" : 20e6
            },
            {
                "name": "Large_Rotor_Large_Gen",
                "rotor_diam": 160,
                "mass": 600,
                "blade_tip_ground_clearance": 30,
                "rated_power": 8,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 25,
                "efficiency" : 0.6,
                "cost" : 24e6
            }
        ]
        self.results = []
        self.wind_speeds = []

    def _load_cp(self, path = "./data/cp_data.dat", polyfit_degree = 5):
        cp_data = pd.read_csv(path, sep='\s+', header=1, on_bad_lines='skip', names=["TSR", "Cp"])

        coeffs = np.polyfit(cp_data['TSR'], cp_data['Cp'], polyfit_degree)
        poly = np.poly1d(coeffs)

        TSR_fine = np.linspace(cp_data['TSR'].min(), cp_data['TSR'].max(), 1000)
        Cp_fine = poly(TSR_fine)

        df_fine = pd.DataFrame(
            {
                'Cp' : Cp_fine,
                'TSR' : TSR_fine,
            }
        )

        max_index = np.argmax(Cp_fine)
        max_Cp = Cp_fine[max_index]
        corresponding_TSR = TSR_fine[max_index]

        self.cp_data = {
            "df" : cp_data,
            "coeffs" : coeffs,
            "poly" : poly,
            "degree" : polyfit_degree,
            "df_fine" : df_fine,
            "Cp_max" : max_Cp,
            "TSR_max_Cp" : corresponding_TSR
        }

    def plot_cp(self, save=None, latex=False):
        """
        Plot the Cp vs TSR curve along with the Betz limit and maximum Cp point.

        Parameters:
            save (str, optional): File path to save the plot. Default is None.
            latex (bool, optional): Use LaTeX for the annotation text. Default is False.
        """
        # Apply seaborn theme
        sns.set_theme(style="ticks")

        # Create the figure
        fig = plt.figure(figsize=(6, 4))

        # Plot the Cp curve (fitted and data points)
        plt.plot(self.cp_data["df_fine"]["TSR"], self.cp_data["df_fine"]["Cp"],
                 label=f"Polynomial Fit (Degree {self.cp_data['degree']})", color="blue")
        plt.scatter(self.cp_data["df"]["TSR"], self.cp_data["df"]["Cp"], label="Data Points", color="orange",
                    marker="+")

        # Plot Betz limit
        betz_limit = 16 / 27
        plt.axhline(betz_limit, linestyle="--", color="red", label="Betz Limit")

        # Mark the maximum Cp point
        max_Cp = self.cp_data["Cp_max"]
        TSR_max_Cp = self.cp_data["TSR_max_Cp"]

        # Add the maximum Cp point with the value in the legend
        max_label = (
            rf"Max Cp ($C_p = {max_Cp:.2f}$, TSR = {TSR_max_Cp:.2f})"
            if latex else
            f"Max Cp (Cp = {max_Cp:.2f}, TSR = {TSR_max_Cp:.2f})"
        )
        plt.plot([TSR_max_Cp], [max_Cp], "or", label=max_label)

        # Add labels, title, and legend
        plt.xlabel("TSR")
        plt.ylabel("Cp")
        plt.title("Power Coefficient (Cp) vs Tip-Speed Ratio (TSR)")
        plt.legend()

        # Save the figure if needed
        if save is not None:
            if not isinstance(save, str):
                raise ValueError("The 'save' parameter must be a string specifying the file path.")
            plt.savefig(save, dpi=300, transparent=True, bbox_inches="tight")

        # Show the plot
        plt.show()

    def _load_data(self, paths):
        """
        Load and process multiple time series data files into a unified DataFrame.

        Parameters:
            paths (list of str): List of file paths to the data files. Each file should contain:
                - "time": A timestamp column (format: "%d-%b-%Y-%H:%M:%S").
                - A second column with numeric data (e.g., wind speed) named based on the file name.

        Returns:
            None: The data is stored in `self.df` as a Pandas DataFrame with a datetime index.
        """
        for i, path in enumerate(paths):
            name = (path.split('/')[-1]).split(".")[0]
            df = pd.read_csv(path, sep='\s+', header=1, on_bad_lines='skip',
                             names=["time", name])
            df["time"] = pd.to_datetime(df["time"], format="%d-%b-%Y-%H:%M:%S")
            df.set_index("time", inplace=True)
            if i==0:
                self.df = df
            else:
                self.df = pd.concat([self.df, df])

    def box_plot(self, labels=None, save=None, title='Boxplot of Wind Speeds', ylabel='Wind Speed (m/s)'):
        """
        Create a boxplot for the loaded time series data.

        Parameters:
            labels (list of str, optional): Custom labels for the boxplot's x-axis.
                Defaults to the column names of `self.df`.
            save (str, optional): File path to save the boxplot image. Default is None.
            title (str, optional): Title of the boxplot. Default is 'Boxplot of Wind Speeds'.
            ylabel (str, optional): Label for the y-axis. Default is 'Wind Speed (m/s)'.

        Returns:
            None: Displays the boxplot. Optionally saves it if `save` is specified.
        """
        if labels is None:
            labels = [key for key in self.df.keys()]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, orient='v', palette="Set2")
        plt.title(title, fontsize=16)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks([i for i in range(len(self.df.keys()))], labels, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save is not None:
            plt.savefig(save, dpi=300)
        plt.show()

    def distribution(self, bins=50, save=None, title='Histogram of Wind Speeds', xlabel='Wind Speed (m/s)',
                     ylabel='Frequency', labels=None):
        """
        Plots histograms for numerical columns and overlays Weibull fit if parameters exist.

        Parameters:
            bins (int): Number of bins for the histograms.
            save (str): Path to save the plot. Default is None (don't save).
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            labels (list): Custom labels for the columns. Default is None (uses column names).
        """
        plt.figure(figsize=(12, 6))

        # Use column names as labels if no custom labels are provided
        if labels is None:
            labels = self.df.columns

        # Iterate through numerical columns and their labels
        for col, label in zip(self.df.columns, labels):
            if self.df[col].dtype in ['float64', 'int64']:  # Only plot numerical columns
                # Plot histogram
                data = self.df[col].dropna()
                plt.hist(data, bins=bins, alpha=0.6, label=f'{label} - Data', edgecolor='black', density=True)

                # Overlay Weibull fit if parameters exist
                if col in self.params:
                    shape = self.params[col]['shape']
                    scale = self.params[col]['scale']
                    loc = self.params[col]['location']

                    # Generate Weibull PDF
                    x = np.linspace(data.min(), data.max(), 500)
                    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)

                    # Add Weibull fit to the plot with parameters in the legend
                    plt.plot(x, weibull_pdf,
                             label=f'{label} Weibull Fit\nShape (k): {shape:.2f}\nScale (λ): {scale:.2f}\nLocation: {loc:.2f}',
                             linewidth=2)

        # Add labels, title, and legend
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim(left=0)  # Ensure x-axis starts at 0

        # Save the plot if a path is provided
        if save is not None:
            plt.savefig(save, dpi=300)

        ax = plt.gca()
        # Display the plot
        plt.show()
        return ax

    def clean_data(self):
        for col in self.df.keys():
            if self.df[col].dtype in ['float64', 'int64']:  # Only apply to numerical columns
                # Create a new column for Z-scores
                self.df.loc[:, 'Z_score'] = zscore(self.df[col], nan_policy='omit')

                # Filter rows using Z-score threshold
                self.df = self.df.loc[(self.df['Z_score'].abs() <= 3) | (self.df[col].isnull())].copy()

                # Drop the temporary Z-score column
                self.df.drop(columns=['Z_score'], inplace=True)

        max_wind = 0
        for wind in self.df.keys():
            max_wind = max(max_wind, self.df[wind].max())
        self.wind_speeds = np.linspace(0, max_wind, len(self.df))

        return self.df

    def fit_weibull(self):
        """
        Fits a Weibull distribution to each numerical column in the DataFrame.
        Stores the shape (k), scale (λ), and location parameters in self.params.
        """
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:  # Only process numerical columns
                # Drop NaN values for fitting
                data = self.df[col].dropna()

                # Fit the Weibull distribution
                shape, loc, scale = weibull_min.fit(data, floc=0)  # Fix location to 0

                # Store the parameters in a dictionary
                self.params[col] = {
                    'wind_distribution': col,
                    'shape': shape,  # k
                    'scale': scale,  # λ
                    'location': loc  # Should be 0 as floc=0
                }

        return self.params

    def _calc_turbine_power_curve(self, wind_speed, r_rotor, cut_in, cut_out, rho=1.225, eta=0.6, Cp=0.593):
        """
        Calculate the power generated by a wind turbine.

        Parameters:
            wind_speed (array-like): Wind speed in meters per second (m/s).
            r_rotor (float): Rotor radius in meters (m).
            cut_in (float): Cut-in wind speed in m/s.
            cut_out (float): Cut-out wind speed in m/s.
            rho (float, optional): Air density in kilograms per cubic meter (kg/m³). Default is 1.225 kg/m³.
            eta (float, optional): Efficiency of the wind turbine. Default is 0.6.
            Cp (float, optional): Power coefficient of the turbine. Default is Betz limit (0.593).

        Returns:
            array-like: Generated power in watts (W).
        """
        # Calculate rotor area
        A = np.pi * r_rotor ** 2

        # Calculate theoretical power
        power = 0.5 * rho * A * wind_speed ** 3 * Cp * eta

        # Set power to 0 outside the cut-in and cut-out range
        power = np.where((wind_speed < cut_in) | (wind_speed > cut_out), 0, power)

        return power

    def _calc_turbine_rotation_speed(self, wind_speed, r_rotor, TSR, cut_in, cut_out, rot_min, rot_max):
        """
        Calculate the rotational speed of the wind turbine rotor.

        Parameters:
            wind_speed (float): Wind speed in meters per second (m/s).
            r_rotor (float): Rotor radius in meters (m).
            TSR (float): Tip-speed ratio of the turbine.
            rot_min (float): Minimum rotational speed in radians per second (rpm).
            rot_max (float): Maximum rotational speed in radians per second (rpm).

        Returns:
            float: Rotational speed in radians per second (rad/s).
        """
        rot_min, rot_max = self.rotmin2rads(rot_min), self.rotmin2rads(rot_max)
        # Rotational speed in radians per second
        omega = (TSR * wind_speed) / r_rotor

        # Set power to 0 outside the cut-in and cut-out range
        omega = np.where((wind_speed < cut_in) | (wind_speed > cut_out), 0, omega)

        # Enforce rotational speed limits
        omega = np.clip(omega, rot_min, rot_max)

        return omega

    def rads2rotmin(self, omega):
        """
        Convert rotational speed from radians per second (rad/s) to revolutions per minute (rpm).

        Parameters:
            omega (float): Rotational speed in radians per second (rad/s).

        Returns:
            float: Rotational speed in revolutions per minute (rpm).
        """
        # Convert rad/s to rpm
        rot_min = omega * (60 / (2 * np.pi))
        return rot_min

    def rotmin2rads(self, omega):
        """
        Convert rotational speed from revolutions per minute (rpm) to radians per second (rad/s).

        Parameters:
            omega (float): Rotational speed in revolutions per minute (rpm).

        Returns:
            float: Rotational speed in radians per second (rad/s).
        """
        # Convert rpm to rad/s
        rad_sec = omega * ((2 * np.pi) / 60)
        return rad_sec

    def _calc_torque_turbine(self, P, omega):
        """
        Calculate the torque generated by the turbine.

        Parameters:
            P (float): Power in watts (W).
            omega (float): Rotational speed in radians per second (rad/s).

        Returns:
            float: Torque in newton-meters (Nm). If omega is zero, torque will be zero to avoid division by zero.
        """
        # Torque = Power / Rotational Speed
        # Ensure no division by zero
        torque = np.where(omega != 0, P / omega, 0)
        return torque

    def _calc_wind_speed(self, shape, scale, loc, cut_in, cut_out, plot=False, max_wind_speed=None, points=1000):
        """
        Generate wind speed distribution based on the Weibull probability density function.

        Parameters:
            shape (float): Shape parameter (k) of the Weibull distribution.
            scale (float): Scale parameter (λ) of the Weibull distribution.
            loc (float): Location parameter (typically 0 for wind speed).
            cut_in (float): Cut-in wind speed in meters per second (m/s).
            cut_out (float): Cut-out wind speed in meters per second (m/s).
            plot (bool, optional): Whether to plot the wind speed probability distribution. Default is False.
            max_wind_speed (float, optional): Maximum wind speed for the range. Default is 1.5 * cut_out.
            points (int, optional): Number of points in the wind speed range. Default is 1000.

        Returns:
            tuple: A tuple containing:
                - wind_speed (numpy.ndarray): Array of wind speeds in meters per second (m/s).
                - wind_probability (numpy.ndarray): Probability density function values for the turbine's operating range.

        Raises:
            AssertionError: If the integral of the PDF is not close to 1 after applying cut-in and cut-out limits.

        Notes:
            - The integral of the Weibull PDF is ensured to be ≤ 1 within the wind speed range.
            - The function can plot the distribution if `plot` is set to True.
        """
        if max_wind_speed is None:
            max_wind_speed = cut_out * 1.5

        # Generate wind speed range
        wind_speed = np.linspace(0, max_wind_speed, points)

        # Calculate Weibull probability density function
        wind_probability_raw = weibull_min.pdf(wind_speed, shape, loc=loc, scale=scale)

        wind_probability = wind_probability_raw.copy()
        wind_probability[(wind_speed < cut_in) | (wind_speed > cut_out)] = 0

        # Assert the integral of PDF within the wind speed range is close to 1
        integral = np.trapezoid(wind_probability, wind_speed)
        assert integral <= 1, f"Integral of wind probability is not close to 1: {integral}"

        # Plot if required
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(wind_speed, wind_probability_raw, label="Original Weibull PDF", linestyle="--")
            plt.plot(wind_speed, wind_probability, label="Turbine Density (cut-in/out applied)", linestyle="-")
            plt.axvline(cut_in, color='g', linestyle='--', label="Cut-in Speed")
            plt.axvline(cut_out, color='r', linestyle='--', label="Cut-out Speed")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Probability Density")
            plt.title("Wind Speed Probability Distribution")
            plt.legend()
            plt.grid()
            plt.show()

        return wind_speed, wind_probability

    def _calc_performance_turbine(self, turbine, param, Cp=0.593, TSR=7.5, plot=False, nb_points=None):
        """
        Calculate the performance metrics of a wind turbine.

        Parameters:
            turbine (dict): Turbine specifications, including:
                - 'rotor_diam' (float): Rotor diameter in meters.
                - 'efficiency' (float): Efficiency of the turbine (fractional).
                - 'cut_in' (float): Cut-in wind speed in m/s.
                - 'cut_out' (float): Cut-out wind speed in m/s.
            param (dict): Weibull distribution parameters, including:
                - 'shape' (float): Shape parameter (k).
                - 'scale' (float): Scale parameter (λ).
                - 'location' (float): Location parameter (floc, typically 0).
            Cp (float, optional): Power coefficient of the turbine (default is 0.593).
            TSR (float, optional): Tip-speed ratio (default is 7.5).
            plot (bool, optional): Whether to plot the turbine performance metrics (default is False).
            nb_points (int, optional): Number of points to compute (Default is None)

        Returns:
            dict: A dictionary containing:
                - 'wind_speeds_m/s': Wind speed values (m/s).
                - 'wind_distribution': Probability density of wind speeds.
                - 'power_W': Power output at each wind speed (W).
                - 'rotational_speed_rad/s': Rotational speed at each wind speed (rad/s).
                - 'torque_Nm': Torque at each wind speed (Nm).
                - 'aep_Wh': Annual energy production (Wh/year).
        """
        result = {}
        # Prepare additional parameters
        extra_args = {}
        if nb_points is not None:
            extra_args['points'] = nb_points

        # Calculate wind speed distribution
        wind_speeds, wind_distribution = self._calc_wind_speed(
            param['shape'], param['scale'], param['location'],
            turbine['cut_in'], turbine['cut_out'],
            **extra_args  # Unpack additional parameters
        )

        result['wind_speeds_m/s'] = wind_speeds
        result['wind_distribution'] = wind_distribution

        # Calculate the power curve for the turbine
        power = self._calc_turbine_power_curve(
            wind_speeds, turbine['rotor_diam'] / 2,
            turbine['cut_in'], turbine['cut_out'],
            eta=turbine['efficiency'], rho=self.physics['air_density']
        )
        power = np.clip(power, 0, turbine['rated_power'] * 1e6) #rated_power is in MW
        result['power_W'] = power

        # Calculate the rotational speed
        rotational_speed = self._calc_turbine_rotation_speed(
            wind_speeds,
            turbine['rotor_diam'] / 2,
            TSR,
            turbine['cut_in'],
            turbine['cut_out'],
            turbine["min_rot_speed"],
            turbine["max_rot_speed"]
        )
        result['rotational_speed_rad/s'] = rotational_speed

        # Calculate the torque
        torque = self._calc_torque_turbine(power, rotational_speed)
        result['torque_Nm'] = torque

        # Compute the Annual Energy Production (AEP)
        aep = np.trapezoid(power * wind_distribution, wind_speeds) * 3600 * 24 * 365  # AEP in Joules/year
        aep = aep / 3600  # Convert AEP to Wh/year
        result['aep_Wh'] = aep

        #Compute Capacity Factor
        rated_annual_power = turbine['rated_power'] * 1e6 #Conversion in MW in W
        rated_annual_power = rated_annual_power * 24 * 365
        capacity_factor = aep / rated_annual_power * 100 #Conversion in %

        result['capacity_factor_%'] = capacity_factor

        # Compute LCOE
        total_cost = turbine['cost'] + (
                    self.physics['life_time_year'] * self.physics['opex_per_MW_per_year'] * turbine['rated_power']) + (
            (self.physics["decommissioning_per_MW"] * turbine['rated_power']) / ((1+self.physics["discount_rate"]) ** self.physics['life_time_year'])
        )
        #Convert to MWh
        total_energy = aep * self.physics['life_time_year'] / 1e6
        LCOE = total_cost / total_energy


        result['LCOE_€/MWh'] = LCOE
        result['total_cost'] = total_cost
        result['total_energy'] = total_energy

        # Plot the performance metrics if requested
        if plot:

            # Plot the wind distribution
            plt.figure()
            plt.plot(wind_speeds, wind_distribution, label="Wind Distribution")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Probability Density")
            plt.title("Wind Speed Distribution")
            plt.legend()
            plt.grid()

            # Plot the power curve
            plt.figure()
            plt.plot(wind_speeds, power, label="Power Output")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Power (W)")
            plt.title("Power Curve")
            plt.legend()
            plt.grid()

            # Plot the rotational speed
            plt.figure()
            plt.plot(wind_speeds, self.rads2rotmin(rotational_speed), label="Rotational Speed")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Rotational Speed (rot/min)")
            plt.title("Rotational Speed vs Wind Speed")
            plt.legend()
            plt.grid()

            # Plot the torque curve
            plt.figure()
            plt.plot(wind_speeds, torque, label="Torque")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Torque (Nm)")
            plt.title("Torque vs Wind Speed")
            plt.legend()
            plt.grid()

            plt.show()

        return result

    def calc_performances(self):
        """
        Calculate and return performance metrics for turbines under different wind distributions.

        The method uses `self.params` (Weibull fit parameters) and `self.turbines` (turbine specifications)
        to compute metrics like power output and annual energy production (AEP) using `_calc_performance_turbine`.

        Returns:
            list: A list of dictionaries, each containing:
                - "wind_distribution" (str): Wind distribution name.
                - "turbine" (str): Turbine name.
                - Other performance metrics like power, torque, rotational speed, and AEP.

        Notes:
            - Results are stored in `self.results` and returned.
            - Each turbine is evaluated for all wind distributions.
        """
        self.results = []
        for key in self.params.keys():
            for turbine in self.turbines:
                result = self._calc_performance_turbine(
                    turbine, self.params[key],
                    Cp=self.cp_data['Cp_max'], TSR=self.cp_data['TSR_max_Cp'])
                result["wind_distribution_name"] = key
                result["turbine"] = turbine['name']

                self.results.append(result)

        return self.results

    def _plot_performances_parameters_multiplot(self, save=None, turbine_names=None, wind_distribution_names=None, cmap='cividis'):
        """
        Plots AEP vs. Capacity Factor with LCOE as a color bar for each wind distribution.

        Parameters:
            save (str): Path to save the plot. If None, the plot is not saved.
            turbine_names (dict): Optional dictionary to map turbine IDs to custom names.
                                  Example: {'Turbine1': 'Custom Name 1'}
            wind_distribution_names (dict): Optional dictionary to map wind distribution IDs to custom names.
                                             Example: {'WindDist1': 'Custom Wind 1'}
            cmap (str): Colormap for LCOE visualization.
        """

        if self.results is None:
            self.calc_performances()

        # Set Seaborn style
        sns.set_theme(style='ticks')

        # Extract unique wind distributions and turbines
        wind_distributions = list(set(res['wind_distribution_name'] for res in self.results))
        turbines = list(set(res['turbine'] for res in self.results))
        markers = ['o', 's', '^', 'D', 'P', '*']  # Define markers for turbines

        # Apply custom names if provided
        wind_dist_labels = (
            {wd: wind_distribution_names.get(wd, wd) for wd in wind_distributions}
            if wind_distribution_names
            else {wd: wd for wd in wind_distributions}
        )
        turbine_labels = (
            {t: turbine_names.get(t, t) for t in turbines}
            if turbine_names
            else {t: t for t in turbines}
        )

        # Create a vertical subplot for each wind distribution
        fig, axes = plt.subplots(len(wind_distributions), 1, figsize=(7, 3 * len(wind_distributions)))

        if len(wind_distributions) == 1:
            axes = [axes]  # Ensure axes is a list when there's only one subplot

        # Track handles for a unified legend
        handles = []

        for ax, wind_dist in zip(axes, wind_distributions):
            # Filter results for the current wind distribution
            filtered_results = [res for res in self.results if res['wind_distribution_name'] == wind_dist]

            # Extract the min and max LCOE values for this wind distribution
            lcoe_values = [res['LCOE_€/MWh'] for res in filtered_results]
            lcoe_min, lcoe_max = min(lcoe_values), max(lcoe_values)

            for turbine, marker in zip(turbines, markers):
                # Filter data for the current turbine
                turbine_data = [res for res in filtered_results if res['turbine'] == turbine]

                # Extract values for plotting
                aep = [res['aep_Wh'] / 1e9 for res in turbine_data]  # Convert AEP to GWh
                capacity_factor = [res['capacity_factor_%'] for res in turbine_data]
                lcoe = [res['LCOE_€/MWh'] for res in turbine_data]

                # Scatter plot
                sc = ax.scatter(
                    aep,
                    capacity_factor,
                    c=lcoe,
                    cmap=cmap,
                    label=turbine_labels[turbine],
                    s=100,
                    marker=marker,
                    edgecolors='black',
                    vmin=lcoe_min,  # Set the min value for the colorbar
                    vmax=lcoe_max  # Set the max value for the colorbar
                )

                # Collect legend handles (black marker)
                if len(handles) < len(turbines):  # Add each turbine only once
                    handles.append(plt.Line2D([], [], color='black', marker=marker, linestyle='', markersize=10,
                                              label=turbine_labels[turbine]))

            # Add a dedicated color bar for each subplot
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
            cbar.set_label('LCOE (€/MWh)')
            cbar.ax.set_ylim(lcoe_min, lcoe_max)  # Dynamically set colorbar limits

            ax.set_title(f'Wind Distribution: {wind_dist_labels[wind_dist]}', fontsize=13, weight='bold')
            ax.set_xlabel('AEP (GWh)', fontsize=12)
            ax.set_ylabel('Capacity Factor (%)', fontsize=12)
            ax.grid(linestyle='--', alpha=0.7)
            ax.set_xlim(0, 50)  # Adjust x-axis limits
            ax.set_ylim(0, 100)  # Adjust y-axis limits

        # Add a unified legend below the plots
        fig.legend(handles=handles, loc='lower center', ncol=len(turbines), title='Turbines',
                   bbox_to_anchor=(0.5, -0.08), fontsize=10, title_fontsize=12)

        plt.tight_layout()

        # Save the plot if save_path is provided
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)  # Create directory if it doesn't exist
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save}")

        # Show the plot
        plt.show()


    def _plot_performances_parameters_monoplot(self, save=None, turbine_names=None, wind_distribution_names=None,
                                     cmap='viridis'):
        """
        Plots AEP vs. Capacity Factor with LCOE as a color bar, using marker edge colors for wind distributions.

        Parameters:
            save (str): Path to save the plot. If None, the plot is not saved.
            turbine_names (dict): Optional dictionary to map turbine IDs to custom names.
                                  Example: {'Turbine1': 'Custom Name 1'}
            wind_distribution_names (dict): Optional dictionary to map wind distribution IDs to custom names.
                                             Example: {'WindDist1': 'Custom Wind 1'}
            cmap (str): Colormap for LCOE visualization.
        """

        if self.results is None:
            self.calc_performances()

        # Set Seaborn style
        sns.set_theme(style='ticks')

        # Extract unique wind distributions and turbines
        wind_distributions = list(set(res['wind_distribution_name'] for res in self.results))
        turbines = list(set(res['turbine'] for res in self.results))
        markers = ['o', 's', '^', 'D', 'P', '*']  # Define markers for turbines

        # Generate unique edge colors for wind distributions
        edge_colors = sns.color_palette('husl', len(wind_distributions))

        # Apply custom names if provided
        wind_dist_labels = (
            {wd: wind_distribution_names.get(wd, wd) for wd in wind_distributions}
            if wind_distribution_names
            else {wd: wd for wd in wind_distributions}
        )
        turbine_labels = (
            {t: turbine_names.get(t, t) for t in turbines}
            if turbine_names
            else {t: t for t in turbines}
        )

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Track handles for turbine and wind distribution legends
        turbine_handles = []
        wind_handles = []
        handle_turbine_legend = True
        for wind_dist, edge_color in zip(wind_distributions, edge_colors):
            # Filter results for the current wind distribution
            filtered_results = [res for res in self.results if res['wind_distribution_name'] == wind_dist]

            for turbine, marker in zip(turbines, markers):
                # Filter data for the current turbine
                turbine_data = [res for res in filtered_results if res['turbine'] == turbine]

                # Extract values for plotting
                aep = [res['aep_Wh'] / 1e9 for res in turbine_data]  # Convert AEP to GWh
                capacity_factor = [res['capacity_factor_%'] for res in turbine_data]
                lcoe = [res['LCOE_€/MWh'] for res in turbine_data]

                # Scatter plot
                sc = ax.scatter(
                    aep,
                    capacity_factor,
                    c=lcoe,
                    cmap=cmap,
                    s=100,
                    marker=marker,
                    edgecolors=edge_color,
                    linewidth=2,
                    vmin=min(res['LCOE_€/MWh'] for res in self.results),
                    vmax=max(res['LCOE_€/MWh'] for res in self.results),
                )

                # Collect turbine handles (black fill with unique markers)
                if handle_turbine_legend:
                    if turbine not in [h.get_label() for h in turbine_handles]:
                        turbine_handles.append(
                            plt.Line2D([], [], color='black', marker=marker, linestyle='', markersize=10,
                                       label=turbine_labels[turbine]))

            # Collect wind distribution handles (edge color with a generic marker)
            if wind_dist not in [h.get_label() for h in wind_handles]:
                wind_handles.append(plt.Line2D([], [], color=edge_color, marker='o', linestyle='', markersize=10,
                                               label=wind_dist_labels[wind_dist]))

            handle_turbine_legend = False

        # Add color bar for LCOE
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
        cbar.set_label('LCOE (€/MWh)')

        # Set axis labels and title
        ax.set_title('Performance Parameters', fontsize=14, weight='bold')
        ax.set_xlabel('AEP (GWh)', fontsize=12)
        ax.set_ylabel('Capacity Factor (%)', fontsize=12)
        ax.grid(linestyle='--', alpha=0.7)
        ax.set_xlim(0, 50)  # Adjust x-axis limits
        ax.set_ylim(0, 100)  # Adjust y-axis limits

        # Add legends
        legend1 = ax.legend(handles=turbine_handles, loc='upper left', title='Turbines', fontsize=12,
                            title_fontsize=12)
        ax.add_artist(legend1)  # Add the turbine legend first
        ax.legend(handles=wind_handles, loc='upper right', title='Wind Distributions', fontsize=12,
                  title_fontsize=12)

        plt.tight_layout()

        # Save the plot if save_path is provided
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)  # Create directory if it doesn't exist
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save}")

        # Show the plot
        plt.show()

    def plot_performance(self, save=None, turbine_names=None, wind_distribution_names=None,
                         cmap='viridis', multiplot=False):

        if multiplot:
            self._plot_performances_parameters_multiplot(save=save, turbine_names=turbine_names,
                                                    wind_distribution_names=wind_distribution_names, cmap=cmap)
        else:
            self._plot_performances_parameters_monoplot(save=save, turbine_names=turbine_names,
                                                   wind_distribution_names=wind_distribution_names, cmap=cmap)












