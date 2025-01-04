import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, weibull_min
from scipy.interpolate import interp1d
import numpy as np
import os

class assignement_parameters():
    def __init__(self):
        self.physics = {
            "air_density" : 1.225,
            "opex_per_MW_per_year" : 1e5,
            "life_time_year" : 25,
            "decommissioning_per_MW" : 2e5,
            "discount_rate" : 0.07,
            "Young_Mod_Steal_GPa" : 210,
            "SN_Curve_MPa" : {
                "log(a)" : 16.3,
                "Wohler_slope_m" : 5
            },
            "n_cycle" : 1e7,
            "max_diam_thickness_ratio" : 350,
            "safety_factor_stress" : 1.25,
            "tip_ground_clearance_m" : 30
        }
        self.turbines = [
            {
                "name": "Small_Rotor_Small_Gen",
                "rotor_diam": 130,
                "mass_RNA_tons": 250,
                "rated_power": 5,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 15,
                "efficiency": 0.6,
                "cost": 15e6
            },
            {
                "name": "Small_Rotor_Large_Gen",
                "rotor_diam": 130,
                "mass_RNA_tons": 400,
                "rated_power": 8,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 15,
                "efficiency": 0.6,
                "cost": 20e6

            },
            {
                "name": "Large_Rotor_Small_Gen",
                "rotor_diam": 160,
                "mass_RNA_tons": 400,
                "rated_power": 5,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 20,
                "efficiency": 0.6,
                "cost": 20e6
            },
            {
                "name": "Large_Rotor_Large_Gen",
                "rotor_diam": 160,
                "mass_RNA_tons": 600,
                "rated_power": 8,
                "min_rot_speed": 3.5,
                "max_rot_speed": 9,
                "cut_in": 3,
                "cut_out": 25,
                "tower_eq_load": 25,
                "efficiency": 0.6,
                "cost": 24e6
            }
        ]

class TowerDesign():
    def __init__(self, selected_turbines):
        self.selected_turbines = selected_turbines
        self.assignement_parameters = assignement_parameters()
        self.physics = self.assignement_parameters.physics
        self.turbines = [
            self._update_turbine_with_distribution(turbine)
            for turbine in self.assignement_parameters.turbines
            if turbine.get('name') in self.selected_turbines.values()
        ]
    
    def _update_turbine_with_distribution(self, turbine):
        for key, value in self.selected_turbines.items():
            if turbine.get('name') == value:
                turbine["wind_distribution"] = key
        return turbine

    def update_turbine(self, turbine_name, parameters, force=False):
        """
        Updates the turbine dictionary with the given parameters.
        Optionally forces the update if `force=True`.

        Parameters:
        turbine_name (str): The name of the turbine to update.
        parameters (dict): A dictionary of parameters to update or add.
        force (bool): Whether to force the update, even if the parameter already exists.
        """
        # Find the turbine by its name
        turbine = next((t for t in self.turbines if t.get('name') == turbine_name), None)

        if turbine is None:
            print(f"Turbine with name '{turbine_name}' not found.")
            return

        # Update turbine with the new parameters
        for param_key, param_value in parameters.items():
            if param_key not in turbine or force:
                turbine[param_key] = param_value
            else:
                print(f"Parameter '{param_key}' already exists and 'force' is not set.")

        return turbine

    def get_turbines(self, turbine_name):
        return [turbine for turbine in self.turbines if turbine_name == turbine.get('name')][0]

    def _calc_tower_height(self, turbine_name):
        """Calculates the height of the tower based on the rotor diameter and tip-to-ground clearance."""
        turbine = self.get_turbines(turbine_name)
        rotor_radius = turbine.get('rotor_diam') / 2
        tower_height = rotor_radius + self.physics['tip_ground_clearance_m']

        self.update_turbine(turbine_name, {"tower_height_m": tower_height}, force=True)

        return tower_height

    def _calc_1P_3P_frequency(self, turbine_name):
        """Calculates the 1P frequency of the turbine (in Hz) based on both max and min rotor speeds, and returns the frequency range."""
        turbine = self.get_turbines(turbine_name)

        # Get the maximum and minimum rotor speeds in RPM
        max_rot_speed_rpm = turbine.get('max_rot_speed')
        min_rot_speed_rpm = turbine.get('min_rot_speed')

        # Convert the rotor speeds from RPM to Hz
        max_f_1P = max_rot_speed_rpm / 60  # Max 1P frequency (Hz)
        min_f_1P = min_rot_speed_rpm / 60  # Min 1P frequency (Hz)

        # Create a DataFrame with the frequency range (start and end)
        frequency_range_1P = {
            'min' : min_f_1P,
            'max' : max_f_1P
        }
        frequency_range_3P = {
            'min' : 3 * min_f_1P,
            'max' : 3 * max_f_1P
        }

        # Update turbine data with the max 1P frequency
        self.update_turbine(turbine_name, {'f_1P': frequency_range_1P, 'f_3P' : frequency_range_3P}, force=True)

        return frequency_range_1P, frequency_range_3P

    def _calc_1st_eigen_frequency(self, turbine_name, d):
        """Calculates the first eigen frequency of the tower based on its dimensions and material properties."""
        turbine = self.get_turbines(turbine_name)

        # Calculate thickness based on D/t ratio
        thickness = (d / self.physics['max_diam_thickness_ratio']) * 2 #Twice the minimum value according the assigment instructions
        inner_diameter = d - (2 * thickness)

        # Calculate the cross-sectional moment of inertia
        moment_of_inertia = np.pi / 64 * (d ** 4 - inner_diameter ** 4)

        # Calculate the first eigen frequency
        tower_height = turbine.get("tower_height_m")
        young_modulus = self.physics.get('Young_Mod_Steal_GPa')
        mass_rna = turbine.get('mass_RNA_tons')  # in tons

        first_eigen_frequency = (3. / tower_height**2) * np.sqrt((young_modulus * 1e6 * moment_of_inertia) / mass_rna)

        self.update_turbine(turbine_name,
                            {'first_eigen_frequency': first_eigen_frequency,
                             'moment_of_inertia' : moment_of_inertia,
                             'thickness' : thickness}, force=True)

        return first_eigen_frequency

    def _calc_fatigue_damage(self, turbine_name, d):
        # Retrieve turbine data
        turbine = self.get_turbines(turbine_name)

        # Calculate the tower thickness based on the D/t ratio (350)
        thickness = turbine.get('thickness')
        inner_diameter = d - 2 * thickness

        # Calculate the moment of inertia (I) for the tower's cylindrical cross-section
        moment_of_inertia = turbine.get('moment_of_inertia')

        S_xy = moment_of_inertia / (d/2)

        # Calculate the equivalent stress (sigma_eq) from the equivalent load
        sigma_eq = turbine.get('tower_eq_load') / S_xy  # Assumes load in MN.m

        # Apply safety factor on the stress
        sigma_eff = sigma_eq * self.physics.get('safety_factor_stress')

        # Retrieve S-N curve parameters from the physics data
        sn_curve = self.physics.get('SN_Curve_MPa')

        # Calculate the number of cycles to failure (N) using the S-N curve
        N = 10 ** (sn_curve.get('log(a)') - sn_curve.get('Wohler_slope_m') * np.log10(sigma_eff))

        # Calculate the fatigue damage using Palmgren-Miner rule (D = n / N)
        fatigue_damage = self.physics.get('n_cycle') / N  # n_cycle is the equivalent number of cycles (e.g., 10^7) - unit %

        # Update turbine data with the calculated fatigue damage
        self.update_turbine(turbine_name, {'fatigue_damage': fatigue_damage, 'N' : N}, force=True)

        return fatigue_damage

    def calc_tower(self, d_min, d_max, n=500):
        d = np.linspace(d_min, d_max, n)
        for turbine in self.turbines:
            self.update_turbine(turbine.get('name'), {"d": d}, force=True)
            self._calc_tower_height(turbine.get('name'))
            self._calc_1P_3P_frequency(turbine.get('name'))
            self._calc_1st_eigen_frequency(turbine.get('name'), d)
            self._calc_fatigue_damage(turbine.get('name'), d)

    def plot_fatigue_and_frequency(self, save = None):
        """
        This function plots the fatigue damage and first eigenfrequency for all turbines,
        using different colors and markers, and includes the 1P and 3P frequency ranges as shaded regions.
        """
        plt.figure(figsize=(10, 5))

        # Color and marker options for different turbines
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        markers = ['o', 's', '^', 'D', 'P', '*']

        # Create axes
        ax = plt.gca()
        ax.set_ylabel("Fatigue Damage")
        ax.set_ylim(0, 1)

        # Create a second y-axis for the first eigen frequency
        ax2 = ax.twinx()
        ax2.set_ylabel('First Eigen Frequency')

        # List to hold legend handles and labels for consolidating the legend
        handles, labels = [], []

        # Iterate through turbines and plot each one
        for idx, turbine in enumerate(self.turbines):
            # Get the distribution type ('low' or 'high') for the turbine
            wind_dist = turbine.get("wind_distribution", "unknown")

            # Choose color and marker based on index
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Plot fatigue damage for each turbine
            line1, = ax.plot(turbine.get('d'), turbine.get('fatigue_damage'), marker=marker, color=color,
                             label=f'{turbine["name"]} - Fatigue Damage')

            # Plot first eigen frequency for each turbine
            line2, = ax2.plot(turbine.get('d'), turbine.get('first_eigen_frequency'), linestyle='--', color=color,
                              label=f'{turbine["name"]} - First Eigen Frequency')

            # Add handles for the legend
            handles.extend([line1, line2])

        # Plot 1P and 3P frequency ranges as shaded regions for all turbines
        ax2.axhspan(turbine.get('f_1P').get('min'), turbine.get('f_1P').get('max'), color='r', alpha=0.3,
                    label='1P Range')
        ax2.axhspan(turbine.get('f_3P').get('min'), turbine.get('f_3P').get('max'), color='g', alpha=0.3,
                    label='3P Range')

        # Add the labels for the 1P and 3P areas
        ax2.text(10, (turbine.get('f_1P').get('min') + turbine.get('f_1P').get('max')) / 2, '1P Range',
                 color='r', fontsize=12, ha='right', va='center')
        ax2.text(10, (turbine.get('f_3P').get('min') + turbine.get('f_3P').get('max')) / 2, '3P Range',
                 color='g', fontsize=12, ha='right', va='center')

        # Add the text annotations for the areas below 1P, between 1P and 3P, and above 3P
        ax2.text(10, turbine.get('f_1P').get('min') / 2, 'Soft-Soft', color='k', fontsize=8, ha='right', va='center', fontweight='bold')
        ax2.text(10, (turbine.get('f_1P').get('max') + turbine.get('f_3P').get('min')) / 2, 'Stiff-Soft', color='k', fontsize=8, ha='right', va='center', fontweight='bold')
        ax2.text(10, (turbine.get('f_3P').get('max') * 1.05), 'Stiff-Stiff', color='k', fontsize=8, ha='right', va='center', fontweight='bold')

        ax2.set_ylim(0, 0.7)

        # Add legend only once
        ax2.legend(handles=handles, loc='upper right')

        # Adding a grid
        ax2.grid(True)

        # Show plot
        plt.title("Fatigue Damage vs. First Eigen Frequency for Multiple Turbines")

        if save is not None:
            plt.savefig(save, dpi=300)

        plt.show()


class ProcessWind():
    def __init__(self, paths, cp_path = None):
        self.paths = paths
        self._load_data(paths)
        if cp_path is not None:
            self._load_cp(path=cp_path)
        else:
            self._load_cp()
        self.assignement_parameters = assignement_parameters()
        self.params = {}
        self.physics = self.assignement_parameters.physics
        self.cp_interpolator = None
        self.turbines = self.assignement_parameters.turbines
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
        fig = plt.figure(figsize=(5, 4))

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
            rf"Max Cp \n($C_p = {max_Cp:.2f}$, TSR = {TSR_max_Cp:.2f})"
            #rf"Max Cp \\ (C_p = {max_Cp:.2f}$, TSR = {TSR_max_Cp:.2f})"
            if latex else
            f"Max Cp \n(Cp = {max_Cp:.2f}, TSR = {TSR_max_Cp:.2f})"
        )
        plt.plot([TSR_max_Cp], [max_Cp], "or", label=max_label)

        # Add labels, title, and legend
        plt.xlabel("TSR")
        plt.ylabel("Cp")
        plt.title("Power Coefficient (Cp) vs Tip-Speed Ratio (TSR)")
        plt.legend(loc="lower left")

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
        plt.figure(figsize=(5, 8))
        sns.boxplot(data=self.df, orient='v', palette="Set2")
        plt.title(title, fontsize=16)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks([i for i in range(len(self.df.keys()))], labels, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save is not None:
            plt.savefig(save, dpi=300)
        plt.show()

    def wind_speed_statistics(self, threshold=12, bins=50):
        """
        Calculates wind speed statistics for numerical columns in the dataset.

        Parameters:
            threshold (float): The wind speed threshold to calculate exceedance probability (default is 12 m/s).
            bins (int): Number of bins for calculating the most frequent wind speed.

        Returns:
            pd.DataFrame: DataFrame with mean, most frequent wind speed, and exceedance probability for each column.
        """
        stats = []

        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:  # Only process numerical columns
                data = self.df[col].dropna()

                # Calculate the most frequent wind speed using a histogram
                hist_counts, bin_edges = np.histogram(data, bins=bins)
                most_frequent_bin_index = np.argmax(hist_counts)
                most_frequent_wind_speed = (bin_edges[most_frequent_bin_index] + bin_edges[
                    most_frequent_bin_index + 1]) / 2

                # Calculate the probability of wind speed exceeding the threshold
                exceedance_prob = (data > threshold).mean()

                # Append statistics for the current column
                stats.append({
                    'Column': col,
                    'Mean Wind Speed (m/s)': data.mean(),
                    'Standard Deviation (m/s)' : data.std(),
                    'Most Frequent Wind Speed (m/s)': most_frequent_wind_speed,
                    f'P(Wind Speed > {threshold} m/s)': exceedance_prob,
                    'Median Wind Speed (m/s)' : data.median()
                })

        # Convert stats to a DataFrame
        stats_df = pd.DataFrame(stats)

        return stats_df

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
        plt.figure(figsize=(10, 5))

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
        plt.legend(loc='upper right', fontsize=12)
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

        # Enforce rotational speed limits
        omega = np.clip(omega, rot_min, rot_max)

        # Set power to 0 outside the cut-in and cut-out range
        omega = np.where((wind_speed < cut_in) | (wind_speed > cut_out), 0, omega)



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

    def plot_turbine_performance(self, wind_distribution_name, turbine_names=None, rectangles=None, save=None,
                                 wind_lim_per=1.05):
        """
        Plots the Power Output, Rotational Speed, and Torque for all turbines for a selected wind distribution.
        Allows shading areas delimited by wind speeds.

        Parameters:
            wind_distribution_name (str): The wind distribution name to filter results.
            turbine_names (dict): Optional dictionary to map turbine IDs to custom names.
                                  Example: {'Turbine1': 'Small Turbine', 'Turbine2': 'Large Turbine'}
            rectangles (list): List of dictionaries defining areas to shade with 'speed_min', 'speed_max', 'color', 'alpha'.
                               Example:
                                   [
                                       {'name': 'Area1', 'speed_min': 5, 'speed_max': 6, 'color': 'r', 'alpha': 0.5},
                                       ...
                                   ]
            save (str): Path to save the plot. If None, the plot is not saved.
            wind_lim_per (float): Percentage multiplier for wind speed axis limit (default: 1.05).
        """
        # Check if results exist
        if self.results is None:
            self.calc_performances()

        # Filter results for the selected wind distribution
        filtered_results = [res for res in self.results if res['wind_distribution_name'] == wind_distribution_name]

        if not filtered_results:
            raise ValueError(f"No results found for wind distribution: {wind_distribution_name}")

        # Define line styles and colors for distinction
        line_styles = ['-', '--', '-.', ':']
        colors = plt.cm.tab10.colors  # Use tab10 colormap for diverse colors

        # Create figure and subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Store handles for the turbine legend and rectangle legend
        legend_handles = []
        rectangle_handles = []

        cut_out = 30

        # Add shaded areas based on rectangles
        if rectangles:
            for rect in rectangles:
                # Add shaded areas to all subplots
                for ax in axes:
                    ax.axvspan(
                        rect['speed_min'], rect['speed_max'], color=rect['color'], alpha=rect['alpha']
                    )
                # Add small rectangles to the legend
                if rect['name'] not in [h.get_label() for h in rectangle_handles]:
                    rectangle_handles.append(
                        plt.Line2D([], [], color=rect['color'], marker='s', markersize=10, linestyle='',
                                   alpha=rect['alpha'], label=rect['name'])
                    )

        # Plot Power Output, Rotational Speed, and Torque
        for i, turbine in enumerate(self.turbines):
            turbine_name = turbine['name']

            # Find turbine-specific results
            turbine_res = next(
                (res for res in filtered_results if res['turbine'] == turbine_name),
                None
            )
            if not turbine_res:
                continue

            # Select line style and color
            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]

            # Use custom turbine names if provided
            display_name = turbine_names.get(turbine_name, turbine_name) if turbine_names else turbine_name

            # Power Output
            axes[0].plot(
                turbine_res['wind_speeds_m/s'], turbine_res['power_W'],
                line_style, color=color, alpha=0.8, label=display_name
            )

            # Rotational Speed
            axes[1].plot(
                turbine_res['wind_speeds_m/s'], self.rads2rotmin(turbine_res['rotational_speed_rad/s']),
                line_style, color=color, alpha=0.8
            )

            # Torque
            axes[2].plot(
                turbine_res['wind_speeds_m/s'], turbine_res['torque_Nm'],
                line_style, color=color, alpha=0.8
            )

            # Add to turbine legend handles
            if display_name not in [h.get_label() for h in legend_handles]:
                legend_handles.append(
                    plt.Line2D([], [], color=color, linestyle=line_style, label=display_name)
                )

            cut_out = turbine['cut_out']

        # Titles and labels
        axes[0].set_title('Power Output', fontsize=12, weight='bold')
        axes[0].set_ylabel('Power (W)', fontsize=12)
        axes[0].grid(linestyle='--', alpha=0.7)

        axes[1].set_title('Rotational Speed', fontsize=12, weight='bold')
        axes[1].set_ylabel('Speed (rpm)', fontsize=12)
        axes[1].grid(linestyle='--', alpha=0.7)

        axes[2].set_title('Torque', fontsize=12, weight='bold')
        axes[2].set_xlabel('Wind Speed (m/s)', fontsize=12)
        axes[2].set_ylabel('Torque (Nm)', fontsize=12)
        axes[2].grid(linestyle='--', alpha=0.7)

        # Set x-axis limits
        max_wind_speed = max(res['wind_speeds_m/s'].max() for res in filtered_results)
        for ax in axes:
            ax.set_xlim(0, cut_out * wind_lim_per)
            ax.set_ylim(0)

        # Combine turbine and rectangle handles for the legend
        all_handles = legend_handles + rectangle_handles

        # Add a single legend under the figure with 4 columns
        fig.legend(
            handles=all_handles, loc='lower center', ncol=4, fontsize=10, title='Legend',
            title_fontsize=12, bbox_to_anchor=(0.5, -0.03)
        )

        # Adjust layout
        #plt.tight_layout(rect=[0, 0.25, 1, 1])  # Leave space for the legend

        # Save plot if save path is provided
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)  # Ensure directory exists
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save}")

        # Show plot
        plt.show()

    def generate_turbine_operation_zones(self, turbine_name, wind_distribution_name, alpha = 0.2):
        """
        Generates control zone rectangles for a specific turbine and wind distribution.

        Parameters:
            turbine_name (str): The name of the turbine.
            wind_distribution_name (str): The name of the wind distribution.

        Returns:
            list: A list of dictionaries defining the control zones.
        """
        # Find turbine data
        turbine = next((t for t in self.turbines if t['name'] == turbine_name), None)
        if not turbine:
            raise ValueError(f"Turbine '{turbine_name}' not found in turbines.")

        # Find results for the specified turbine and wind distribution
        res = next((r for r in self.results if
                    r['turbine'] == turbine_name and r['wind_distribution_name'] == wind_distribution_name), None)
        if not res:
            raise ValueError(
                f"No results found for turbine '{turbine_name}' and wind distribution '{wind_distribution_name}'.")

        # Retrieve turbine parameters
        cut_in = turbine['cut_in']
        cut_out = turbine['cut_out']

        # Calculate transition speeds
        try:
            tc1_end = res['wind_speeds_m/s'][
                np.where(res['rotational_speed_rad/s'] > self.rotmin2rads(turbine['min_rot_speed']))[0][0]]
            tc2_end = res['wind_speeds_m/s'][
                np.where(res['rotational_speed_rad/s'] > (self.rotmin2rads(turbine['max_rot_speed']) * 0.999))[0][0]]
            tc3_end = res['wind_speeds_m/s'][np.where(res['power_W'] / 1e6 > turbine['rated_power'] * 0.999)[0][0]]
        except IndexError as e:
            raise ValueError(f"Could not calculate transition speeds due to missing or incomplete data: {e}")

        # Define rectangles for the control zones
        rec = [
            {
                'name': '1 - Torque Control',
                'speed_min': cut_in,
                'speed_max': tc1_end,
                'color': 'g',
                'alpha': alpha
            },
            {
                'name': '2 - Optimum Torque Control',
                'speed_min': tc1_end,
                'speed_max': tc2_end,
                'color': 'orange',
                'alpha': alpha
            },
            {
                'name': '3 - Torque Control',
                'speed_min': tc2_end,
                'speed_max': tc3_end,
                'color': 'b',
                'alpha': alpha
            },
            {
                'name': '4 - Pitch Control',
                'speed_min': tc3_end,
                'speed_max': cut_out,
                'color': 'purple',
                'alpha': alpha
            }
        ]

        return rec

