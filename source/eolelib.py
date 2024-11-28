import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, weibull_min
import numpy as np

class ProcessWind():
    def __init__(self, paths):
        self.paths = paths
        self._load_data(paths)
        self.params = {}


    def _load_data(self, paths):
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

        # Display the plot
        plt.show()

    def clean_data(self):
        for col in self.df.keys():
            if self.df[col].dtype in ['float64', 'int64']:  # Only apply to numerical columns
                # Create a new column for Z-scores
                self.df.loc[:, 'Z_score'] = zscore(self.df[col], nan_policy='omit')

                # Filter rows using Z-score threshold
                self.df = self.df.loc[(self.df['Z_score'].abs() <= 3) | (self.df[col].isnull())].copy()

                # Drop the temporary Z-score column
                self.df.drop(columns=['Z_score'], inplace=True)

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
                    'shape': shape,  # k
                    'scale': scale,  # λ
                    'location': loc  # Should be 0 as floc=0
                }

        return self.params
