import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BivariateAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the BivariateAnalysis class with a DataFrame.
        """
        self.df = df

    def scatter_plot(self, x: str, y: str, hue: str = None):
        """
        Scatter plot for two numerical variables.
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        :param hue: (Optional) Column name for color encoding (categorical variable).
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, palette='coolwarm', alpha=0.7)
        plt.title(f'Scatter Plot: {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def bar_plot(self, x: str, y: str, hue: str = None, aggfunc: str = 'mean'):
        """
        Bar plot for relationships between categorical and numerical variables.
        :param x: Column name for the categorical variable.
        :param y: Column name for the numerical variable.
        :param hue: (Optional) Column name for color grouping.
        :param aggfunc: Aggregation function ('mean', 'sum', etc.). Default is 'mean'.
        """
        grouped_data = self.df.groupby([x] + ([hue] if hue else []))[y].agg(aggfunc).reset_index()
        sns.barplot(data=grouped_data, x=x, y=y, hue=hue, palette='pastel')
        plt.title(f'{aggfunc.capitalize()} of {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(f'{aggfunc.capitalize()} of {y}')
        plt.xticks(rotation=45)
        plt.show()

    def box_plot(self, x: str, y: str):
        """
        Box plot for comparing numerical variable distributions across categories.
        :param x: Column name for the categorical variable.
        :param y: Column name for the numerical variable.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x=x, y=y, palette='Set3')
        plt.title(f'Box Plot: {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.show()

    def violin_plot(self, x: str, y: str):
        """
        Violin plot for showing distribution and density of numerical variables across categories.
        :param x: Column name for the categorical variable.
        :param y: Column name for the numerical variable.
        """
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.df, x=x, y=y, palette='muted')
        plt.title(f'Violin Plot: {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.show()

    def heatmap(self):
        """
        Heatmap for visualizing correlations between numerical variables.
        """
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        corr = self.df[numerical_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()


