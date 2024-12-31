import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MultivariateAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the MultivariateAnalysis class with a DataFrame.
        """
        self.df = df

    def scatter_plot_matrix(self, hue: str = None):
        """
        Scatter plot matrix to visualize pairwise relationships between numerical variables.
        :param hue: Column name for color encoding (categorical variable).
        """
        numerical_columns = self.df.select_dtypes(include=['int64']).columns
        sns.pairplot(self.df[numerical_columns], hue=hue, diag_kind='kde', palette='Set2')
        plt.suptitle('Scatter Plot Matrix', y=1.02)
        plt.show()

    def correlation_heatmap(self):
        """
        Heatmap to visualize the correlation between numerical variables.
        """
        numerical_columns = self.df.select_dtypes(include=['int64']).columns
        corr = self.df[numerical_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

    def boxplot_by_category(self, categorical: str, numerical: str):
        """
        Boxplot to compare the distribution of a numerical variable across categories.
        :param categorical: Column name for the categorical variable.
        :param numerical: Column name for the numerical variable.
        """
        sns.boxplot(data=self.df, x=categorical, y=numerical, palette='Set3')
        plt.title(f'Boxplot of {numerical} by {categorical}')
        plt.xlabel(categorical)
        plt.ylabel(numerical)
        plt.xticks(rotation=45)
        plt.show()

    def grouped_bar_plot(self, categorical: str, numerical: str, aggfunc: str = 'mean'):
        """
        Grouped bar plot to analyze the relationship between a categorical and numerical variable.
        :param categorical: Column name for the categorical variable.
        :param numerical: Column name for the numerical variable.
        :param aggfunc: Aggregation function ('mean', 'sum', etc.). Default is 'mean'.
        """
        grouped_data = self.df.groupby(categorical)[numerical].agg(aggfunc).reset_index()
        sns.barplot(data=grouped_data, x=categorical, y=numerical, palette='pastel')
        plt.title(f'{aggfunc.capitalize()} of {numerical} by {categorical}')
        plt.xlabel(categorical)
        plt.ylabel(f'{aggfunc.capitalize()} of {numerical}')
        plt.xticks(rotation=45)
        plt.show()

    def bubble_plot(self, x: str, y: str, size: str, hue: str = None):
        """
        Bubble plot to visualize three variables.
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        :param size: Column name for bubble size.
        :param hue: (Optional) Column name for color encoding.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=x, y=y, size=size, hue=hue, sizes=(10, 100), alpha=0.6, palette='viridis',
            legend='full', edgecolor='black')
        plt.title(f'Bubble Plot: {x} vs {y} with size {size}')
        plt.xlabel(x)
        plt.ylabel(y)
        
        plt.show()

