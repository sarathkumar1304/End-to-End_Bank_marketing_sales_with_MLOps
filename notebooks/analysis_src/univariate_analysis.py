# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# class UnivariateAnalysis:
#     def NumericalAnalysis(self,df:pd.DataFrame,column:str, sqr:bool= False,log:bool=False):
#         sns.histplot((df[column]), bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.title(f'{column} Distribution')
#         plt.show()
#         if sqr:
#             sns.histplot(np.sqrt(df[column]), bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
#             plt.xlabel(column)
#             plt.ylabel('Frequency')
#             plt.title(f'Squared {column} Distribution')
#             plt.show()
#         if log:
#             sns.histplot(np.sqrt(df[column]), bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
#             plt.xlabel(column)
#             plt.ylabel('Frequency')
#             plt.title(f'Log_{column} Distribution')
#             plt.show()
#     def CategoricalAnalysis(self,df: pd.DataFrame, column: str):
#         value_counts = df[column].value_counts()
#         explode = [0.1] * len(value_counts)  # Create an explode list of the same length as the number of categories
#         colors = plt.cm.Paired.colors[:len(value_counts)]  # Use a colormap to generate enough colors

#         plt.figure(figsize=(8, 6))
#         plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
#                 explode=explode, shadow=True, startangle=140, colors=colors)
#         plt.title(f'Distribution of {column}', fontsize=14)
#         plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
#         plt.show()

#     def box_plot(self,df:pd.DataFrame,column:str):
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x=df[column], palette="Set1")
#         plt.title(f'Box Plot of {column}')
#         plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class UnivariateAnalysis:
    def NumericalAnalysis(self, df: pd.DataFrame, column: str, sqr: bool = False, log: bool = False):
        if df[column].isnull().any():
            print(f"Missing values detected in {column}. Please handle them before analysis.")
            return

        # Original distribution
        sns.histplot(df[column], bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'{column} Distribution')
        plt.show()
        
        # Violin plot
        sns.violinplot(x=df[column], palette="muted")
        plt.title(f'Violin Plot of {column}')
        plt.show()
        
        if sqr:
            sns.histplot(np.sqrt(df[column]), bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
            plt.xlabel(f'Square Root of {column}')
            plt.ylabel('Frequency')
            plt.title(f'Square Root {column} Distribution')
            plt.show()
        
        if log:
            sns.histplot(np.log1p(df[column]), bins=20, kde=True, color='c', edgecolor='black', linewidth=1.2)
            plt.xlabel(f'Log of {column}')
            plt.ylabel('Frequency')
            plt.title(f'Log {column} Distribution')
            plt.show()
        
        # Summary statistics
        print(f"Summary statistics for {column}:\n{df[column].describe()}")
    
    def CategoricalAnalysis(self, df: pd.DataFrame, column: str):
        value_counts = df[column].value_counts()
        
        # Pie chart
        explode = [0.1] * len(value_counts)
        colors = plt.cm.Paired.colors[:len(value_counts)]
        
        plt.figure(figsize=(8, 6))
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                explode=explode, shadow=True, startangle=140, colors=colors)
        plt.title(f'Distribution of {column}', fontsize=14)
        plt.axis('equal')
        plt.show()
        
        # Count plot
        sns.countplot(x=df[column], palette="pastel")
        plt.title(f'Count Plot of {column}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()
    
    def box_plot(self, df: pd.DataFrame, column: str):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column], palette="Set1")
        plt.title(f'Box Plot of {column}')
        plt.show()
