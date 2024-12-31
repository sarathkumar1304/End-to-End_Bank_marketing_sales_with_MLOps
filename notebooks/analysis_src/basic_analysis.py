import pandas as pd
import numpy as np


class BasicDataInspection:
    def inspect_data(self,df:pd.DataFrame):
        basic_info = df.info()
        return basic_info
    
    def check_null_values(self,df:pd.DataFrame):
        null_values = df.isnull().sum()
        return null_values
    
    def check_duplicate_values(self,df:pd.DataFrame):
        duplicate_values = df.duplicated().sum()
        return duplicate_values
    
    def numerical_statistical_summary(self,df:pd.DataFrame):
        summary = df.describe(include=[int,float])
        return summary
    
    def categorical_statistical_summary(self,df:pd.DataFrame):
        summary = df.describe(include=[object])
        return summary
