import pandas as pd
from typing import List
from utils.logger import logger


class DataPreProcessing:
    def drop_column(self,df:pd.DataFrame,column:str):
        """
        Drops the specified column from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to be dropped.

        Returns:
            pd.DataFrame: DataFrame with the specified column dropped.
        """
        try:
            logger.info(f"Dropping column: {column}")
            df.drop(column, axis=1, inplace=True)
            logger.info(f"Column {column} dropped successfully.")
            return df
        except KeyError as e:
            logger.error(f"Column {column} not found in the DataFrame.")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during column drop: {e}")
            raise e
        
    def fill_missing_values(self,df:pd.DataFrame,column:List[str],method:str="mean")->pd.DataFrame:
        """
        Fills missing values in the DataFrame using the specified method.

        Args:
            df (pd.DataFrame): Input DataFrame.
            method (str): Method to fill missing values. Options: 'mean', 'median', 'mode', 'custom'.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.
        """
        try:
            logger.info(f"Filling missing values with method: {method}")
            if method == "mean":
                for col in column:
                    df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                for col in column:
                    df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                for col in column:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                logger.error(f"Unknown method: {method}. Using default method: 'mean'")
                df.fillna(df.mean(), inplace=True)
            logger.info("Missing values filled successfully.")
            return df
        except Exception as e:
            logger.error(f"Unexpected error during missing value fill: {e}")
            raise e
        
