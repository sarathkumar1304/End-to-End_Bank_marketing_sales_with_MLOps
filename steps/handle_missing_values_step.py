import pandas as pd
from utils.logger import logger
from prefect import flow,task 
from src.data_preprocessing import DataPreProcessing
from typing import List


@task(name ="Handle Missing Values Task",log_prints=True)
def data_preprocessing_task(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Initiating Data Preprocessing")
    preprocessing = DataPreProcessing()

    # Drop specific column
    preprocessing.drop_column(df, "poutcome")
    logger.info("Dropped column: poutcome")

    # Extract categorical columns that contain only null values
    null_categorical_columns = [col for col in df.select_dtypes(include=['O']).columns if df[col].isnull().sum()>0]
    logger.info("Categorical columns with only null values: {}".format(null_categorical_columns))

    # Fill missing values for categorical columns
    df = preprocessing.fill_missing_values(df, null_categorical_columns, method="mode")


    logger.info("Data Preprocessing Completed for columns: {}".format(null_categorical_columns))
    return df

    
   