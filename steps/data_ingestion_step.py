from src.data_ingestion import DataIngestion
import pandas as pd
from utils.logger import logger
from prefect import task


@task(name= "Data Ingestion Task",log_prints=True)
def data_ingestion_step(data_path:str)->pd.DataFrame:
    try:
        logger.info("Starting data ingestion step")
        data_ingestion = DataIngestion()
        df = data_ingestion.ingest_data(data_path)
        logger.info("Data ingestion successful step ")
        return df
    except Exception as e:
        logger.error(f"Error in data ingestion {e}")
        raise e