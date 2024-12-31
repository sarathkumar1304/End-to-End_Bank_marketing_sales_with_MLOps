import pandas as pd
from utils.logger import logger

class DataIngestion:
    """
    Handles the ingestion of data from a specified file path.
    """
    def ingest_data(self, data_path: str) -> pd.DataFrame:
        """
        Reads a CSV file from the given path and returns a DataFrame.

        Args:
            data_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the ingested data.

        Raises:
            Exception: If an error occurs during data ingestion.
        """
        try:
            logger.info("Starting data ingestion process.")
            df = pd.read_csv(data_path)
            logger.info(f"Data ingestion successful. Shape of the DataFrame: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {data_path}. Ensure the path is correct.")
            raise e
        except pd.errors.EmptyDataError as e:
            logger.error("The file is empty. Please provide a valid CSV file.")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during data ingestion: {e}")
            raise e
