import os
import pandas as pd
import logging
from ucimlrepo import fetch_ucirepo

def fetch_and_store_data(dataset_id, folder_name="data", file_name="dataset.csv"):
    """
    Fetches a dataset from the UCI ML repository, stores it as a CSV file in the specified folder.

    Parameters:
        dataset_id (int): ID of the dataset in the UCI repository.
        folder_name (str): Name of the folder to save the CSV file. Defaults to 'data'.
        file_name (str): Name of the CSV file to save. Defaults to 'dataset.csv'.

    Returns:
        str: Path to the saved CSV file.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Fetch dataset
        logging.info(f"Fetching dataset with ID {dataset_id} from UCI repository...")
        dataset = fetch_ucirepo(id=dataset_id)

        # Data (as pandas DataFrames)
        X = dataset.data.features
        y = dataset.data.targets

        # Combine features and target into a single DataFrame
        df = pd.concat([X, y], axis=1)


        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            logging.info(f"Created directory: {folder_name}")

        # Save the DataFrame to a CSV file
        file_path = os.path.join(folder_name, file_name)
        df.to_csv(file_path, index=False)

        logging.info(f"Dataset saved to: {file_path}")

        return file_path

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

# Example usage
if __name__ == "__main__":
    dataset_id = 222  # Bank Marketing dataset ID
    csv_path = fetch_and_store_data(dataset_id, folder_name="data", file_name="bank_marketing.csv")
    print(f"Dataset successfully saved at: {csv_path}")