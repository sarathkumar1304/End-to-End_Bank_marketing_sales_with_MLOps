import pandas as pd
from utils .logger import logger
from prefect import task
from src.outlier_detection import OutlierDetector

@task(name="Outlier Detection Task",log_prints=True)
def outlier_detection_task(data: pd.DataFrame) -> pd.DataFrame:

    detector = OutlierDetector(data)
    cleaned_data = detector.run_outlier_detection()
    return cleaned_data

