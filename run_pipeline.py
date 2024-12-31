from utils.logger import logger
from training_pipeline import training_pipeline
from mlflow import get_tracking_uri
from prefect import flow


@flow(name="run_pipeline", log_prints=True)
def run():
    df = training_pipeline()
    print(
        "Now run \n"
        f"mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To view the results in the MLflow UI"
        "You can also view the results in the MLflow UI by clicking on the link above."
    )
    return df   

if __name__ == "__main__":
    run()