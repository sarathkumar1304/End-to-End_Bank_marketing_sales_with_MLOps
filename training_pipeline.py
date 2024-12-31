from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import data_preprocessing_task
from steps.outlier_detection_step import outlier_detection_task
from steps.feature_engineering_step import  feature_engineer
from steps.data_splitting_step import data_splitter_task
from steps.model_building_step import model_builder_task    
from steps.model_evaluation_step import model_evaluation_task
from steps.data_resampling_step import data_resampling_task
from steps.hyperparameter_tuning_step import hyperparameter_tuning_task 
from mlflow import get_tracking_uri

from prefect import flow

@flow(name="Training_Pipeline",log_prints=True)
def training_pipeline():
    data_path = "data/bank_marketing.csv"
    df = data_ingestion_step(data_path)
    df = data_preprocessing_task(df)
    # df.to_csv("data/processed_data.csv",index=False)
    processed_data =  feature_engineer(df)
    cleaned_data = outlier_detection_task(processed_data)
    X_train, X_test, y_train, y_test = data_splitter_task(cleaned_data, target_column="y")
    X_resample, y_resample = data_resampling_task(method="smote", X_train=X_train, y_train=y_train)
    model = model_builder_task(model_name="random_forest", X_train=X_resample, y_train=y_resample)
    # tuned_model = hyperparameter_tuning_task(model_name="random_forest", X_train=X_resample, y_train=y_resample)

    # Evaluate the tuned model
    metrics = model_evaluation_task(model, X_test=X_test, y_test=y_test)

    return model

if __name__ == "__main__":
    training_pipeline()
    print(
        "Now run \n"
        f"mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To view the results in the MLflow UI"
        "You can also view the results in the MLflow UI by clicking on the link above."
    )