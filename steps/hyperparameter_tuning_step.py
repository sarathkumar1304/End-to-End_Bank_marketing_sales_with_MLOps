from prefect import task
from src.hyperparameter_tuning import HyperparameterTuning
from src.model_building import ModelBuilding
from utils.logger import logger
import os 
import joblib
@task(name="Hyperparameter Tuning Task", log_prints=True)
def hyperparameter_tuning_task(model_name: str, X_train, y_train):
    """
    Task for hyperparameter tuning.

    Parameters:
    - model_name : str
        Name of the model to tune.
    - X_train : pd.DataFrame
        Training features.
    - y_train : pd.Series
        Training labels.

    Returns:
    - best_model : Any
        The model with tuned hyperparameters.
    """
    logger.info("Starting hyperparameter tuning task...")
    tuner = HyperparameterTuning()
    model_builder = ModelBuilding()

    # Define hyperparameter grids for different models
    param_grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["liblinear", "lbfgs"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "decision_tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "xgboost": {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
        },
    }

    if model_name not in param_grids:
        logger.error(f"No hyperparameter grid defined for model '{model_name}'.")
        raise ValueError(f"No hyperparameter grid for '{model_name}'.")

    # Build the base model
    base_model = model_builder.get_model(model_name, X_train, y_train)

    # Perform hyperparameter tuning
    best_model = tuner.tune_hyperparameters(
        model=base_model,
        param_grid=param_grids[model_name],
        X_train=X_train,
        y_train=y_train,
        search_type="grid",
    )

    logger.info(f"Hyperparameter tuning completed for {model_name}.")
    model_dir = "Artifact"
    os.makedirs(model_dir, exist_ok=True)  # Ensure the models directory exists
    model_path = os.path.join(model_dir, "tuned_model.pkl")
    joblib.dump(best_model, model_path)  # Save model pipeline as 'model.pkl'
    logger.info(f"Model saved at {model_path}")
    logger.info("Model building step completed successfully.")
    return best_model
