import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Any, Dict
from utils.logger import logger

class HyperparameterTuning:
    def tune_hyperparameters(
        self,
        model: Any,
        param_grid: Dict[str, Any],
        X_train,
        y_train,
        search_type: str = "grid",
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Any:
        """
        Perform hyperparameter tuning on the given model.

        Parameters:
        - model : Any
            The model to be tuned.
        - param_grid : Dict[str, Any]
            The hyperparameter grid for tuning.
        - X_train : pd.DataFrame
            Training features.
        - y_train : pd.Series
            Training labels.
        - search_type : str, default="grid"
            The type of search ("grid" for GridSearchCV or "random" for RandomizedSearchCV).
        - cv : int, default=5
            Number of cross-validation folds.
        - scoring : str, default="accuracy"
            Metric for evaluating models.

        Returns:
        - best_model : Any
            The model with the best parameters.
        """
        logger.info(f"Starting hyperparameter tuning with {search_type} search...")

        search = None
        if search_type == "grid":
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
        elif search_type == "random":
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv, scoring=scoring, n_iter=20)
        else:
            logger.error(f"Invalid search type: {search_type}")
            raise ValueError("Search type must be 'grid' or 'random'.")

        search.fit(X_train, y_train)
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_}")

        best_model = search.best_estimator_
        return best_model
