import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
from typing import Annotated
from utils.logger import logger


class DataResampling:


    def resample_data(self, method: str,X_train:pd.DataFrame,y_train:pd.Series) ->Annotated[pd.DataFrame, pd.Series]:
        """
        Resamples the data using the specified method.

        Parameters:
        
        method : str
            The resampling method to use. Options are 'smote', 'random_under', and 'smoteenn'.

        Returns:
        
        pd.DataFrame
            The resampled data.
        """
       

        if method == "smote":
            resampler = SMOTE(random_state=42,k_neighbors=5)
        elif method == "random_under":
            resampler = RandomUnderSampler(random_state=42)
        elif method == "smoteenn":
            resampler = SMOTEENN(random_state=42,n_jobs=-1)
        else:
            raise ValueError("Invalid resampling method. Choose from 'smote', 'random_under', and 'smoteenn'.")

        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        # resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
        logger.info(f"Resampled data shape: {Counter(y_resampled)}")

        return X_resampled, y_resampled