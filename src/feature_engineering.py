import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from abc import ABC, abstractmethod
from utils.logger import logger

# Abstract Base Class for Feature Engineering Strategies
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            if (df[feature] <= 0).any():
                raise ValueError(f"Log transformation requires all values in '{feature}' to be positive.")
            df[feature] = np.log(df[feature])
        return df

# Standard Scaling
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.features] = self.scaler.fit_transform(df[self.features])
        return df

# Min-Max Scaling
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = MinMaxScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.features] = self.scaler.fit_transform(df[self.features])
        return df

# One-Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying One-Hot Encoding for Categorical Features : {}".format(self.features))
        for feature in self.features:
            encoded = pd.DataFrame(self.encoder.fit_transform(df[[feature]]), 
                                   columns=self.encoder.get_feature_names_out([feature]))
            df = pd.concat([df.drop(columns=[feature]), encoded], axis=1)
        logger.info("One-Hot Encoding Complete")
        return df

# Label Encoding
class LabelEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoders = {feature: LabelEncoder() for feature in features}

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying Label Encoding for Categorical Features : {}".format(self.features))
        for feature in self.features:
            df[feature] = self.encoders[feature].fit_transform(df[feature])
        logger.info("Label Encoding Complete")
        return df

# Feature Engineering Pipeline
class FeatureEngineering:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self,strategy :FeatureEngineeringStrategy):
        logger.info("Switching feature engineering strategy")
        self._strategy = strategy
    def apply_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        logger.info("Applying feature Engineering")
        return self._strategy.apply_transformation(df)

    
        

# Example Usage
if __name__ == "__main__":
    # data = "data/bank_marketing"
    # df = pd.DataFrame(data)

    # pipeline = FeatureEngineering()
    # pipeline.add_step(LogTransformation(features=['A']))
    # pipeline.add_step(MinMaxScaling(features=['B']))
    # pipeline.add_step(OneHotEncoding(features=['C']))

    # transformed_df = pipeline.apply(df)
    # print(transformed_df)
    pass
