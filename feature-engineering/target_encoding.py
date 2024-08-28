from category_encoders import TargetEncoder
import numpy as np
import pandas as pd


def discretize_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Discretizes a numerical feature in the DataFrame by rounding it to the nearest integer.

    Args:
        df (pd.DataFrame): The input DataFrame containing the feature to be discretized.
        feature (str): The name of the feature to be discretized.

    Returns:
        pd.DataFrame: The DataFrame with the specified feature discretized to integers.
    """

    df1 = df.copy()
    df1[feature] = np.round(df1[feature]) 
    return df1


def target_encode(
    df: pd.DataFrame, features_to_encode: list[str], target_col: str
) -> pd.DataFrame:
    """
    Encodes categorical features in the DataFrame using target encoding based on the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be encoded.
        features_to_encode (list(str)): List of feature names to be target encoded.
        target_col (str): The name of the target column used for encoding.

    Returns:
        pd.DataFrame: The DataFrame with specified features target encoded.
    """

    df1 = df.copy()
    t = TargetEncoder(cols = features_to_encode, return_df = True) # Returns an object of the class TargetEncoder
    df1[features_to_encode] = t.fit_transform(df[features_to_encode], df[target_col])
    return df1

    # In df1, replace features_to_encode with what is returned from fit_transform
    # 2, 3 fit and transform
    # 
    # fit and then transform, where we pass target_col
    # x and y go into fit fit(x,y)

    
