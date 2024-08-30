import pandas as pd
import numpy as np
from typing import Callable

def convert_datetime(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Convert the specified feature to a datetime object and return modified dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing the feature to be converted.
        feature (str): The name of the feature to be converted.

    Returns:
        pd.DataFrame: The DataFrame with the specified feature converted to datetime.
    """

    df1 = df.copy()
    df1[feature] = pd.to_datetime(df[feature], format = "'%Y-%m-%dT%H:%M:%S'")

    return df1

def create_month(df: pd.DataFrame, feature: str, name: str) -> pd.DataFrame:
    """
    Convert a "month" feature from the specified datetime feature and return modified dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing the datetime feature.
        feature (str): The name of the datetime feature.
        name (str): The name of new "month" feature to be created.

    Returns:
        pd.DataFrame: The DataFrame including the "month" feature.
    """

    #df1 = df.copy()
    #df1[feature] = pd.to_datetime(df[feature], format = "'%Y-%m-%dT%H:%M:%S'")

    df[name] = df[feature].dt.month

    return df

def aggregate_months(df: pd.DataFrame, group: str, agg_func: str) -> pd.DataFrame:
    """
    Perform aggregation function on dataframe based on grouping feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing features to be aggregated.
        group (str): The name of the feature by which to group.
        agg_func (str): The name of the aggregating function.

    Returns:
        pd.DataFrame: The aggregated dataframe.
    """

    df1 = df.copy()

    #df1['Timestamp'] = pd.to_datetime(df['Timestamp'], format = "'%Y-%m-%dT%H:%M:%S'")

    #df1[group] = df1['Timestamp'].dt.month

    move_group = df1.pop(group)
    df1.insert(0, group, move_group)

    df1 = df1.groupby(group).agg(func = agg_func)

    return df1

    #.agg
    
