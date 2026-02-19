"""
Additional utils used by Michael, used to avoid cluttering SV utils.
"""

from loguru import logger
import sys
from pandas import DataFrame
import pandas as pd
import numpy as np

def vectorized_exclude_func(df: DataFrame, excl_type: str, std: int, num_transactions:int) -> DataFrame:
    """
    Function that calculates or mean of dest account excluding the current transaction in a vectorized manner.
    Warning: This function can not be used with df.apply() and requires the entire dataframe to be passed in.
    df: DataFrame containing the transaction data.
    excl_type: Type of max with exlcusion, either "maxDest" or "maxOrig".
    std: Standard deviation of the noise to be added.
    num_transactions: Number of transactions in the dataframe.

    returns: DataFrame with added column for max with exclusion.
    """

    type_dict = {
        "maxDest": {
            "group_col": "nameDest",
            "transaction_col": "num_transDest"
        },
        "maxOrig": {
            "group_col": "nameOrig",
            "transaction_col": "num_transOrig"
        }
    }

    group_col = type_dict[excl_type]['group_col']
    transaction_col = type_dict[excl_type]['transaction_col']

    df_copy = df.copy()
    noise = np.random.normal(0, std, num_transactions)
    max_vals = df_copy.groupby(group_col)['amount'].transform('max')
    second_max = df_copy.groupby(group_col)['amount'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else x.iloc[0])

    df_copy[excl_type] = np.where(
        (max_vals == df_copy['amount']) & (df_copy[transaction_col] > 1),
        second_max,
        max_vals
    )

    df_copy['maxDest'] += noise

    return df_copy



def simple_log(msg:str, log_type:str = "info") -> None:
    """
    Create simple log util
    
    :param log_type: log type e.g. info
    :type log_type: str
    :param msg: message to be logged
    :type msg: str
    """
    match log_type:
        case "info":
            logger.info(msg)
        case _:
            raise ValueError("Invalid log_type")