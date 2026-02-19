import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
import seaborn as sns
sns.set_theme()
from utils_m import simple_log, vectorized_exclude_func
from tqdm import tqdm

def main():
    
    tqdm.pandas()

    # Added this progress bar for my own sanity
    progress_bar = tqdm(total=20)

    """ Account for missing values """

    simple_log("Setting up df")

    # load the dataset
    df = pd.read_csv ('paysim.csv')
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})

    progress_bar.update(1)

    # if oldBalance = newBalance = 0, account is likely from external institution
    df['externalDest'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)
    df['externalOrig'] = ((df['oldbalanceOrig'] == 0) & (df['newbalanceOrig'] == 0)).astype(int)

    progress_bar.update(1)

    # Update the values in the 'newbalanceDest' column to 'oldbalanceDest +- amount'
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    df['oldbalanceOrig'] = df['newbalanceOrig'] + df['amount']

    progress_bar.update(1)


    """ Feature Engineering """

    simple_log("Feature engineering")

    # gaussian noise std
    num_transactions = df.shape[0]
    std = 0.01*(df['amount'].quantile(0.75) - df['amount'].min())

    progress_bar.update(1)

    # calculate the overall mean of both the destination and origin account (excluding current transaction) and add noise
    noise = np.random.normal(0, std, num_transactions)
    df['num_transDest'] = df.groupby('nameDest')['nameDest'].transform('count')
    df['meanDest'] = df.groupby('nameDest')['amount'].transform('mean')
    df['meanDest'] = df.progress_apply(exclude_current_meanDest, axis=1)
    df['meanDest'] += noise

    progress_bar.update(1)

    noise = np.random.normal(0, std, num_transactions)
    df['num_transOrig'] = df.groupby('nameOrig')['nameOrig'].transform('count')
    df['meanOrig'] = df.groupby('nameOrig')['amount'].transform('mean')
    df['meanOrig'] = df.progress_apply(exclude_current_meanOrig, axis=1)
    df['meanOrig'] += noise

    progress_bar.update(1)

    # swap places for consistency
    num_transDest = df.pop('num_transDest')
    df['num_transDest'] = num_transDest
    num_transOrig = df.pop('num_transOrig')
    df['num_transOrig'] = num_transOrig

    progress_bar.update(1)

    # calculate the overall max of both the destination and origin account (excluding current transaction) and add noise

    df = vectorized_exclude_func(df, "maxDest", std, num_transactions)

    progress_bar.update(1)

    df = vectorized_exclude_func(df, "maxOrig", std, num_transactions)

    progress_bar.update(1)

    # sort the values according to timestep
    df = df.sort_values('step')

    # calculate the rolling average of last 3 and 7 transactions for each recipient
    df['meanDest3'] = df.groupby('nameDest')['amount'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['meanDest7'] = df.groupby('nameDest')['amount'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)

    progress_bar.update(1)

    # calculate the rolling maximum of last 3 and 7 transactions for each recipient
    df['maxDest3'] = df.groupby('nameDest')['amount'].rolling(window=3, min_periods=1).max().reset_index(0, drop=True)
    df['maxDest7'] = df.groupby('nameDest')['amount'].rolling(window=7, min_periods=1).max().reset_index(0, drop=True)

    progress_bar.update(1)

    # rearrange column order
    df = df.reindex(columns=[col for col in df.columns if col not in ['num_transDest', 'num_transOrig', 'externalDest', 'externalOrig', 'isFraud']] + ['num_transDest', 'num_transOrig', 'externalDest', 'externalOrig', 'isFraud'])

    progress_bar.update(1)

    """ Prepare the training and test sets """

    simple_log("Preparing data sets")

    # delete name columns that will not be used
    df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

    # split the dataset into training and validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
    val_set_size = 0.10/0.25
    test_df, val_df = train_test_split(test_df, test_size=val_set_size, random_state=42, shuffle=True)

    # reset index
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # remove non numerical columns of training set and save for later
    type_col = train_df.pop('type')
    num_transDest_col = train_df.pop('num_transDest')
    num_transOrig_col = train_df.pop('num_transOrig')
    externalDest_col = train_df.pop('externalDest')
    externalOrig_col = train_df.pop('externalOrig')
    isFraud_col = train_df.pop('isFraud')

    progress_bar.update(1)

    # fit the scaler on the train data and transform
    scaler = StandardScaler()
    scaler = scaler.fit(train_df)
    train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)

    progress_bar.update(1)

    # add the non numerical columns back
    train_df = pd.concat([train_df, type_col, num_transDest_col, num_transOrig_col, externalDest_col, externalOrig_col, isFraud_col], axis=1)

    progress_bar.update(1)

    # do the same on the test set, use the scaler fit on the train set
    type_col = test_df.pop('type')
    num_transDest_col = test_df.pop('num_transDest')
    num_transOrig_col = test_df.pop('num_transOrig')
    externalDest_col = test_df.pop('externalDest')
    externalOrig_col = test_df.pop('externalOrig')
    isFraud_col = test_df.pop('isFraud')
    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    test_df = pd.concat([test_df, type_col, num_transDest_col, num_transOrig_col, externalDest_col, externalOrig_col, isFraud_col], axis=1)

    progress_bar.update(1)

    # do the same on the val set, use the scaler fit on the train set
    type_col = val_df.pop('type')
    num_transDest_col = val_df.pop('num_transDest')
    num_transOrig_col = val_df.pop('num_transOrig')
    externalDest_col = val_df.pop('externalDest')
    externalOrig_col = val_df.pop('externalOrig')
    isFraud_col = val_df.pop('isFraud')
    val_df = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    val_df = pd.concat([val_df, type_col, num_transDest_col, num_transOrig_col, externalDest_col, externalOrig_col, isFraud_col], axis=1)

    progress_bar.update(1)

    # one-hot encode the 'type' column
    train_df = pd.get_dummies(train_df, columns=['type'])
    test_df = pd.get_dummies(test_df, columns=['type'])
    val_df = pd.get_dummies(val_df, columns=['type'])

    progress_bar.update(1)

    # move the 'isFraud' column to the end of the dataframe
    is_fraud_col = train_df.pop('isFraud')
    train_df['isFraud'] = is_fraud_col

    is_fraud_col = test_df.pop('isFraud')
    test_df['isFraud'] = is_fraud_col

    is_fraud_col = val_df.pop('isFraud')
    val_df['isFraud'] = is_fraud_col

    progress_bar.update(1)

    # save the train, validation and test set
    train_df.to_csv("./code/deep-symbolic-optimization-pytorch/dso/dso/task/regression/data/train_df.csv",header=False, index=False)
    test_df.to_csv("./code/deep-symbolic-optimization-pytorch/dso/dso/task/regression/data/test_df.csv",header=False, index=False)
    val_df.to_csv("./code/deep-symbolic-optimization-pytorch/dso/dso/task/regression/data/val_df.csv",header=False, index=False)

    progress_bar.update(1)

if __name__ == '__main__':
    main()