import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import *


np.random.seed(0)

def set_label(x):
    if x == "negative":
        return 0.0
    elif x == "positive":
        return 1.0
    else:
        return 2

def preprocess_covidx():
    # we added the separation and header because the data is not organized, one can run without sep and header
    # to understand the difference
    raw_train_df = pd.read_csv(COVIDX_ORIGINAL_TRAIN_TXT, sep=" ", header=None)
    raw_val_df   = pd.read_csv(COVIDX_ORIGINAL_VALID_TXT, sep=" ", header=None)
    raw_test_df  = pd.read_csv(COVIDX_ORIGINAL_TEST_TXT, sep=" ", header=None)

    # Columns are added because it was seen that column names were 0,1,2,3, so new column names are added
    # which are given in descriptions
    raw_train_df.columns = ['patient id', 'filename', 'class', 'data source']
    # Since we are doing image classification, patient id and data source is of no importance to us, so
    # we cn drop them
    raw_train_df = raw_train_df.drop(['patient id', 'data source'], axis=1)
    raw_train_df["labels"] = raw_train_df["class"].apply(set_label)
    # raw_train_df["labels"] = (raw_train_df["class"] ==
    #                           "positive").astype(np.float32)
    
    raw_val_df.columns = ['patient id', 'filename', 'class', 'data source']
    # Since we are doing image classification, patient id and data source is of no importance to us, so
    # we cn drop them
    raw_val_df = raw_val_df.drop(['patient id', 'data source'], axis=1)
    raw_val_df["labels"] = raw_val_df["class"].apply(set_label)
    
    raw_test_df.columns = ['patient id', 'filename', 'class', 'data source']
    # Since we are doing image classification, patient id and data source is of no importance to us, so
    # we cn drop them
    raw_test_df = raw_test_df.drop(['patient id', 'data source'], axis=1)
    raw_test_df["labels"] = raw_test_df["class"].apply(set_label)
    
    train_df = raw_train_df
    valid_df = raw_val_df
    test_df  = raw_test_df
    # split into train and test
#     train_df, valid_df = train_test_split(raw_train_df, train_size=0.8, random_state=0)
    train_df.to_csv(COVIDX_TRAIN_CSV, index=False)
    
    valid_df.to_csv(COVIDX_VALID_CSV, index=False)

    # test_df["labels"] = (test_df["class"] ==
    #                           "positive").astype(np.float32)
    test_df.to_csv(COVIDX_TEST_CSV, index=False)

    print(train_df.shape)
    print(valid_df.shape)
    print(test_df.shape)


if __name__ == "__main__":
    preprocess_covidx()