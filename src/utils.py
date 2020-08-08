import os
import time
from tqdm import tqdm
import copy
import gc
import argparse
import config
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score


def preprocess_ohe(df, train_mode=True):
    
    if train_mode:
        x = df.drop(["label", "kfold"], axis=1)
        #y = df["label"]
    else:
        x = df

    cols_rm = config.COLS_RM
    cols_num = [col for col in x.select_dtypes(include='number').columns if col not in cols_rm]
    cols_cat = [col for col in x.select_dtypes(include='object').columns if col not in cols_rm]

    # 欠損値補完
    x[cols_num] = x[cols_num].fillna(x[cols_num].mean())
    x[cols_cat] = x[cols_cat].fillna(x[cols_cat].mode())

    # カテゴリカルデータを、One-hot化
    x = pd.get_dummies(x, drop_first=True, columns=cols_cat)
    
    # 標準化
    if train_mode:
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

        os.makedirs(f"../models/{config.NAME}/", exist_ok=True)
        joblib.dump(scaler, f'../models/{config.NAME}/scaler.pkl')
        joblib.dump(list(x.columns), f'../models/{config.NAME}/train_cols.pkl')

        df = pd.concat([df[["label", "kfold"]], x], axis=1)
        return df

    else:
        scaler = joblib.load(f'../models/{config.NAME}/scaler.pkl')
        train_cols = joblib.load(f'../models/{config.NAME}/train_cols.pkl')
        for c in set(train_cols) - set(x.columns):
            x.loc[:, c] = 0
        x = x[train_cols]
        x = pd.DataFrame(scaler.transform(x), columns=train_cols)
    
        return x


def preprocess_emb(df, train_mode=True):
    """
    categorical変数は，labelencode済みを仮定
    """
    
    if train_mode:
        x = df.drop(["label", "kfold"], axis=1)
        #y = df["label"]
    else:
        x = df

    rm_cols = config.COLS_RM
    cont_cols = config.CONT_COLS
    cat_cols = config.CAT_COLS

    x = x.replace("-", np.nan)

    x[cont_cols] = x[cont_cols].astype("float")
    # x[cat_cols] = x[cat_cols].astype("object")

    # 欠損値補完
    x[cont_cols] = x[cont_cols].fillna(x[cont_cols].mean())
    # x[cat_cols] = x[cat_cols].fillna(x[cat_cols].mode())
    
    # 標準化
    if train_mode:
        scaler = StandardScaler()
        x.loc[:, cont_cols] = scaler.fit_transform(x[cont_cols].values)

        os.makedirs(f"../models/{config.NAME}/", exist_ok=True)
        joblib.dump(scaler, f'../models/{config.NAME}/scaler.pkl')
        #joblib.dump(list(x.columns), f'../models/{config.NAME}/train_cols.pkl')

        df = pd.concat([df[["label", "kfold"]], x], axis=1)
        return df

    else:
        scaler = joblib.load(f'../models/{config.NAME}/scaler.pkl')
        # train_cols = joblib.load(f'../models/{config.NAME}/train_cols.pkl')
        # for c in set(train_cols) - set(x.columns):
        #     x.loc[:, c] = 0
        # x = x[train_cols]
        x.loc[:, cont_cols] = scaler.fit_transform(x[cont_cols].values)
    
        return x



def mean_target_encoding(data):
    
    df = copy.deepcopy(data)

    cat_cols = list(set(config.CAT_COLS) & set(config.FEATURES))

    encoded_dfs = []
    for fold in range(config.NUM_FOLDS):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for col in cat_cols:
            mapping_dict = dict(
                df_train.groupby(col)["label"].mean()
            )
            # column_enc is the new column we have wtih mean encoding
            df_valid.loc[:, col+"_enc"] = df_valid[col].map(mapping_dict)
        
        encoded_dfs.append(df_valid)
    
    encoded_df = pd.concat(encoded_dfs, axis=0)

    # encode test data
    df_test = pd.read_pickle(config.TEST_FILE)
    for col in cat_cols:
        mapping_dict = dict(
            df.groupby(col)["label"].mean()
        )
        # column_enc is the new column we have wtih mean encoding
        df_test.loc[:, col+"_enc"] = df_test[col].map(mapping_dict)
    
    df_test.to_pickle(config.TEST_FILE)

    return encoded_df
