import os
import gc
import argparse
import config
import joblib
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils
import models

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    submit_df = pd.read_csv("../input/sample_submission.csv")
    df = pd.read_pickle(config.TEST_FILE)

    cont_cols = config.CONT_COLS
    cat_cols = config.CAT_COLS
    
    df_tot = pd.merge(submit_df, df, on="id", how="left")
    features = [col for col in df_tot.columns if col not in submit_df.columns]
    submit_df = df_tot.loc[:, submit_df.columns]
    df = df_tot.loc[:, features].reset_index(drop=True)

    label_encoders = joblib.load(f'../models/{config.NAME}/label_encoders.pkl')
    df[cat_cols] = df[cat_cols].astype("str")
    for col in cat_cols:
        le = label_encoders[col]
        df.loc[:, col] = df[col].fillna("NONE")
        df.loc[:, col] = df[col].map(lambda s: 'UNKNOWN' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, 'UNKNOWN')
        df.loc[:, col] = le.transform(df[col].values)
    
    num_unique_cat_cols = joblib.load(f'../models/{config.NAME}/num_unique_cat_cols.pkl')

    df = utils.preprocess_emb(df, train_mode=False)

    x_cont_test = torch.tensor(df[cont_cols].values, dtype=torch.float32).to(device)
    x_cat_test = torch.tensor(df[cat_cols].values, dtype=torch.long).to(device)
    
    num_folds = config.NUM_FOLDS
    pred = np.zeros(len(df))
    for fold in range(num_folds):
        model = models.DLModel(num_cont_cols=x_cont_test.shape[1], num_unique_cat_cols=num_unique_cat_cols).to(device)
        model.load_state_dict(torch.load(f"../models/{config.NAME}/{fold}_best_param.pt"))
        model.eval()
        pred += model(x_cont_test, x_cat_test).cpu().detach().numpy() / num_folds
    
    submit_df.loc[:, "time_played"] = pred
    submit_df.to_csv(f"../output/{config.NAME}.csv", index=False)
    print('finish test')
    print('-'*60)
    print('-'*60)
