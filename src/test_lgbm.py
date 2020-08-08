import os
import gc
import argparse
import config
import utils
import joblib
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score


if __name__ == "__main__":
    submit_df = pd.read_csv("../input/sample_submission.csv")
    df = pd.read_pickle(config.TEST_FILE)

    df = df.replace("-", np.nan)

    df_tot = pd.merge(submit_df, df, on="id", how="left")
    features = [col for col in df_tot.columns if col not in submit_df.columns]
    submit_df = df_tot.loc[:, submit_df.columns]
    df = df_tot.loc[:, features].reset_index(drop=True)


    df[config.CONT_COLS] = df[config.CONT_COLS].astype('float')
    df[config.CAT_COLS] = df[config.CAT_COLS].astype('category')
    
    features = config.FEATURES
    
    # for col in config.CONT_COLS:
    #     features.append("relative_team_"+col)
    #     features.append("relative_team_position_"+col)

    # # target_encodingのカラム追加
    # cat_cols = list(set(config.CAT_COLS) & set(config.FEATURES))
    # for col in cat_cols:
    #     features.append(col+"_enc")

    df = df[features]


    num_folds = config.NUM_FOLDS
    pred = np.zeros(len(df))
    for fold in range(num_folds):
        model = joblib.load(f"../models/{config.NAME}/{fold}.bin")
        pred += model.predict(df, num_iteration=model.best_iteration) / num_folds
    
    submit_df.loc[:, "time_played"] = pred
    submit_df.to_csv(f"../output/{config.NAME}.csv", index=False)
    print('finish test')
    print('-'*60)
    print('-'*60)
