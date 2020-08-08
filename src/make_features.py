import os
import time
from tqdm import tqdm
import gc
import argparse
import joblib
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score

import config
import utils



def add_features(df):

    for colname in ['prev3_div', 'prev2_div', 'prev1_div', 
                    'prev3_num_played', 'prev2_num_played', 'prev1_num_played', 
                    'prev3_scores', 'prev2_scores', 'prev1_scores', 
                    'prev3_time_played', 'prev2_time_played', 'prev1_time_played']:
        df[colname] = df[colname].replace('-',0)
        df[colname] = df[colname].fillna(0)
        df[colname] = df[colname].map(int)


    df["birthdate"] = df["birthdate"].apply(lambda x: int(x[:4]))
    df["nationality"] = df["nationality"].isnull() * 1


    # 入れても変化なし
    # df["toJ1"] = ((df["div"]=="J1") & (df["prev1_div"]!="J1")) * 1
    # df["fromJ1"] = ((df["div"]!="J1") & (df["prev1_div"]=="J1")) * 1


    df["ratio_prev12_time_played"] = np.nan
    df["ratio_prev23_time_played"] = np.nan
    df["ratio_prev13_time_played"] = np.nan
    df.loc[(df["prev2_time_played"]!=0), "ratio_prev12_time_played"] = df.loc[(df["prev2_time_played"]!=0), "prev1_time_played"] / df.loc[(df["prev2_time_played"]!=0), "prev2_time_played"]
    df.loc[(df["prev3_time_played"]!=0), "ratio_prev23_time_played"] = df.loc[(df["prev3_time_played"]!=0), "prev2_time_played"] / df.loc[(df["prev3_time_played"]!=0), "prev3_time_played"]
    df.loc[(df["prev3_time_played"]!=0), "ratio_prev13_time_played"] = df.loc[(df["prev3_time_played"]!=0), "prev1_time_played"] / df.loc[(df["prev3_time_played"]!=0), "prev3_time_played"]


    df["ratio_prev12_scores"] = np.nan
    df["ratio_prev23_scores"] = np.nan
    df["ratio_prev13_scores"] = np.nan
    df.loc[(df["prev2_scores"].notnull()), "ratio_prev12_scores"] = df.loc[(df["prev2_scores"]!=0), "prev1_scores"] / df.loc[(df["prev2_scores"]!=0), "prev2_scores"]
    df.loc[(df["prev3_scores"]!=0), "ratio_prev23_scores"] = df.loc[(df["prev3_scores"]!=0), "prev2_scores"] / df.loc[(df["prev3_scores"]!=0), "prev3_scores"]
    df.loc[(df["prev3_scores"]!=0), "ratio_prev13_scores"] = df.loc[(df["prev3_scores"]!=0), "prev1_scores"] / df.loc[(df["prev3_scores"]!=0), "prev3_scores"]


    # for colname in ['is_youth', 'j1_total_num_played', 'j1_total_scores', 'j2_total_num_played', 'j2_total_scores', 
    #                 'j3_total_num_played', 'j3_total_scores', 'na_total_num_played', 'na_total_scores']:
    #     df[colname] = df[colname].fillna(0)
    

    # for colname in ['nationality']:
    #     df[colname] = df[colname].fillna('japan')
    

    # for colname in ['prev3_div', 'prev2_div', 'prev1_div', 
    #                 'prev3_num_played', 'prev2_num_played', 'prev1_num_played', 
    #                 'prev3_scores', 'prev2_scores', 'prev1_scores', 
    #                 'prev3_time_played', 'prev2_time_played', 'prev1_time_played']:
    #     df[colname] = df[colname].replace('-',0)
    #     df[colname] = df[colname].fillna(0)
    #     df[colname] = df[colname].map(int)
    

    # for colname in ['rat_full_play', 'rat_out_play',
    #    'rat_in_play', 'rat_inout_play', 'rat_bench_play', 'rat_susp_play',
    #    'rat_full_play_first', 'rat_out_play_first', 'rat_in_play_first',
    #    'rat_inout_play_first', 'rat_bench_play_first', 'rat_susp_play_first',
    #    'rat_full_play_second', 'rat_out_play_second', 'rat_in_play_second',
    #    'rat_inout_play_second', 'rat_bench_play_second',
    #    'rat_susp_play_second']:
    #     # 前年の所属リーグがJ1ないしJ2のチームで、値がnullの選手は、0で補完
    #     ext_rows = (df['prev1_div'].isin(['1', '2'])) & (df[colname].isnull())
    #     df.loc[ext_rows, colname] = df.loc[ext_rows, colname].fillna(0)
    #     # 前年の所属リーグがJ1ないしJ2のチームでなく、値がnullの選手は、-999で補完
    #     ext_rows = (~df['prev1_div'].isin(['1', '2'])) & (df[colname].isnull())
    #     df.loc[ext_rows, colname] = df.loc[ext_rows, colname].fillna(-999)
    #     df[colname] = df[colname].map(float)



    # teamごとのグループ特徴量と，相対特徴量
    # positionごとも
    usecols = [
        "birthdate",
        "salary",
    ]
    for col in usecols:
        df[col] = df[col].replace("-", 0)
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype("float")

        df_team_mean = df.groupby("team")[[col]].mean().rename(columns={col: f"{col}_mean_team"}).reset_index()

        df_team_position_mean = df.groupby(["team", "position"])[[col]].mean().rename(columns={col: f"{col}_mean_team_position"}).reset_index()

        df = df.merge(df_team_mean, on="team", how="left")
        df[f"relative_team_{col}"] = df[col] / df[f"{col}_mean_team"]
        df = df.drop([f"{col}_mean_team"], axis=1)

        df = df.merge(df_team_position_mean, on=["team", "position"], how="left")
        df[f"relative_team_position_{col}"] = df[col] / df[f"{col}_mean_team_position"]
        df = df.drop([f"{col}_mean_team_position"], axis=1)
    
    return df


if __name__ == "__main__":
    df_train = pd.read_pickle(config.TRAIN_FILE)
    df_test = pd.read_pickle(config.TEST_FILE)

    df_train = add_features(df_train)
    df_test = add_features(df_test)

    df_train.to_pickle(config.TRAIN_FILE)
    df_test.to_pickle(config.TEST_FILE)
