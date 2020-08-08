import os
import numpy as np
import pandas as pd

import config


label = 'time_played'




if __name__ == "__main__":
    df_2016 = pd.read_csv("../input/train_2016.csv")
    df_2017 = pd.read_csv("../input/train_2017.csv")
    df_2018 = pd.read_csv("../input/train_2018.csv")
    df_2019 = pd.read_csv("../input/test.csv")


    # event_play_yearとの結合，各選手の試合出場状況のサマリcsv
    event = pd.read_csv("../input/orgn/event_play_2015.csv")
    df_2016 = pd.merge(df_2016, event, on=["team", "name"], how="left")

    event = pd.read_csv("../input/orgn/event_play_2016.csv")
    df_2017 = pd.merge(df_2017, event, on=["team", "name"], how="left")

    event = pd.read_csv("../input/orgn/event_play_2017.csv")
    df_2018 = pd.merge(df_2018, event, on=["team", "name"], how="left")

    event = pd.read_csv("../input/orgn/event_play_2018.csv")
    df_2019 = pd.merge(df_2019, event, on=["team", "name"], how="left")


    # matchとの結合, 各チームの各年度の結果サマリcsv
    match = pd.read_csv("../input/orgn/match.csv")
    df_2016 = pd.merge(df_2016, match[match.year==2016].drop(["year"], axis=1), on="team", how="left")
    df_2017 = pd.merge(df_2017, match[match.year==2017].drop(["year"], axis=1), on="team", how="left")
    df_2018 = pd.merge(df_2018, match[match.year==2018].drop(["year"], axis=1), on="team", how="left")
    df_2019 = pd.merge(df_2019, match[match.year==2019].drop(["year"], axis=1), on="team", how="left")

    # WEの結合, 各チームの各年度の結果サマリcsv
    # RCの結合, 各チームの各年度の結果サマリcsv
    we = pd.read_csv("../input/orgn/event_we_2015.csv")
    rc = pd.read_csv("../input/orgn/event_rc_2015.csv")
    df_2016 = pd.merge(df_2016, we, on=["team", "name"], how="left")
    df_2016 = pd.merge(df_2016, rc, on=["team", "name"], how="left")

    we = pd.read_csv("../input/orgn/event_we_2016.csv")
    rc = pd.read_csv("../input/orgn/event_rc_2015.csv")
    df_2017 = pd.merge(df_2017, we, on=["team", "name"], how="left")
    df_2017 = pd.merge(df_2017, rc, on=["team", "name"], how="left")

    we = pd.read_csv("../input/orgn/event_we_2017.csv")
    rc = pd.read_csv("../input/orgn/event_rc_2015.csv")
    df_2018 = pd.merge(df_2018, we, on=["team", "name"], how="left")
    df_2018 = pd.merge(df_2018, rc, on=["team", "name"], how="left")

    we = pd.read_csv("../input/orgn/event_we_2018.csv")
    rc = pd.read_csv("../input/orgn/event_rc_2015.csv")
    df_2019 = pd.merge(df_2019, we, on=["team", "name"], how="left")
    df_2019 = pd.merge(df_2019, rc, on=["team", "name"], how="left")


    ###########################################################################
    # 昨年度との割合特徴量の作成
    cols_common = ["id", "salary", 
    'ratio_full_play','ratio_out_play', 'ratio_in_play', 'ratio_inout_play', 'ratio_bench_play', 'ratio_susp_play', 
    'ratio_full_play_first', 'ratio_out_play_first',
    'ratio_in_play_first', 'ratio_inout_play_first', 'ratio_bench_play_first',
    'ratio_susp_play_first', 'ratio_full_play_second', 'ratio_out_play_second',
    'ratio_in_play_second', 'ratio_inout_play_second', 'ratio_bench_play_second', 
    'ratio_full_play_last3', 'ratio_out_play_last3',
    'ratio_in_play_last3', 'ratio_inout_play_last3', 'ratio_bench_play_last3', 'ratio_susp_play_last3',
    'ratio_full_play_last8', 'ratio_out_play_last8',
    'ratio_in_play_last8', 'ratio_inout_play_last8', 'ratio_bench_play_last8',
    'ratio_susp_play_last8',
    'ratio_W', 'ratio_E', 'ratio_WE', 'ratio_WWE',
    'ratio_W_first', 'ratio_E_first', 'ratio_WE_first', 'ratio_WWE_first',
    'ratio_W_second', 'ratio_E_second', 'ratio_WE_second', 'ratio_WWE_second',
    'ratio_W_last8', 'ratio_E_last8', 'ratio_WE_last8', 'ratio_WWE_last8',
    'ratio_W_last3', 'ratio_E_last3', 'ratio_WE_last3', 'ratio_WWE_last3',
    'ratio_R', 'ratio_C',
    'ratio_R_first', 'ratio_C_first',
    'ratio_R_second', 'ratio_C_second',
    'ratio_R_last8', 'ratio_C_last8',
    'ratio_R_last3', 'ratio_C_last3',]
    
    rename_dict = {}
    for c in cols_common:
        if c != "id":
            df_2016[c] = df_2016[c].replace("-", 0).astype("float")
            df_2017[c] = df_2017[c].replace("-", 0).astype("float")
            df_2018[c] = df_2018[c].replace("-", 0).astype("float")
            df_2019[c] = df_2019[c].replace("-", 0).astype("float")
            rename_dict[c] = f"prev_{c}"
    df_2017 = pd.merge(df_2017, df_2016[cols_common].rename(columns=rename_dict), on="id", how="left")
    df_2018 = pd.merge(df_2018, df_2017[cols_common].rename(columns=rename_dict), on="id", how="left")
    df_2019 = pd.merge(df_2019, df_2018[cols_common].rename(columns=rename_dict), on="id", how="left")

    for c in cols_common:
        if c != "id":
            df_2016[f"ratio_{c}"] = np.nan
            df_2017[f"ratio_{c}"] = np.nan
            df_2018[f"ratio_{c}"] = np.nan
            df_2019[f"ratio_{c}"] = np.nan
            df_2017.loc[(df_2017[f"prev_{c}"]!=0), f"ratio_{c}"] =  df_2017.loc[(df_2017[f"prev_{c}"]!=0), c] / df_2017.loc[(df_2017[f"prev_{c}"]!=0), f"prev_{c}"]
            df_2018.loc[(df_2018[f"prev_{c}"]!=0), f"ratio_{c}"] =  df_2018.loc[(df_2018[f"prev_{c}"]!=0), c] / df_2018.loc[(df_2018[f"prev_{c}"]!=0), f"prev_{c}"]
            df_2019.loc[(df_2019[f"prev_{c}"]!=0), f"ratio_{c}"] =  df_2019.loc[(df_2019[f"prev_{c}"]!=0), c] / df_2019.loc[(df_2019[f"prev_{c}"]!=0), f"prev_{c}"]

    ###########################################################################
    
    df_train = pd.concat([df_2016, df_2017], axis=0).reset_index(drop=True)
    df_train = pd.concat([df_train, df_2018], axis=0).reset_index(drop=True)
    df_test = df_2019


    # print(df_train.select_dtypes("number").columns)
    # print(df_train.select_dtypes("object").columns)


    os.makedirs("../input/orgn", exist_ok=True)
    df_train.to_pickle(config.TRAIN_FILE)
    df_test.to_pickle(config.TEST_FILE)