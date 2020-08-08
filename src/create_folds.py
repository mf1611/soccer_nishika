import numpy as np
import pandas as pd
import os
import pickle
from sklearn import model_selection
import config

def create_folds(data):
    """
    目的変数の値を，ビンに分割して，StratifiedKFolds
    """

    data["kfold"] = -1

    # 行のランダム入れ替え
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # ビンの数, スタージェスの公式
    num_bins = np.floor(1 + np.log2(len(data)))

    data.loc[:, "bins"] = pd.cut(data["label"], bins=int(num_bins), labels=False)

    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (tr_idx, val_idx) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[val_idx, "kfold"] = f
    
    data = data.drop("bins", axis=1)

    return data


if __name__ == "__main__":
    df = pd.read_pickle(config.TRAIN_FILE)
    df = df.rename(columns={"time_played": "label"})

    df = create_folds(df)
    os.makedirs("../input/orgn/folds", exist_ok=True)
    df.to_pickle(config.TRAIN_FOLD_FILE)


