import numpy as np
import pandas as pd
import os
import pickle
from sklearn import model_selection
import config

import random
from collections import Counter, defaultdict


def stratified_group_k_fold(X, y, groups, k, seed=42):
    """
    input:
        X: dataframe
        y: target series
        groups: 指定するgroup変数のseries
        k: fold数
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices




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


    # kf = model_selection.StratifiedKFold(n_splits=config.NUM_FOLDS)
    #for f, (tr_idx, val_idx) in enumerate(stratified_group_k_fold(data, data.bins.values, groups=np.array(data['team'].values), k=config.NUM_FOLDS, seed=42)):
    for f, (tr_idx, val_idx) in enumerate(stratified_group_k_fold(data, data.bins.values, groups=np.array(data['name'].values), k=config.NUM_FOLDS, seed=42)):
        data.loc[val_idx, "kfold"] = f
    
    data = data.drop("bins", axis=1)

    return data


if __name__ == "__main__":
    df = pd.read_pickle(config.TRAIN_FILE)
    df = df.rename(columns={"time_played": "label"})

    df = create_folds(df)

    os.makedirs("../input/orgn/folds", exist_ok=True)
    df.to_pickle(config.TRAIN_FOLD_FILE)


