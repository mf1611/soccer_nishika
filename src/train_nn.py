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
import models
import dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

#####################################################################
# mlflow
import mlflow  
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

#mlflow.set_tracking_uri('./mlruns/')

# 別の実験を実施する際に，異なるディレクトリで管理
# EXPERIMENT_NAME = "test"
# mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.start_run(run_id=None) # run_idの初期化（ファイル名となる）
mlflow.set_tag(MLFLOW_RUN_NAME, config.NAME)  # run_nameを，config.NAMEにして管理

# run_id = mlflow.active_run().info.run_id
# ARTIFACT_URI = f"../mlruns/"  # artifactが保存されるuriが変更できないので，無理やり指定 
# print(mlflow.get_artifact_uri()) 


####################################################################

# output log in case of lightgbm
import logging
from lightgbm.callback import _format_eval_result

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

#####################################################################


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def run(df, fold):
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.DLDataset(df_train, cont_cols=cont_cols, cat_cols=cat_cols, output_col="label")
    valid_dataset = dataset.DLDataset(df_valid, cont_cols=cont_cols, cat_cols=cat_cols, output_col="label")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)


    model = models.DLModel(num_cont_cols=len(cont_cols), num_unique_cat_cols=num_unique_cat_cols).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PARAMS_NN["lr"])
    best_loss = 10000
    patience = 0
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(config.PARAMS_NN["NUM_EPOCHS"]):
        start_time = time.time()
        
        model.train()
        train_loss = 0.
        for y_batch, x_cont_batch, x_cat_batch in train_loader:
            y_batch = y_batch.to(device)
            x_cont_batch = x_cont_batch.to(device)
            x_cat_batch = x_cat_batch.to(device)

            y_pred = model(x_cont_batch, x_cat_batch)
            loss = loss_fn(y_batch, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += np.sqrt(loss.item()) / len(train_loader)
        
        model.eval()
        valid_loss = 0.
        for y_batch, x_cont_batch, x_cat_batch in valid_loader:
            y_batch = y_batch.to(device)
            x_cont_batch = x_cont_batch.to(device)
            x_cat_batch = x_cat_batch.to(device)

            y_pred = model(x_cont_batch, x_cat_batch)
            loss = loss_fn(y_batch, y_pred)
            valid_loss += np.sqrt(loss.item()) / len(valid_loader)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        logger.info(f'Epoch {(epoch)}, train_loss: {train_loss}, valid_loss: {valid_loss}, time: {(time.time()-start_time)}')
        
        # Eearly Stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_param = model.state_dict()
            torch.save(best_param, f"../models/{config.NAME}/{fold}_best_param.pt")
            patience = 0
        else:
            patience += 1
            if patience >= config.PARAMS_NN["NUM_EarlyStopping"]:
                torch.cuda.empty_cache()
                break
        
        torch.cuda.empty_cache()

    print(f"Fold={fold}, RMSE={best_loss}")

    return best_loss, train_loss_list, valid_loss_list


if __name__ == "__main__":

    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # make logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    sc = logging.StreamHandler()
    logger.addHandler(sc)
    fh = logging.FileHandler(f'../logs/{config.NAME}.log')
    logger.addHandler(fh)


    # read data
    df = pd.read_pickle(config.TRAIN_FOLD_FILE)
    cont_cols = config.CONT_COLS
    cat_cols = config.CAT_COLS

    # print(df.isnull().sum())

    # labelencodeしてから，embedding
    num_unique_cat_cols = []
    label_encoders = {}
    df[cat_cols] = df[cat_cols].astype("str")
    for col in cat_cols:
        label_encoders[col] = LabelEncoder()
        df.loc[:, col] = df[col].fillna("NONE")
        df.loc[:, col] = label_encoders[col].fit_transform(df[col].values)
        num_unique_cat_cols.append(len(label_encoders[col].classes_))
    os.makedirs(f"../models/{config.NAME}/", exist_ok=True)
    joblib.dump(label_encoders, f'../models/{config.NAME}/label_encoders.pkl')
    joblib.dump(num_unique_cat_cols, f'../models/{config.NAME}/num_unique_cat_cols.pkl')

    df = utils.preprocess_emb(df, train_mode=True)

    df = df[config.FEATURES+["label", "kfold"]]


    num_folds = config.NUM_FOLDS
    score = 0
    for i in range(num_folds):
        rmse, train_loss_list, valid_loss_list = run(fold=i)
        score += rmse / num_folds

    print(f"CV score={score}")

    plt.plot(range(len(train_loss_list)), train_loss_list, color="r", label='train')
    plt.plot(range(len(valid_loss_list)), valid_loss_list, color="g", label='valid')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"../loss/{config.NAME}_{score}.png")


    ###############################################################################
    # mlflow log
    mlflow.log_param('cv_strategy', config.CV)
    mlflow.log_param('model_name', config.MODEL)
    mlflow.log_param('model_params', config.PARAMS_NN)  # model parameters 
    mlflow.log_param('description', config.DESCRIPTION) 
    # mlflow.log_param('features', features)  # features
    mlflow.log_metric('cv_score', score)  # cv_score

    mlflow.log_artifact(f"../loss/{config.NAME}_{score}.png")  # loss
    mlflow.log_artifact(f'../logs/{config.NAME}.log') # log file
