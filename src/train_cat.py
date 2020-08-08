import os
import gc
import argparse
import config
import utils
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score


#####################################################################
# mlflow
import mlflow  
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

#mlflow.set_tracking_uri('../mlruns/')
#mlflow.create_experiment(name="LGBM" , artifact_location='../mlruns/')

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

def run(df, fold):
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["label", "kfold"], axis=1)#.values
    y_train = df_train["label"]#.values

    x_valid = df_valid.drop(["label", "kfold"], axis=1)#.values
    y_valid = df_valid["label"]#.values

    # lgb dataset
    tr_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_valid, label=y_valid)

    # read params
    params = config.PARAMS_LGBM

    model = lgb.train(params, tr_data, num_boost_round=10000, valid_sets = [tr_data, val_data], verbose_eval=None, early_stopping_rounds=100, callbacks=callbacks)
    
    preds = model.predict(x_valid, num_iteration=model.best_iteration)

    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"Fold={fold}, RMSE={rmse}")


    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = df_train.drop(["label", "kfold"], axis=1).columns
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold

    # save the model
    os.makedirs(f"../models/{config.NAME}", exist_ok=True)
    joblib.dump(model, f"../models/{config.NAME}/{fold}.bin")

    return rmse, fold_importance_df


if __name__ == "__main__":

    # make logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    sc = logging.StreamHandler()
    logger.addHandler(sc)
    fh = logging.FileHandler(f'../logs/{config.NAME}.log')
    logger.addHandler(fh)
    callbacks = [log_evaluation(logger, period=100)]


    # read data
    df = pd.read_pickle(config.TRAIN_FOLD_FILE)

    df = df.replace("-", np.nan)

    #df = df[config.CONT_COLS + config.CAT_COLS + ["label", "kfold"]]
    df[config.CONT_COLS] = df[config.CONT_COLS].astype('float')
    df[config.CAT_COLS] = df[config.CAT_COLS].astype('category')

    df = df[config.FEATURES+["label", "kfold"]]


    num_folds = config.NUM_FOLDS
    score = 0
    feature_importance_df = pd.DataFrame()
    for i in range(num_folds):
        rmse, fold_importance_df = run(fold=i)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        score += rmse / num_folds

    print(f"CV score={score}")    

    # output importance
    feature_importance_mean = feature_importance_df[['feature','importance']].groupby('feature').mean().sort_values(by="importance", ascending=False).reset_index()
    feature_importance_mean.to_csv(f"../importance/{config.NAME}_{score}.csv", index=False)
    # sns.barplot(feature_importance_mean.head(50).loc[:,'feature'], feature_importance_mean.head(50).loc[:,'importance'])
    # plt.xticks(rotation=90)
    # plt.savefig(f"../importance/{config.NAME}_{score}.png")

    ###############################################################################
    # mlflow log
    mlflow.log_param('cv_strategy', config.CV)
    mlflow.log_param('model_name', config.MODEL)
    mlflow.log_param('model_params', config.PARAMS_LGBM)  # model parameters 
    mlflow.log_param('description', config.DESCRIPTION) 
    # mlflow.log_param('features', features)  # features
    mlflow.log_metric('cv_score', score)  # cv_score

    mlflow.log_artifact(f"../importance/{config.NAME}_{score}.csv")  # importance
    mlflow.log_artifact(f'../logs/{config.NAME}.log') # log file
