NAME = "exp-032"


DESCRIPTION = "Reproducibility test"


CV = "StratioifiedFold_3splits"

# 5より良かった
NUM_FOLDS = 3


TRAIN_FILE = "../input/orgn/folds/train.pkl"
TEST_FILE = "../input/orgn/test.pkl"
TRAIN_FOLD_FILE = "../input/orgn/folds/train_folds.pkl"


MODEL_OUTPUT = '../models/'

MODEL = "LGBM"
# MODEL = "CB"
# MODEL = "NN"


COLS_RM = [
    "id",
    "name"
]

CONT_COLS = [
    #'height', 'weight', 
    'nth_year', 'is_youth',
    'j1_total_num_played', 'j1_total_scores', 'j2_total_num_played',
    'j2_total_scores', 'j3_total_num_played', 'j3_total_scores',
    'na_total_num_played', 'na_total_scores', 
    'team_win_ratio', 'team_draw_ratio',
    'team_loss_ratio', 'team_score_mean', 'team_scored_mean',
    'team_win_ratio_first', 'team_draw_ratio_first',
    'team_loss_ratio_first', 'team_score_mean_first',
    'team_scored_mean_first', 'team_win_ratio_second',
    'team_draw_ratio_second', 'team_loss_ratio_second',
    'team_score_mean_second', 'team_scored_mean_second',
    'prev3_num_played', 'prev2_num_played', 'prev1_num_played',
    'prev3_scores', 'prev2_scores', 'prev1_scores', 'prev3_time_played',
    'prev2_time_played', 'prev1_time_played',
    "birthdate",
    "salary",
    "ratio_prev12_time_played", "ratio_prev23_time_played", "ratio_prev13_time_played",
    "ratio_prev12_scores", "ratio_prev23_scores", "ratio_prev13_scores",
    'ratio_full_play','ratio_out_play', 'ratio_in_play', 'ratio_inout_play', 'ratio_bench_play', 'ratio_susp_play', "ratio_part_play", 
    'ratio_full_play_first', 'ratio_out_play_first',
    'ratio_in_play_first', 'ratio_inout_play_first', 'ratio_bench_play_first', 'ratio_susp_play_first', 
    'ratio_part_play_first',
    'ratio_full_play_second', 'ratio_out_play_second',
    'ratio_in_play_second', 'ratio_inout_play_second', 'ratio_bench_play_second', 'ratio_susp_play_second', 'ratio_part_play_second',
    'ratio_full_play_last3', 'ratio_out_play_last3',
    'ratio_in_play_last3', 'ratio_inout_play_last3', 'ratio_bench_play_last3', 'ratio_susp_play_last3', 'ratio_part_play_last3',
    'ratio_full_play_last8', 'ratio_out_play_last8',
    'ratio_in_play_last8', 'ratio_inout_play_last8', 'ratio_bench_play_last8', 'ratio_susp_play_last8','ratio_part_play_last8',
    'ratio_W', 'ratio_E', 'ratio_WE', 'ratio_WWE',
    'ratio_W_first', 'ratio_E_first', 'ratio_WE_first', 'ratio_WWE_first', 'ratio_Exit_first', 'ratio_W_any_first',
    'ratio_W_second', 'ratio_E_second', 'ratio_WE_second', 'ratio_WWE_second', 'ratio_Exit_second', 'ratio_W_any_second',
    'ratio_W_last8', 'ratio_E_last8', 'ratio_WE_last8', 'ratio_WWE_last8', 'ratio_Exit_last8', 'ratio_W_any_last8',
    'ratio_W_last3', 'ratio_E_last3', 'ratio_WE_last3', 'ratio_WWE_last3', 'ratio_Exit_last3', 'ratio_W_any_last3',
    'ratio_R', 'ratio_C',
    'ratio_R_first', 'ratio_C_first',
    'ratio_R_second', 'ratio_C_second',
    'ratio_R_last8', 'ratio_C_last8',
    'ratio_R_last3', 'ratio_C_last3',
    "relative_team_salary", "relative_team_position_salary",
    "relative_team_birthdate", "relative_team_position_birthdate",

]


CAT_COLS = [
    'team', 'position', 'nationality',
    'prev3_team', 'prev2_team', 'prev1_team', 
    'prev3_div', 'prev2_div','prev1_div', 'div',
    #"No", # Noは数値の方が良かった
    #"toJ1", "fromJ1",
]


# null_importanceで選ばれた特徴量
FEATURES = [
    'relative_team_position_birthdate', 'relative_team_position_salary', 'j1_total_num_played', 'prev1_time_played', 'relative_team_salary', 'j2_total_num_played', 'ratio_bench_play', 'No', 'ratio_prev23_time_played', 'ratio_prev12_scores', 'ratio_full_play', 'prev2_num_played', 'relative_team_birthdate', 'prev1_num_played', 'ratio_prev12_time_played', 'weight', 'ratio_full_play_second', 'prev2_time_played', 'salary', 'ratio_prev13_time_played', 'ratio_full_play_last3', 'ratio_prev23_scores', 'j1_total_scores', 'ratio_bench_play_first', 'prev3_num_played', 'ratio_full_play_last8', 'ratio_full_play_first', 'ratio_salary', 'prev3_time_played', 'nth_year', 'height', 'position', 'ratio_bench_play_last8', 'j2_total_scores', 'team_scored_mean_second', 'team_scored_mean', 'team_score_mean_second', 'team_draw_ratio_first', 'team_score_mean', 'prev_salary', 'team_loss_ratio_first', 'team_draw_ratio', 'team_score_mean_first', 'team_scored_mean_first', 'ratio_W', 'ratio_prev13_scores', 'ratio_part_play_first', 'birthdate', 'team_draw_ratio_second', 'ratio_out_play_first', 'prev3_team', 'ratio_bench_play_second', 'prev1_scores', 'ratio_in_play', 'team_win_ratio_first', 'ratio_out_play', 'ratio_part_play_last8', 'team_loss_ratio', 'prev2_scores', 'ratio_part_play_second', 'team_win_ratio_second', 'prev_ratio_out_play_second', 'ratio_in_play_second', 'na_total_num_played', 'team_loss_ratio_second', 'ratio_out_play_second', 'prev1_team', 'team_win_ratio', 'ratio_part_play', 'prev2_team', 'na_total_scores', 'prev_ratio_W', 'ratio_bench_play_last3', 'j3_total_num_played', 'team', 'ratio_in_play_first', 'prev_ratio_out_play_first', 'ratio_out_play_last8', 'ratio_in_play_last8', 'ratio_W_any_second', 'prev_ratio_out_play', 'prev_ratio_bench_play_first', 'ratio_R', 'ratio_part_play_last3', 'ratio_ratio_full_play_second', 'ratio_W_any_first', 'prev_ratio_in_play', 'prev_ratio_W_first', 'nationality', 'prev3_scores', 'prev_ratio_full_play_last8', 'prev_ratio_full_play_second', 'prev_ratio_full_play', 'ratio_out_play_last3', 'prev_ratio_bench_play', 'prev_ratio_full_play_first', 'ratio_susp_play', 'prev_ratio_bench_play_last8', 'ratio_W_any_last8', 'ratio_ratio_out_play', 'prev_ratio_susp_play', 'is_youth', 'prev_ratio_bench_play_last3', 'ratio_ratio_full_play', 'ratio_in_play_last3', 'ratio_ratio_W_second', 'prev_ratio_in_play_last8', 'ratio_ratio_bench_play_first', 'prev_ratio_W_last8', 'ratio_W_any_last3', 'prev_ratio_W_second', 'j3_total_scores', 'ratio_ratio_in_play', 'ratio_R_first', 'prev_ratio_E', 'ratio_ratio_full_play_first', 'prev_ratio_in_play_first', 'ratio_R_second', 'prev_ratio_R', 'ratio_C', 'ratio_susp_play_first', 'ratio_W_last8', 'ratio_ratio_full_play_last8', 'prev2_div', 'prev_ratio_in_play_last3', 'prev_ratio_full_play_last3', 'ratio_W_any', 'prev_ratio_bench_play_second', 'ratio_W_second', 'ratio_ratio_out_play_last8', 'prev_ratio_susp_play_last8', 'prev_ratio_in_play_second', 'prev1_div', 'ratio_ratio_out_play_second', 'div', 'prev3_div', 'prev_ratio_C', 'prev_ratio_out_play_last3', 'ratio_ratio_bench_play_last8', 'ratio_ratio_W_first', 'ratio_C_first', 'ratio_ratio_in_play_last8', 'ratio_ratio_W', 'ratio_ratio_bench_play', 'ratio_susp_play_last8', 'ratio_ratio_in_play_first', 'ratio_E', 'prev_ratio_inout_play', 'ratio_W_first', 'ratio_ratio_bench_play_second'
]


PARAMS_LGBM = {
    "objective": 'regression',
    "boosting": "gbdt",
    "metric": 'rmse',
    "learning_rate": 0.001,
    "num_leaves": 31,
    "max_depth": -1,
    # "min_data_in_leaf": 30,
    "min_child_samples": 20,
    # "feature_fraction": 0.9,
    # "bagging_freq": 1,
    # "bagging_fraction": 0.9,
    # "bagging_seed": 11,
    # "lambda_l1": 0.1,
    "verbosity": -1,
    "nthread": -1,
    "random_state": 42,
}


PARAMS_CAT = {
    "loss_function": "RMSE",
    'eval_metric': 'RMSE',
    "iterations": 100000,
    'learning_rate': 0.001,
    'depth': 5,
    'early_stopping_rounds': 100,
    'random_seed': 42,
    'allow_writing_files': False,
    'task_type': "CPU",
    'min_child_samples': 20,
}


PARAMS_NN = {
    "NUM_EPOCHS": 100,
    "NUM_EarlyStopping": 10,
    "lr": 5e-2,
}