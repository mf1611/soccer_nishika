#! /bin/sh

cd src/
python make_data.py
python make_features.py
# python create_folds.py
python create_folds2.py
python null_importance_lgbm.py