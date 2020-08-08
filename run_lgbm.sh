#! /bin/sh

cd src/
python make_data.py
python make_features.py
python create_folds2.py
python train_lgbm.py
python test_lgbm.py