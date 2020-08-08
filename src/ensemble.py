import pandas as pd


lgbm = pd.read_csv(f"../output/exp-008.csv")
nn = pd.read_csv(f"../output/exp-007.csv")

lgbm["time_played"] = lgbm["time_played"] * 0.8 + nn["time_played"] * 0.2

print(lgbm.head())

lgbm.to_csv(f"../output/ensemble_exp7_8.csv", index=False)


# 結局，上記でアンサンブルすると下がってしまう．．．