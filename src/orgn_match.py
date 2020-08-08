import pandas as pd
import json
from collections import defaultdict

# チームIDをチーム名に変換する辞書作成
team_df = pd.read_csv('../input/team.csv')
team_dic = dict(zip(team_df['team_id'], team_df['team_name']))

record_list = defaultdict(list)

# 2015 ~ 2018シーズン
for year in range(2015, 2019):
    team_df_year = team_df[team_df.year == year]
    for div, team_id in zip(team_df_year["div"], team_df_year["team_id"]):
        # 2stageかどうか
        is_2stage_2015 = year == 2015 and div == "J1"
        is_2stage_2016 = year == 2016 and div == "J1"

        # 後で１試合あたりの指標に変換するために試合数の情報を記憶
        num_match = 34 if div == 'J1' else 42

        if is_2stage_2015:
            first_df = pd.read_csv(f"../input/match/match_{year}_{div}_{team_id}_s1.csv")
            second_df = pd.read_csv(f"../input/match/match_{year}_{div}_{team_id}_s2.csv")
            full_df = pd.concat([first_df, second_df])
        elif is_2stage_2016:
            first_df = pd.read_csv(f"../input/match/match_{year}_{div}_{team_id}.csv")
            second_df = pd.read_csv(f"../input/match/match_{year}_{div}_{team_id}_2.csv")
            full_df = pd.concat([first_df, second_df])
        else:
            full_df = pd.read_csv(f"../input/match/match_{year}_{div}_{team_id}.csv")
            first_df = full_df.iloc[:num_match // 2]
            second_df = full_df.iloc[num_match // 2:]

        record_list["year"].append(year + 1)
        record_list["div"].append(div)
        record_list["team"].append(team_dic[team_id])

        for season, df in zip(["", "_first", "_second"], [full_df, first_df, second_df]):
            tmp = df["勝敗"].value_counts()
            record_list[f"team_win_ratio{season}"].append(
                tmp["○"] / df.shape[0] if "○" in tmp.keys() else 0)
            record_list[f"team_draw_ratio{season}"].append(
                tmp["△"] / df.shape[0] if "△" in tmp.keys() else 0)
            record_list[f"team_loss_ratio{season}"].append(
                tmp["●"] / df.shape[0] if "●" in tmp.keys() else 0)
            record_list[f"team_score_mean{season}"].append(
                df["スコア"].apply(lambda x: int(x.split(" - ")[0])).mean())
            record_list[f"team_scored_mean{season}"].append(
                df["スコア"].apply(lambda x: int(x.split(" - ")[1])).mean())

df = pd.DataFrame(record_list)
df.to_csv(f'../input/orgn/match.csv', index=False)