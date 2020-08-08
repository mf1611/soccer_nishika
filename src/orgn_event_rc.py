import pandas as pd
import json

# チームIDをチーム名に変換する辞書作成
team_df = pd.read_csv('../input/team.csv')
team_dic = dict(zip(team_df['team_id'], team_df['team_name']))


# ○: フル出場, ▲: 途中出場, ▽: 途中退場, ◆: 途中出場途中退場, ×: 出場停止, ※: 他大会の影響で出場停止, B: ベンチ入り
# 各々の回数をカウント
tmp = ['ratio_R', 'ratio_C']
colnames = ['ratio_R', 'ratio_C']
colnames += [i + '_first' for i in tmp]
colnames += [i + '_second' for i in tmp]
colnames += [i + '_last8' for i in tmp]
colnames += [i + '_last3' for i in tmp]

# W: 警告, E: 退場, WE: 警告1退場1, WWE: 1試合2警告による退場
characters = ['R', 'C'] * 5  # first, second, last8, last3, full

# 2015 ~ 2018シーズンのjsonをcsvに変換
for year in range(2015, 2019):
    dfs = []
    team_df_year = team_df[team_df.year == year]
    for div, team_id in zip(team_df_year["div"], team_df_year["team_id"]):
        # 2stageかどうか(J1かつ，2017年以前は2stage)
        is_2stage = year < 2017 and div == "J1"

        # 後で１試合あたりの指標に変換するために試合数の情報を記憶
        if div == 'J1':
            num_match = 34
        else:
            num_match = 42

        if is_2stage:
            with open(f"../input/event/event_rc_{year}_{div}_{team_id}_s1.json") as f:
                first_dic = json.load(f)
            with open(f"../input/event/event_rc_{year}_{div}_{team_id}_s2.json") as f:
                second_dic = json.load(f)

            first_df = pd.DataFrame.from_dict(first_dic, orient='index')
            second_df = pd.DataFrame.from_dict(second_dic, columns=list(range(17, 34)), orient='index')
            full_df = pd.concat([first_df, second_df], axis=1, join='outer')

            # stageごとに分けて集計
            first_df = full_df.iloc[:, :num_match // 2]
            second_df = full_df.iloc[:, num_match // 2:]
            
            # 年度終わりのN試合
            last8_df = full_df.iloc[:, -5:]
            last3_df = full_df.iloc[:, -3:]
        else:
            with open(f"../input/event/event_rc_{year}_{div}_{team_id}.json") as f:
                dic = json.load(f)

            full_df = pd.DataFrame.from_dict(dic, orient='index')

            # シーズン前半・後半に分けて集計するためにdataframeを分割
            first_df = full_df.iloc[:, :num_match // 2]
            second_df = full_df.iloc[:, num_match // 2:]

            last8_df = full_df.iloc[:, -8:]
            last3_df = full_df.iloc[:, -3:]


        # 各行（各選手）ごとに，各状態の合計
        data = []
        for colname, ch in zip(colnames, characters):
            if 'first' in colname:
                data.append(first_df.apply(lambda d: d.str.contains(ch)).sum(axis=1).values)
            elif 'second' in colname:
                data.append(second_df.apply(lambda d: d.str.contains(ch)).sum(axis=1).values)
            elif 'last8' in colname:
                data.append(last8_df.apply(lambda d: d.str.contains(ch)).sum(axis=1).values)
            elif 'last3' in colname:
                data.append(last3_df.apply(lambda d: d.str.contains(ch)).sum(axis=1).values)
            else:
                data.append(full_df.apply(lambda d: d.str.contains(ch)).sum(axis=1).values)
        # 1試合あたりの値に変換
        df = pd.DataFrame(data).T
        df /= num_match
        df.columns = colnames
        df['team_id'] = team_id
        df['team'] = df['team_id'].map(int).map(team_dic)
        df.drop(['team_id'], axis=1, inplace=True)
        df['name'] = full_df.index  # 選手名

        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(f'../input/orgn/event_rc_{year}.csv', index=False)