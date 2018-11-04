import pandas as pd
from statistics import mean
import ast

df = pd.read_csv('test.csv', converters={'rank': ast.literal_eval})
df = df.drop(['appID.1', 'week.1'], axis=1)

df['mean_rank'] = -1
df['rank_previous_week'] = -1

current_id = -1
for idx, row in df.iterrows():
    df.loc[idx, 'mean_rank'] = mean(row['rank'])
    # print('current id {}'.format(current_id))
    # print('row appId {}'.format(row.appID))

    if current_id == row.appID:
        # write rank from previous week
        df.loc[idx, 'rank_previous_week'] = df.loc[idx-1, 'mean_rank']
    elif current_id == -1:
        current_id = row['appID']
        continue
    current_id = row['appID']

df.to_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\reviews_rank_for_appID_and_week.csv', index=False)