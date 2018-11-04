import pandas as pd

app_charts = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_charts_top200.csv')
# app_details = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_details_top200.csv')
app_reviews = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_reviews_top200.csv', infer_datetime_format=True)

app_reviews['timestamp'] = pd.to_datetime(app_reviews['timestamp'])
app_reviews['week'] = app_reviews['timestamp'].dt.week
app_charts['timestamp'] = pd.to_datetime(app_charts['timestamp'])
app_charts['week'] = app_charts['timestamp'].dt.week
app_charts = app_charts[(app_charts.chart != 'Top Grossing')]

grp = app_charts.groupby(by=['appID', 'week', 'chart'])['rank'].apply(list)
app_charts_grouped = grp.reset_index()

app_reviews.set_index(['appID', 'week'])
id_week_grp = app_reviews.groupby(by=['appID', 'week'])['content'].apply(list)
app_reviews_grouped = id_week_grp.reset_index()

df = pd.concat([app_reviews_grouped, app_charts_grouped], axis=1, keys='week', join='inner')
df.to_csv('test.csv', index=False)

