import pandas as pd

# get all dfs in place with proper columns
# reduce all dfs to only top200 ranks appID's
# ... join all and make a massive data set!?

details_columns =['id', 'timestamp', 'day', 'firstseen', 'title', 'developer', 'developerwebsite', 'supportwebsite', 'editorial_badge',
                  'category', 'update_date', 'purchases', 'price', 'screenshots', 'description', 'quan_description', 'releasenotes',
                  'compatibility', 'watch', 'content_rating', 'size', 'language', 'quan_language', 'english', 'appversion',
                  'ratings', 'rating', 'ratingscurrentversion', 'ratingcurrentversion', 'moreapps', 'quan_moreapps',
                  'alsoapps', 'quan_alsoapps', 'partofbundle', 'quan_partofbundle', 'app_url']
reviews_columns = ['appID', 'timestamp', 'day', 'reviewID', 'title', 'author', 'content', 'updated', 'rating', 'appversion']

app_reviews = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\sbm_app_reviews.csv',
                          infer_datetime_format=True, header=None, names=reviews_columns)
top200_ids = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\top200_appIDs_only.csv')
app_details = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\sbm_app_details.csv', header=None,
                          names=details_columns)
# app_charts = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_charts_top200.csv')

cols_to_del = ['developer', 'developerwebsite', 'supportwebsite', 'editorial_badge', 'purchases', 'screenshots',
                          'description', 'quan_description', 'releasenotes', 'compatibility', 'watch', 'content_rating',
                          'size', 'language', 'quan_language', 'english', 'moreapps', 'quan_moreapps', 'alsoapps', 'quan_alsoapps',
                          'partofbundle','quan_partofbundle','app_url']
app_details = app_details.drop(columns=cols_to_del)
print('read all data')

ids = top200_ids.appID.tolist()
# app_reviews_top200 = pd.DataFrame(columns=app_reviews.columns)
cnt = 0

print('begin reduce app review data')
app_reviews_top200 = app_reviews[app_reviews['appID'].isin(ids)]
# for row in app_reviews.itertuples():
#     if row.appID in ids:
#         app_reviews_top200.loc[cnt, reviews_columns] = row[1:11]
#         cnt += 1
app_reviews_top200.to_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_reviews_top200.csv', index=False)
print('done saving app reviews top 200')

# details_columns_reduced = app_details.columns.tolist()
# app_details_top200 = pd.DataFrame(columns=app_details.columns)
cnt = 0

print('begin reduce app details data')
app_details_top200 = app_details[app_details['id'].isin(ids)]
# for row in app_details.itertuples():
#     if row.id in ids:
#         app_details_top200.loc[cnt, details_columns_reduced] = row[1:]
#         cnt += 1
app_details_top200.to_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\app_details_top200.csv', index=False)
print('done saving app details top 200')
