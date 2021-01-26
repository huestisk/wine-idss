import pandas as pd

df1 = pd.read_json('database/winemag-data-130k-v2.json')
print(df1.shape)

df1.drop_duplicates(keep=False,inplace=True)
print(df1.shape)

del df1['taster_name']
del df1['taster_twitter_handle']

df2 = pd.read_json('database/winemag-data-newest-46500.json')
print(df2.shape)
df3 = pd.read_json('database/winemag-data-next-39492.json')
print(df3.shape)

df = pd.concat([df1, df2, df3])
print(df.shape)

df.drop_duplicates(keep=False, inplace=True)
print(df.shape)

df.to_pickle("./database.pkl")

# print(str(df["price"].isna().sum()) + " / " + str(len(df)))