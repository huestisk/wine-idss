import pandas as pd

# Read JSON files
df1 = pd.read_json('data/winemag-data-130k-v2.json')
df2 = pd.read_json('data/winemag-data-newest-46500.json')
df3 = pd.read_json('data/winemag-data-next-39492.json')

# Remove unnecessary columns
del df1['taster_name']
del df1['taster_twitter_handle']

# Concatenate
df = pd.concat([df2, df3, df1])
total_len = df.shape[0]

# Quick Analysis
print("There are " + str(len(df['description'].unique())) + " unique descriptions.")
print("There are " + str(len(df['title'].unique())) + " unique titles.")

# Remove duplicates
df.drop_duplicates(keep='last', inplace=True)
print("A total of " + str(total_len) + " reviews were scraped, " + str(df.shape[0]) + " were non-duplicates.")

# Save
df.to_pickle("data/data.pkl")