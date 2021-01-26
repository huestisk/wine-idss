import pandas as pd
from tabulate import tabulate

df = pd.read_pickle("data/data.pkl")

print("There are " + str(len(df['description'].unique())) + " unique reviews, with " +
        str(len(df['title'].unique())) + " unique titles.")

na_title = df.loc[df["title"].isna()]
na_price = df.loc[df["price"].isna()]
na_variety = df.loc[df["variety"].isna()]
na_province = df.loc[df["province"].isna()]
na_country = df.loc[df["country"].isna()]
na_points = df.loc[df["points"].isna()]


nans = pd.DataFrame({'NaN Values': [len(na_title), len(na_price), 
        len(na_variety), len(na_province), len(na_country), len(na_points)]},
        index=["title", "price", "variety", "province", "country", "points"])

print(nans.to_markdown(tablefmt="grid"))


