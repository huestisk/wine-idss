import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from collections import Counter

df = pd.read_pickle("data/data.pkl")
cols = df.columns
# print(cols.values)

# print("There are " + str(len(df)) + " unique reviews, with " +
#         str(len(df['title'].unique())) + " unique titles.")

na_title = df.loc[df["title"].isna()]
na_price = df.loc[df["price"].isna()]
na_variety = df.loc[df["variety"].isna()]
na_province = df.loc[df["province"].isna()]
na_country = df.loc[df["country"].isna()]
na_points = df.loc[df["points"].isna()]
na_designation = df.loc[df["designation"].isna()]

mean_price = round(df["price"].mean(skipna=True),3)
sigma_price = round(df["price"].std(skipna=True),3)
max_price = df["price"].max()
min_price = df["price"].min()

mean_points = round(df["points"].mean(skipna=True),3)
sigma_points = round(df["points"].std(skipna=True),3)
max_points = df["points"].max()
min_points = df["points"].min()

analysis = pd.DataFrame({
    'Unique Values': [len(df['title'].unique()), len(df['price'].unique()), len(df['variety'].unique()), len(df['province'].unique()), len(df['country'].unique()), len(df['points'].unique()), len(df['designation'].unique())],
    'NaN Values': [len(na_title), len(na_price), len(na_variety), len(na_province), len(na_country), len(na_points), len(na_designation)],
    'Max': ['N/A', max_price, 'N/A', 'N/A', 'N/A', max_points, 'N/A'],
    'Min': ['N/A', min_price, 'N/A', 'N/A', 'N/A', min_points, 'N/A'],
    'Average': ['N/A', mean_price, 'N/A', 'N/A', 'N/A', mean_points, 'N/A'],
    'Sigma': ['N/A', sigma_price, 'N/A', 'N/A', 'N/A', sigma_points, 'N/A']},
    index=["title", "price", "variety", "province", "country", "points", "designation"])


# print(analysis.to_markdown(tablefmt="grid"))

def plot_counts(xdata, ydata, category):
    plt.title("Top 10 Occurences of the Category '" + str(category) +"'")
    plt.xlabel(str(category)), plt.ylabel("Count")

    plt.bar(xdata, ydata)

    plt.xticks(rotation=90)
    plt.show()


def plot_dist(data, category):
    plt.title("Distribution of the Category '" + str(category) +"'")
    plt.xlabel(str(category)), plt.ylabel("Density")

    if category=="price":
        ax = sns.distplot(
            data,
            bins=np.logspace(np.log10(data.min()),np.log10(data.max()), 50),
            norm_hist=True
        )
        ax.set_xscale('log')
    else:
        sns.distplot(
            data,
            bins=20,
            fit=norm,
            norm_hist=True,
            kde=False
        )      

    plt.show()

for idx, col in enumerate(cols):
    if col=='description' or col=='title' or col=='winery':
        continue
    elif col=='points' or col=='price':
        plot_dist(df[col].dropna(), col)
    else:
        top10_counts = Counter(df[col].dropna()).most_common(10)
        xdata = [point[0] for point in top10_counts]
        ydata = [point[1] for point in top10_counts]
        plot_counts(xdata, ydata, col)

# print("There are " + str(len(na_price)) + " NaN values for the price out of " + str(len(df)) + " total reviews (" + str(round(100*len(na_price) / len(df), 2)) + "%).")



