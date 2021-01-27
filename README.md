# wine-idss

## Data Scraping

The data source and scraper was adapted from https://github.com/zackthoutt/wine-deep-learning. The scaper was adapted and updated with improved runtime. All code regarding the scraping can be found under "data/".

## Data Analysis

Improved data analysis was added. Specifically checking if NaN values should be removed. Since the distributions are changed when removed, all data is used to compute the features. The notebook analyzing this is "data_analysis/data_analysis.ipynb".

## Proprocessing

Combining words has been reduced to only the manual dictionaries. Features now include both words and bigrams based on separately compute TF-IDF analyses. The code was streamlined into one main script "preprocessing/preprocessing.py".

## Models


## Recommender


## Run
To run the webapp use the command:

```
streamlit run <path to folder>/UI.py
```
