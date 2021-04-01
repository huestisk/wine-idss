import pandas as pd
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

filename = 'preprocessing/vectorizer.pkl'

# Load data
df = pd.read_pickle("data/data.pkl")
descriptions = df["description"]

# Get features
try:
    # vectorizer = pickle.load(open(filename, 'rb'))
    X = vectorizer.transform(descriptions)
except:
    vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1,2))
    X = vectorizer.fit_transform(descriptions)

# Perform SVD
svd = TruncatedSVD(n_components=500, n_iter=5, random_state=42)
svd.fit(X)

# save models
pickle.dump(vectorizer, open('mdoels/vectorizer.pkl', 'wb'))
pickle.dump(svd, open('models/svd.pkl', 'wb'))

print('done!')
