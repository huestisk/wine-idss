import numpy as np
import pandas as pd

import re
import pickle
import nltk
from nltk.corpus import stopwords

from combine_by_stem import replace_by_common_descriptors

MAX_IDX = 750

#  stopwords (common english words) # nltk.download('stopwords')
stops = set(stopwords.words("english"))

#  tokenizer # nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def bigrams_from_sentence(sentence):
    # Split into wordlist
    wordlist = sentence.split()
    wordlist = [word.lower() for word in wordlist]

    # Get bigrams
    bigrams = [wordlist[i]+" "+wordlist[i+1] for i in range(len(wordlist)-1)]

    # Filter wordlist
    filtered_wordlist = [replace_by_common_descriptors(word)
        for word in wordlist if word not in stops]
    
    # Replace with common descriptor, remove stopwords
    filtered_bigrams = []
    for bigram in bigrams:
        words = bigram.split()
        if words[0] in stops or words[1] in stops:
            continue

        stm = replace_by_common_descriptors(bigram)
        if " " not in stm:
            try:
                filtered_wordlist.remove(words[0])
            except ValueError:
                pass
            try:
                filtered_wordlist.remove(words[1])
            except ValueError:
                pass
            filtered_wordlist.append(stm)
        else:
            filtered_bigrams.append(stm)

    return filtered_wordlist, filtered_bigrams

def bigrams_from_descriptions(descriptions):
    wordlists = []
    bigrams = []
    for description in descriptions:
        if type(description)==str: # Only one sentence
            wordlist, bigram = bigrams_from_sentence(description)
            bigrams.append(bigram)
            wordlists.append(wordlist)
            continue
        
        wordlist = []
        bigram = []
        for sentence in description:
            wl, b = bigrams_from_sentence(sentence)
            bigram += b
            wordlist += wl
        
        bigrams.append(bigram)
        wordlists.append(wordlist)

    return wordlists, bigrams

def convert2dict(wordlists):
    dic = dict()
    for idx, wordlist in enumerate(wordlists):
        for word in wordlist:
            word = re.sub(" ","_", word)            
            if word in dic:
                dic[word].append(idx)
            else:
                dic[word] = [idx]
    return dic

def compute_tf_idf(words, num_reviews):
    """Computes the TF-IDF values"""
    freq_of_term = np.array([len(value) for value in words.values()])
    num_reviews_w_term = np.array([len(set(value)) for value in words.values()])

    term_freq = freq_of_term / sum(freq_of_term)
    inverse_document_freq = np.log(num_reviews / num_reviews_w_term)
    tf_idf = term_freq * inverse_document_freq

    output = np.empty((len(words),3),dtype=object)
    output[:,0] = np.array(list(words.keys()),dtype=object)
    output[:,1] = tf_idf
    output[:,2] = np.array(list(words.values()),dtype=object)

    return output

def get_top_features(tfidf_words, tfidf_bigrams):
    # combine
    tf_idf = np.concatenate((tfidf_words, tfidf_bigrams), axis=0)
    # sort
    ascending_idx = np.argsort(tf_idf[:, 1])
    sorted_tf_idf = tf_idf[ascending_idx[::-1]]
    return sorted_tf_idf[:MAX_IDX]

def convert2dfm(tf_idf, length):
    columns = tf_idf[:, 0]
    dfm = np.zeros((length, len(columns)), dtype=float)
    for idx in range(len(tf_idf)):
        for doc in tf_idf[idx, 2]:
            dfm[doc, idx] += 1
    return pd.DataFrame(data=dfm, columns=columns)

if __name__ == "__main__":

    # Load data
    df = pd.read_pickle("data/data.pkl")
    num_reviews = len(df)
    cols = df.columns
    descriptions = df["description"]

    """ Basic Preprocessing """
    try:
        with open('preprocessing/wordlists.p', 'rb') as fp:
            wordlists = pickle.load(fp)

        with open('preprocessing/bigrams.p', 'rb') as fp:
            bigrams = pickle.load(fp)
    except:
        print("Wordlists and Bigrams could not be loaded.")

        # tokenize
        raw_descriptions = [tokenizer.tokenize(description) for description in descriptions]

        # clean
        cleaned_descriptions = [[re.sub("[^a-zA-Z']"," ", sentence) for sentence in description]
            for description in raw_descriptions]

        # get bigrams & remove stopwords
        wordlists, bigrams = bigrams_from_descriptions(cleaned_descriptions)

        # save
        with open('preprocessing/wordlists.p', 'wb') as fp:
            pickle.dump(wordlists, fp)

        with open('preprocessing/bigrams.p', 'wb') as fp:
            pickle.dump(bigrams, fp)

    """ TF-IDF """
    try:
        with open('preprocessing/tfidf_words.p', 'rb') as fp:
            tfidf_words = pickle.load(fp)

        with open('preprocessing/tfidf_bigrams.p', 'rb') as fp:
            tfidf_bigrams = pickle.load(fp)
    except:
        tfidf_words = compute_tf_idf(convert2dict(wordlists), num_reviews)
        print("Done with TF-IDF for words.")

        tfidf_bigrams = compute_tf_idf(convert2dict(bigrams), num_reviews)
        print("Done with TF-IDF for bigrams.")

        # save
        with open('preprocessing/tfidf_words.p', 'wb') as fp:
            pickle.dump(tfidf_words, fp)

        with open('preprocessing/tfidf_bigrams.p', 'wb') as fp:
            pickle.dump(tfidf_bigrams, fp)

    """ Create Features """
    top_features = get_top_features(tfidf_words, tfidf_bigrams)
    dfm = convert2dfm(top_features, num_reviews)

    # save
    with open('data/dfm.pkl', 'wb') as fp:
        pickle.dump(dfm, fp)


