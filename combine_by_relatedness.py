import numpy as np
import nltk
from nltk.corpus import stopwords
import requests

# download stopwords
nltk.download('stopwords')

test_words = ('wine','flavors','fruit','aromas','palate','acidity',
    'finish','tannins','drink','cherry','ripe','black','notes','red',
    'spice','rich','fresh','nose','oak','berry','dry','now','plum','soft',
    'fruits','blend','apple','crisp','blackberry','offers','sweet',
    'texture','white','while','shows','through','light','citrus','dark',
    'bright','vanilla','well','at','cabernet','very','more','full','pepper',
    'juicy','fruity','good','raspberry','firm','green','some','touch','peach',
    'lemon','character','chocolate','will','not','dried','balanced','pear',
    'out','structure','years','sauvignon','up','or','be','spicy','smooth',
    'all','pinot','made','concentrated','herb','tannic','also','note','just',
    'into','herbal','tart','there','like','wood','hint','flavor','licorice',
    'mineral','fine','bit','still','long','mouth','give','merlot','creamy',
    'there\'s','currant','so','clean','toast','balance','opens','age','alongside',
    'dense','orange','along','style','leather','lead','full-bodied','savory',
    'syrah','structured','aging','delicious','earthy','tobacco','over','hints',
    'tight','slightly')

test_counts = np.ones((len(test_words),))

def testFunction(words, counts):
    """ Function to test that matlab binding works correctly """
    print("The word " + str(words[0]) + " appears " + str(counts[0]) + " times.")

def remove_stopwords(words, counts):
    # Stop words are common english words
    stops = set(stopwords.words("english"))

    meaningful_words = []
    meaningful_counts = dict()
    # Save new struct
    for w, c in zip(words, counts):
        if not w in stops:
            meaningful_words.append(w)
            meaningful_counts[(w)] = c

    return meaningful_words, meaningful_counts

def get_relations(words, counts):

    # Stop words are common english words
    stops = set(stopwords.words("english"))

    # Get relations
    relations = dict()
    for word, count in zip(words, counts):

        # Filter common words
        if word in stops:
            continue

        # Get relations from web api
        url = 'http://api.conceptnet.io/c/en/' + str(word)

        try:
            obj = requests.get(url).json()

            # Save in struct
            nodes = [edge['end']['label'] for edge in obj['edges']]
            rels = [edge['rel']['label'] for edge in obj['edges']]

            relations[(word)] = (count, nodes, rels)
        except:
            print("There was an error for: \"" + str(word) + "\"")

    return relations

# Test in Python first
if __name__ == "__main__":
    rels = get_relations(test_words[0:9], test_counts[0:9])
    print(rels['wine'][0])



