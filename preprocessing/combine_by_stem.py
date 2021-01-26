
from nltk.stem import PorterStemmer 

stemmer = PorterStemmer()

dic = {}

dic[stemmer.stem("darkfruit")] = "dark fruit, blackcurrant, blueberry, blue fruits, black cherry, black fruit, blackberry, cassis, juniper, mulberry, plum".split(', ')
dic[stemmer.stem("redfruit")] = "red fruit, red cherry, red berry, red currant, boysenberry, cranberry,raspberry, strawberry".split(', ')
dic[stemmer.stem("ripefruit")] = "ripe berry, apricot, fruit cake, fig, jam, overripe, porty,prune, raisin, shrivelled, ripe fruit".split(', ')
dic[stemmer.stem("savoury")] = "savoury, sea salt, barbeque sauce, beef stock, gamey, iodine,meat, oyster, pancetta, salty, salami, sea spray, seaweed, soysauce, steak, vegemite".split(', ')
dic[stemmer.stem("nuts")] = "nuts, almond, chestnut, nutty".split(', ')
dic[stemmer.stem("sage")] = "herbal, bay leaf, herbaceous, rosemary, tea, thyme, sage".split(', ')
dic[stemmer.stem("weedy")] = "green, dill, grass, capsicum, sappy, shaded, stalky, vegetal, weedy".split(', ')
dic[stemmer.stem("oaky")] = "oaky, cigar box, coffee beans, burnt, butterscotch, caramel, cedar,chocolate, cocoa, coconut, mocha, tarry, vanilla, woody".split(', ')
dic[stemmer.stem("cookedveg")] = "cooked vegetable, canned green bean, sulphide, vegetable, eggplant".split(', ')
dic[stemmer.stem("zesty")] = "jaffa, orange, chinotto, rhubarb, zesty".split(', ')
dic[stemmer.stem("earthy")] = "earthy, forest floor, dirt, dust, fungal, mossy, muddy,mushroom, musk".split(', ')
dic[stemmer.stem("inorganic")] = ["minerality", "graphite", "petichor", "unctuous", "oily", "petroleum", "plastic" , "tar", "rubber", "diesel", "smoky" ]
dic[stemmer.stem("weedy")] += ["stemmy", "stalky", "vegetal", "cat's pee", "asparagus", "green", "grassy", "sage", "eucalyptus", "jalapeno", "dill", "bell pepper", "gosseberry", "quince"]
dic[stemmer.stem("oaky")] += ["smoky", "charcoal", "sweet tobacco", "toasty"]
dic[stemmer.stem("flower")] = ["white flowers", "violet"]
dic[stemmer.stem("ripefruit")] += ["jammy", "ripe", "juicy"]
dic[stemmer.stem("spice")] = ["spicy", "musky", "bright", "pepper"]
dic[stemmer.stem("acid")] = ["bright", "astringent", "austere"]
dic[stemmer.stem("tannin")] = ["bitter", "harsh", "agressive"]

for name, l in dic.items():
    dic[name] = [stemmer.stem(e) for e in l]

"""pause"""

def replace_by_common_descriptors(words):

  descriptors = [] 
  for word in words:
    # convert to stem
    descriptor = stemmer.stem(word)
    # combine by stem
    for name, l in dic.items():
      if descriptor in l:
        descriptor = name
    # append
    descriptors.append(descriptor)

  return descriptors