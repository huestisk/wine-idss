from nltk.stem import PorterStemmer 

stemmer = PorterStemmer()

dic = {}

dic[stemmer.stem("darkfruit")] = "dark fruit, blackcurrant, blueberry, blue fruits, black cherry, black fruit, blackberry, cassis, juniper, mulberry, plum".split(', ')
dic[stemmer.stem("redfruit")] = "red fruit, red cherry, red berry, red currant, boysenberry, cranberry,raspberry, strawberry".split(', ')
dic[stemmer.stem("ripefruit")] = "ripe berry, apricot, fruit cake, fig, jam, overripe, porty,prune, raisin, shrivelled, ripe fruit".split(', ')
dic[stemmer.stem("savoury")] = "savoury, sea salt, barbeque sauce, beef stock, gamey, iodine,meat, oyster, pancetta, salty, salami, sea spray, seaweed, soysauce, steak, vegemite".split(', ')
dic[stemmer.stem("nuts")] = "nuts, almond, chestnut, nutty".split(', ')
dic[stemmer.stem("sage")] = "herbal, bay leaf, herbaceous, rosemary, tea, thyme".split(', ')
dic[stemmer.stem("weedy")] = "green, dill, grass, capsicum, sappy, shaded".split(', ')
dic[stemmer.stem("oaky")] = "oaky, cigar box, coffee beans, burnt, butterscotch, caramel, cedar,chocolate, cocoa, coconut, mocha, tarry, vanilla, woody".split(', ')
dic[stemmer.stem("cookedveg")] = "cooked vegetable, canned green bean, sulphide, vegetable, eggplant".split(', ')
dic[stemmer.stem("zesty")] = "jaffa, orange, chinotto, rhubarb, zesty".split(', ')
dic[stemmer.stem("earthy")] = "earthy, forest floor, dirt, dust, fungal, mossy, muddy,mushroom, musk".split(', ')
dic[stemmer.stem("inorganic")] = ["minerality", "graphite", "petichor", "unctuous", "oily", "petroleum", "plastic" , "tar", "rubber", "diesel" ]

dic[stemmer.stem("weedy")] += ["stemmy", "stalky", "cat's pee", "asparagus", "green", "grassy", "eucalyptus", "jalapeno", "dill", "bell pepper", "gosseberry", "quince"]
dic[stemmer.stem("oaky")] += ["smoky", "charcoal", "sweet tobacco", "toasty"]
dic[stemmer.stem("flower")] = ["white flowers", "violet"]
dic[stemmer.stem("ripefruit")] += ["jammy", "ripe", "juicy"]
dic[stemmer.stem("spice")] = ["spicy", "musky", "bright", "pepper"]
dic[stemmer.stem("acid")] = ["astringent", "austere"]
dic[stemmer.stem("tannin")] = ["bitter", "harsh", "agressive"]

replacement_dic = {}

for root, words in dic.items():
  for word in words:
    stm = stemmer.stem(word)
    if stm not in replacement_dic:
      replacement_dic[stm] = root
    elif root == replacement_dic[stm]:
      continue
    else:
      raise Exception("Stem double mapped.")

def replace_by_common_descriptors(word):
  stm = stemmer.stem(word)
  if stm in replacement_dic:
    return replacement_dic[stm]
  else:
    return stm