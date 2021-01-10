import streamlit as st
import numpy as np
import pandas as pd
import time

from recommend_wine import WineRecommender

# Wine Recommender Class
data = pd.read_csv('data/winemag-data-130k-v2.csv')
wine_recommender = WineRecommender(data)

# Streamlit app
st.title('Wine Decision Support System')
st.write("**Enter a description of a wine and we will suggest you wines that you may like.**")
user_input = st.text_input('Describe your favorite wine')

_price_range = np.unique(np.round(np.logspace(-1, np.log10(3300), 200)))
price_range = st.select_slider(
    'What is your price range ?',
    options=list(_price_range),
    value=(_price_range[0], _price_range[-1]))

st.write('Price range:', price_range)

if st.button('Run'):

    wine_recommender.set_input_text(user_input)
    wine_recommender.set_price_range(price_range)

    my_bar = st.progress(0)
    with st.spinner('Running...'):
        variety, province, result = wine_recommender.recommend()

    st.success('Done!')
    st.write('Looks like you would most like ' + variety + '\n and other wines from ' + province + '.')
    st.write('Here is a collection of similar wines:')
    
    st.table(result.assign(hack='').set_index('hack'))
    # st.table(result)