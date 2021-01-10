import streamlit as st
import numpy as np
import pandas as pd
import time

from recommend_wine import WineRecommender

# Wine Recommender Class
data = pd.read_csv('data/winemag-data-130k-v2.csv')
wine_recommender = WineRecommender(data)

# Streamlit app
st.title('Wine recommendation Decision Support System')
st.write("**Short introduction to the IDSS**")
user_input = st.text_input('Describe your favorite wine')

_price_range = np.unique(np.round(np.logspace(-1, np.log10(3300), 200)))
price_range = st.select_slider(
    'What is your price range ?',
    options=list(_price_range),
    value=(_price_range[0], _price_range[-1]))
# price_range = st.slider(
#     'What is your price range ?',
    # 4.0, 3300.0, (0.0, 3300.0))
st.write('Price range:', price_range)

if st.button('Run'):

    wine_recommender.set_input_text(user_input)
    wine_recommender.set_price_range(price_range)

    my_bar = st.progress(0)
    with st.spinner('Running...'):
        result = wine_recommender.recommend()

    st.success('Done!')
    st.write('Here is a collection of similar wines:')
    
    st.table(result.assign(hack='').set_index('hack'))
    # st.table(result)