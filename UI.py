import streamlit as st
import numpy as np
import pandas as pd
import time

st.title('Wine recommendation Decision Support System')

st.write("**Short introduction to the IDSS**")
user_input = st.text_input('Describe your favorite wine')
price_range = st.slider(
    'What is your price range ?',
    4.0, 3300.0, (0.0, 3300.0))
st.write('Price range:', price_range)
if st.button('Run'):    
    result = pd.DataFrame(
        np.random.randn(10, 5),
        columns=('col %d' % i for i in range(5)))
    my_bar = st.progress(0)
    with st.spinner('Running...'):
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
    st.success('Done!')
    st.write('Here is a collection of similar wines:')
    st.table(result)