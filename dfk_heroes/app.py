import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as pl

from model import ClassRankExtractor, DateFeaturesExtractor, ToCategory
from utils import get_dataset_description, hero_to_feature, hero_to_display
from PIL import Image
from custom_shap import get_shap_values, custom_waterfall
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    jewel = Image.open(os.path.join(Path(__file__).parent, 'data/favicon.png'))

    
    st.set_page_config(
     page_title="DFK Heroes Price Prediction",
     page_icon=jewel,
     layout="centered",
     initial_sidebar_state="expanded",
 )
        
    st.markdown(
    """
            <style>
        @font-face {
        font-family: 'Lora';
        font-weight: 400;
        src: local('Lora Regular'), local('Lora-Regular'), url(https://fonts.gstatic.com/s/lora/v14/0QIvMX1D_JOuMwf7I-NP.woff2) format('woff2');
        unicode-range: U+0460-052F, U+1C80-1C88, U+20B4, U+2DE0-2DFF, U+A640-A69F, U+FE2E-FE2F;
        }

        h1, h2, h3, h4 [class*="css"]  {
            font-family: 'Lora';
            color: #FBE375
        }
        
        
        </style>

        """,
            unsafe_allow_html=True,
        )
    st.image(os.path.join(Path(__file__).parent, 'data/logo.png'))
    st.subheader('Created by: Mrmarx & Gambarim')
    st.markdown('This app aims to answer the question: how valuable is my DFK Heroes using AI')
    
    
    @st.cache(allow_output_mutation=True)
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df = pd.read_csv(os.path.join(Path(__file__).parent, 'data/tavern_data.csv'))
        explainer = joblib.load(os.path.join(Path(__file__).parent, 'data/explainer.joblib'))
        return pipe, df, explainer
    # Create a text element and let the reader know the data is loading.
    
    st.markdown(get_dataset_description())
    pipe, df, explainer = load_data()
    
    st.markdown("""
    Predict the price of a hero
    ---------------------------
    
    You can simply predict an hero's price using it's `hero_id`.
    """)
    hero_id = st.number_input('hero_id', min_value=0)
    
    def predict(hero_id):
        feature = hero_to_feature(hero_id)
        return feature, pipe.predict(feature)[0]
    
    if st.button('Predict price'):
        feature, price = predict(hero_id)
        st.dataframe(hero_to_display(feature.copy(deep=True)))
        st.write('Hero price:')
        col1, col2, _ = st.columns([1.8, 1, 20])
        with col1: 
            st.write(f'{price:.3f}')
        with col2:
            st.image(jewel, width=24)
            
        shap_values = get_shap_values(explainer, pipe[:-1].transform(feature))
        custom_waterfall(explainer,shap_values, feature)
        st.pyplot(bbox_inches='tight')
        pl.clf()
    
    
if __name__== '__main__':
    main()


