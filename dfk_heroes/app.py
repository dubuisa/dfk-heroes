from dfk_heroes.plots import advanced_analytics
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as pl
import json
from model import ClassRankExtractor, DateFeaturesExtractor, ToCategory
from utils import get_dataset_description, hero_to_feature, hero_to_display
from PIL import Image
from custom_shap import get_shap_values, custom_waterfall
import base64
from plots import advanced_analytics
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    jewel = base64.b64encode(open(os.path.join(Path(__file__).parent, 'data/favicon.png'), "rb").read()).decode()
    
    st.set_page_config(
     page_title="DFK Heroes Price Prediction",
     page_icon=Image.open(os.path.join(Path(__file__).parent, 'data/favicon.png')),
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
        
        .big-font {
            font-size : 1.7rem;
        }
        
        code, p, label {
            font-size : 1.3rem;
        }
        
        .container {
            display: flex;
        }
        .logo-text {
            font-size : 1.7rem;
        }
        .logo-img {
            float:left;
            width: 42px;
            height: 42px;
        }
        
        </style>

        """,
            unsafe_allow_html=True,
        )
    st.image(os.path.join(Path(__file__).parent, 'data/logo.png'))
    st.subheader('Created by: Mrmarx & Gambarim')
    st.markdown("""
    Welcome to DFK-Hero Price Prediction (HPP), our submission for the Data Visualisation Contest.
We want to show that Data can be visualised in a more informative way thanks to AI.

HPP can decompose the price of a hero into multiple reasons. For this we used the tavern auction starting from the 23 of January 2022 until the 28 of January 2022 in order to train an accurate model with roughly 8000 sales.            
    """)
    
    
    @st.cache(allow_output_mutation=True)
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df_cv = pd.read_csv(os.path.join(Path(__file__).parent, 'data/cross_validation.csv'))
        explainer = joblib.load(os.path.join(Path(__file__).parent, 'data/explainer.joblib'))
        return pipe, df_cv, explainer
    # Create a text element and let the reader know the data is loading.
    
    st.markdown(get_dataset_description())
    pipe, df_cv, explainer = load_data()
    
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
        c = st.container()
        
        feature, price = predict(hero_id)
        c.json(json.dumps(hero_to_display(feature.copy(deep=True))))
        c.markdown(
            f"""
            <div class="container">
                <p class="logo-text">The predicted price is {price:.3f}</p>
                <img class="logo-img" src="data:image/png;base64,{jewel}">
            </div>
            """,
            unsafe_allow_html=True
        )
        shap_values = get_shap_values(explainer, pipe[:-1].transform(feature))
        print(shap_values)
        custom_waterfall(explainer,shap_values, feature)
        c.pyplot(bbox_inches='tight')
        
        
        pl.clf()
    st.markdown("""
    Advanced Analytics
    ---------------------------
    
    Data tends to form clusters of similar attributes and behaviours. These clusters typically evolves over time but contains a huge amount of insights with respect to the data.

    Below is an interactive T-SNE cluster plot. If you drag your mouse whilst holding the left mouse button, characteristics of all the features are automatically shown. Herewith one can get meaningfull insights of different groups in for example: we can see that the mining profession behaves very differently from the other professions and forms clusters of high prices.

    Furthermore, it is binned into 5 equal groups, for better interpretability. Each seperate color stands for a specific group, where green signals the top 20% most expensive of (predicted) hero price in Defi Kingdoms and red signals the bottom 20% of heroes with respect to price.
    """)
    
    

    st.altair_chart(advanced_analytics(df_cv, width=600)

    )
    
if __name__== '__main__':
    main()


