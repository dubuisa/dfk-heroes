
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as pl
import json
from model import ClassRankExtractor, DateFeaturesExtractor, ToCategory
import utils
from PIL import Image
from custom_shap import get_shap_values, custom_waterfall
import base64
import plots


def main():
    st.set_page_config(
        page_title="DFK Heroes Price Prediction",
        page_icon=Image.open(os.path.join(Path(__file__).parent, 'data/favicon.png')),
        layout="centered",
        initial_sidebar_state="expanded",
    )
    def predict(hero_id):
        feature = utils.hero_to_feature(hero_id)
        return feature, pipe.predict(feature)[0]
    
    @st.cache(allow_output_mutation=True)
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df_cv = pd.read_csv(os.path.join(Path(__file__).parent, 'data/cross_validation.csv'))
        df_price_impact = pd.read_csv(os.path.join(Path(__file__).parent, 'data/jewel_price_impact.csv'))
        explainer = joblib.load(os.path.join(Path(__file__).parent, 'data/explainer.joblib'))
        return pipe, df_cv, df_price_impact, explainer
    pipe, df_cv, df_price_impact,  explainer = load_data()

    #warmup explainer
    get_shap_values(explainer, pipe[:-1].transform(predict(0)[0]))

    jewel = base64.b64encode(open(os.path.join(Path(__file__).parent, 'data/favicon.png'), "rb").read()).decode()
    st.set_option('deprecation.showPyplotGlobalUse', False)
   
        
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
            font-size : 1.7rem !important;;
        }
        
        p, label, li, ul, li code {
            font-size : 1.3rem !important;
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
    st.subheader('Created by Mrmarx & Gambarim')
    st.markdown("""
    Welcome to DFK-Hero Price Prediction (HPP), our submission for the Data Visualisation Contest.
    
We want to show that Data can be visualised in a more informative way thanks to AI.

HPP can decompose the price of a hero into multiple reasons. For this we used the tavern auction starting from the 23 of January 2022 until the 28 of January 2022 in order to train an accurate model with roughly 8000 sales.            
    """)
    
    avg_price = explainer.expected_value
    st.markdown(f"Right now, on average, a hero price is worth {avg_price:.2f} JEWEL. Nonetheless we know that some heroes are cheaper and somes heroes are WAY higher than that. Here is a plot showing price distribution (red line is the average):")
    st.altair_chart(plots.price_distribution(df_cv, avg_price,  width=700))
    
    st.markdown("""
    Price Explanation
    ---------------------------
    
    But what drives the price so much ? This is where AI comes in.
    In 2022, novel techniques in Artificial Intelligence allows us to not only predict but also *understand* what drives the predictions.

    Heroes price in Defi Kingdoms is mainly driven by the following features:

    """)
    st.altair_chart(plots.price_explanation(df_price_impact, width=700))
    st.markdown("""
                As you can see, the rarity, the profession and the main class of the hero are the top 3 price drivers.
                """
    )
    
    st.markdown("""
    Advanced Analytics
    ---------------------------
    
    Data tends to form clusters of similar attributes and behaviours. These clusters typically evolves over time but contains a huge amount of insights with respect to the data.

    Below is an interactive T-SNE cluster plot. If you drag your mouse whilst holding the left mouse button, characteristics of all the features are automatically shown. Herewith one can get meaningfull insights of different groups in for example: we can see that the mining profession behaves very differently from the other professions and forms clusters of high prices.

    Furthermore, it is binned into 5 equal groups, for better interpretability. Each seperate color stands for a specific group, where green signals the top 20% most expensive of (predicted) hero price in Defi Kingdoms and red signals the bottom 20% of heroes with respect to price.
    """)
    st.altair_chart(plots.advanced_analytics(df_cv, width=600))
    
    
    st.markdown("""
    Predict the price of a hero
    ---------------------------
    Want to try with your own hero, estimate its price and understand it?
    
    Simply enter your `hero_id` below to compute it.
    """)

    hero_id = st.number_input('hero_id', min_value=0)
    
    if st.button('Predict price'):
        c = st.container()
        
        feature, price = predict(hero_id)
        c.json(json.dumps(utils.hero_to_display(feature.copy(deep=True))))
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
    
    st.markdown(utils.get_dataset_description())
    st.markdown(utils.get_futures_evolutions())
if __name__== '__main__':
    main()


