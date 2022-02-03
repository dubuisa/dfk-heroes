
from audioop import avg
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
        hero = utils.hero_to_feature(hero_id)
        feature = pipe[:-1].transform(hero.copy(deep=True))
        return feature, pipe[-1].predict(feature)[0]
    
    @st.cache(allow_output_mutation=True)
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df_cv = pd.read_csv(os.path.join(Path(__file__).parent, 'data/cross_validation.csv'))
        df_price_impact = pd.read_csv(os.path.join(Path(__file__).parent, 'data/jewel_price_impact.csv'))
        explainer = joblib.load(os.path.join(Path(__file__).parent, 'data/explainer.joblib'))
        return pipe, df_cv, df_price_impact, explainer
    pipe, df_cv, df_price_impact,  explainer = load_data()

    #warmup explainer
    get_shap_values(explainer, predict(0)[0])

    avg_price = explainer.expected_value
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
        
        p, label, li, ul, li code {
            font-size : 1.3rem !important;
        }
        
        .container {
            display: flex;
        }
        .red{
            color : #ff0051 !important;
        }
        .green {
            color : #19c558 !important;
        }
        .white {
            color : #ffffff !important;
        }
        </style>

        """,
            unsafe_allow_html=True,
        )
    st.image(os.path.join(Path(__file__).parent, 'data/logo.png'))
    st.subheader('Created by Dubuisa & Gambarim')
    st.markdown("""
    Welcome to `DFK-Heroes Price Prediction`, our submission for the Data Visualisation Contest.
    
We want to show that data can be visualised in a more informative way thanks to AI.

`DFK-Heroes Price Prediction` can decompose the price of a hero into multiple components. To do this, we used the tavern auction starting from the 23rd January 2022 until the 28th January 2022 in order to train an accurate model with roughly 8000 sales.

Disclamer: As extreme prices and `gen0` were under represented within the dataset, this AI works best for heroes that are valued less than 500 JEWEL. 
    """)
    
    st.markdown("""
    Predict the price of a hero
    ---------------------------
    Want to try with your own hero, estimate its price and understand it?
    
    Simply enter your `hero_id` below to compute it.
    """)

    hero_id = st.number_input('hero_id', min_value=0)
    
    if st.button('Predict price'):
        c = st.container()
        
        feature, _ = predict(hero_id)
        c.json(json.dumps(utils.hero_to_display(feature.copy(deep=True))))
        shap_values = get_shap_values(explainer, feature)
        c.markdown(utils.shap_to_text(shap_values, feature, avg_price, jewel), unsafe_allow_html=True)
        custom_waterfall(explainer,shap_values, feature)
        c.pyplot(bbox_inches='tight')
        
        pl.clf()
    
    st.markdown(f"""
        How does it work?
        ---------------------------
        
        Right now, on average, a hero price is worth {avg_price:.2f} JEWEL. Nonetheless we know that some heroes are cheaper and somes heroes are WAY more expensive than that. Here is a plot showing price distribution (red line is the average):
        
                """)
    st.altair_chart(plots.price_distribution(df_cv, avg_price,  width=700))
    
    st.markdown("""
    Price Explanation
    ---------------------------
    
    So why does the price vary so much ? This is where AI comes in.
    In 2022, novel techniques in Artificial Intelligence allows us not only to predict but also *understand* what drives the predictions.

    Heroes price in Defi Kingdoms is mainly driven by the following features:

    """)
    st.altair_chart(plots.price_explanation(df_price_impact, width=700))
    st.markdown("""
                As you can see, the `rarity`, the `profession` and the `class rank` (basic, advanced, elite or exalted) of the hero are the top 3 price drivers.
                
                Interestingly enough, AI finds out that depending on the time of the day you can get more or less JEWEL for your hero too!
                """
    )
    
    st.markdown("""
    Advanced Analytics
    ---------------------------
    
    Data tends to form clusters of similar attributes and behaviours. These clusters typically evolve over time but contain a huge amount of insights with respect to the data.

    Below is an interactive t-SNE cluster plot. If you drag your mouse whilst holding the left mouse button, characteristics of all the features are automatically shown. Herewith one can get meaningful insights of different groups. 
    
    For example: we can see that the mining profession behaves very differently from the other professions and forms clusters of high prices.

    Furthermore, it is binned into 5 equal groups, for better interpretability. Each seperate color stands for a specific group, where green signals the top 20% most expensive of (predicted) hero price in DeFi Kingdoms and red signals the bottom 20% of heroes with respect to price.
    """)
    st.altair_chart(plots.advanced_analytics(df_cv, width=600))
    
    st.markdown(utils.get_dataset_description())
    st.markdown(utils.get_futures_evolutions())
    st.markdown("""
    Source code
    ---------------------------
           
       https://github.com/dubuisa/dfk-heroes         
    """)
if __name__== '__main__':
    main()


