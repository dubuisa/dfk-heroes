import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import json
import os
from pathlib import Path

from model import ClassRankExtractor, DateFeaturesExtractor, ToCategory
from hero import hero
from utils import get_dataset_description
from PIL import Image

def main():
    jewel = Image.open(os.path.join(Path(__file__).parent, 'data/favicon.png'))

    
    st.set_page_config(
     page_title="DFK Heroes Price Prediction",
     page_icon=jewel,
     layout="wide",
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
        
        .fullScreenFrame > div {
           display: flex;
           justify-content: center;
        }
        </style>

        """,
            unsafe_allow_html=True,
        )
    st.image(os.path.join(Path(__file__).parent, 'data/logo.png'))
    st.subheader('Created by: Antoine Dubuis & Karim Steiner')
    st.markdown('This app aims to answer the question: how valuable is my DFK Heroes using AI')
    
    
    @st.cache
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df = pd.read_csv(os.path.join(Path(__file__).parent, 'data/tavern_data.csv'))
        return pipe, df
    # Create a text element and let the reader know the data is loading.
    
    st.markdown(get_dataset_description())
    pipe, df = load_data()
    
    st.markdown("""
    Predict the price of a hero
    ---------------------------
    
    You can simply predict an hero's price using it's `hero_id`.
    """)
    hero_id = st.number_input('hero_id', min_value=0)
    
    def hero_to_feature(hero_id, rpc='https://api.harmony.one/'):
        h = hero.get_hero(hero_id, rpc)
        h = hero.human_readable_hero(h)
        mapping = {
            'strength' : 'STR',
            'agility': 'AGI',
            'intelligence' : 'INT',
            'wisdom' : 'WIS',
            'luck' : 'LCK',
            'vitality' : 'VIT',
            'endurance': 'END',
            'dexterity': 'DEX'
        }

        return pd.DataFrame.from_records([{
                    'rarity': h['info']['rarity'],
                    'generation': h['info']['generation'] ,
                    'mainClass': h['info']['class'].capitalize(),
                    'subClass': h['info']['subClass'].capitalize(),
                    'statBoost1': mapping[h['info']['statGenes']['statBoost1']],
                    'statBoost2': mapping[h['info']['statGenes']['statBoost2']],
                    'profession': h['info']['statGenes']['profession'],
                    'summons': h['summoningInfo']['summons'],
                    'maxSummons': h['summoningInfo']['maxSummons'],
                    'timeStamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
    
    def predict(hero_id):
        feature = hero_to_feature(hero_id)
        return feature, pipe.predict(feature)[0]
    
    if st.button('Predict price'):
        feature, price = predict(hero_id)
        st.dataframe(feature)
        st.write('Hero price:')
        col1, col2, _ = st.columns([1, 1, 20])
        with col1: 
            st.write(f'{price:.3f}')
        with col2:
            st.image(jewel, width=24)
    
    
    
if __name__== '__main__':
    main()


