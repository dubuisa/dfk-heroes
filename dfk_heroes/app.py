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

def main():
    st.title('Predicting DFK Heroes price using Artificial Intelligence')
    st.subheader('Created by: Antoine Dubuis & Karim Steiner')
    @st.cache
    def load_data():
        pipe = joblib.load(os.path.join(Path(__file__).parent, 'data/model.joblib'))
        df = pd.read_csv(os.path.join(Path(__file__).parent, 'data/tavern_data.csv'))
        return pipe, df
    # Create a text element and let the reader know the data is loading.
    
    data_load_state = st.text('Loading data...')
    
    pipe, df = load_data()
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    
    hero_id = st.number_input('Enter your hero id!', min_value=0)
    
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

        return {
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
        }
    
    def predict(hero_id):
        feature = hero_to_feature(hero_id)
        return feature, pipe.predict(pd.DataFrame.from_records([feature]))[0]
    
    if st.button('Predict price'):
        feature, price = predict(hero_id)
        st.json(json.dumps(feature))
        st.write(f'Hero values: {price:.3f} JEWEL')

    
    
if __name__== '__main__':
    main()


