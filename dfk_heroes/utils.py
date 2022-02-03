from hero import hero
import pandas as pd
import numpy as np
import datetime

def get_dataset_description():
    return """
        Dataset
    ---------------------------

    **Tavern Dataset Characteristics:**  

        :Number of Instances: 8000 

        :Number of Attributes: 11 numeric/categorical predictive.

        :Attribute Information (in order):
            - id            Hero id
            - rarity        Rarity of the hero
            - generation    Generation of Hero
            - mainClass     MainClass of the Hero
            - subClass      SubClass of the Hero
            - statBoost1    First StatBoost
            - statBoost2    Second StatBoost
            - profession    Profession of the Hero
            - summons       Remaining Summons
            - maxSummons    Total Summons
            - soldPrice     Price sold in JEWEL
            - timeStamp     timestamp of the sale

        :Missing Attribute Values: None
        
        :Description: This dataset contains all heroes transaction from 2022-01-21 13:17:04 to 2022-01-28 00:01:13.
        
        :Origin: DeFi Kingdom GraphQL APIv5
        
    **Heroes Data:**
    
        :Origin Smart Contract data retrieved using 0rtis' DefiKingdoms toolbox 
        
        :Reference https://github.com/0rtis/dfk
        
    """
    
def get_futures_evolutions():
    return """
        Futures Works
    ---------------------------
    
    Because a masterpiece is never finished, here are a few ideas of what we would like to implement next: 
    
    - Provide an API
    - Periodically and automatically retrain model based on previous weeks' sales 
    - Simulate and calculate the impact of `level-up` and `summons` on the hero price
    - Use additionnal features (`level`, `xp`, `shiny`, `stats`, etc)
    - Additional charts
    - Support for `gen0`
    - Card display for hero
    """    
    
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
    
    remaining_summons = h['summoningInfo']['maxSummons']-h['summoningInfo']['summons']
    if remaining_summons < 0:
        remaining_summons = h['summoningInfo']['maxSummons']
        
    return pd.DataFrame.from_records([{
                'id': hero_id,
                'rarity': h['info']['rarity'],
                'generation': h['info']['generation'] ,
                'mainClass': h['info']['class'].capitalize(),
                'subClass': h['info']['subClass'].capitalize(),
                'statBoost1': mapping[h['info']['statGenes']['statBoost1']],
                'statBoost2': mapping[h['info']['statGenes']['statBoost2']],
                'profession': h['info']['statGenes']['profession'],
                'summons': remaining_summons,
                'maxSummons': h['summoningInfo']['maxSummons'],
                'timeStamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
def hero_to_display(feature):
    mapping = {
        0 : 'Monday',
        1 : 'Tuesday',
        2 : 'Wednesday',
        3 : 'Thursday',
        4 : 'Friday',
        5 : 'Saturday',
        6 : 'Sunday'
    }
    return feature.assign(buyWeekDay = lambda X: X['buyWeekDay'].map(mapping)).to_dict('records')[0]

def above_below(total, threshold):
    if total>threshold:
        return f'<code class=green>{total-threshold:.2f} JEWEL</code> above'
    22568
    return f'<code class=red>{threshold-total:.2f} JEWEL</code> below'
#ff0051

def plus_minus(val, extra_text=" JEWEL"):
    clz = "green"
    if val < 0:
        clz = "red"
    return f'<code class={clz}>{val:.2f}{extra_text}</code>'

def explain(df, top_n):
    tx = []
    for index, row in df.head(top_n).iterrows():
        tx.append(f"<li>{index} = {row['feature']}  =>  {plus_minus(row['impact'])}</li>")
    tx.append(f"<li>other features  => {plus_minus(df.iloc[top_n:]['impact'].sum())}</li>")
    return '\n'.join(tx)

def equation(df, top_n, total, avg_price):
    tx = [
        f"<code class=white>{avg_price:.2f} +</code>"
    ]
    for index, row in df.head(top_n).iterrows():
        tx.append(f"{plus_minus(row['impact'], extra_text='')} +")
    tx.append(f"{plus_minus(df.iloc[top_n:]['impact'].sum(), extra_text='')} = {total:.2f} JEWEL")
    return '\n'.join(tx)

def shap_to_text(shap_values, feature, avg_price, top_n = 3):
    shap_df = (pd.DataFrame(shap_values, columns = feature.columns)
               .T
               .rename(columns={0 : 'impact'})
               .sort_index()
               .merge(feature.T.rename(columns={0 : 'feature'}), right_index=True, left_index=True)
               .assign(abs_val = lambda X: np.abs(X['impact']))
               .sort_values('abs_val', ascending=False)[['feature', 'impact']]
               )
    total = shap_values.sum()+avg_price
    return f"""
        <div>
        <p>The predicted price is {total:.2f} JEWEL</p>
        <p>It is {above_below(total, avg_price)} the average hero price ({avg_price:.2f} JEWEL)</p>
        <p>This can be explained by:</p>
        <ul>
            {explain(shap_df, top_n)}
        <ul>
        <p>Thus, the total predicted value can be computed as follow:</p>
        <p>
            {equation(shap_df, top_n, total, avg_price)}
        <p>
        </br>
        <p>The plot below explain this prediction in details:</p>
        </div>
    """