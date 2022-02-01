from hero import hero
import pandas as pd
import datetime

def get_dataset_description():
    return """
        Tavern dataset
    ---------------------------

    **Dataset Characteristics:**  

        :Number of Instances: 8000 

        :Number of Attributes: 11 numeric/categorical predictive. soldPrice is usually the target.

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