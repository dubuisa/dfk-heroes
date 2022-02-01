def get_dataset_description():
    return """
        Tavern Data dataset
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