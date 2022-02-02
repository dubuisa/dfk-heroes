from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from scipy import stats
import shap
from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np


import joblib
from pathlib import Path

import os
import warnings
warnings.filterwarnings("ignore")

class DateFeaturesExtractor(TransformerMixin, BaseEstimator): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        tmp = pd.to_datetime(X['timeStamp'])
        X['buyWeekDay'] = tmp.dt.weekday
        X['buyHour'] = tmp.dt.hour
        return X.drop(columns=['timeStamp'])

    
class ClassRankExtractor(TransformerMixin, BaseEstimator): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        basic = {k: 'Basic' for k in ['Priest', 'Warrior', 'Knight', 'Archer', 'Thief', 'Pirate', 'Monk', 'Wizard']}
        advanced = {k: 'Advanced' for k in ['Paladin', 'DarkKnight', 'Ninja', 'Summoner']}
        elite = {k: 'Elite' for k in ['Dragoon', 'Sage']}
        exalted = {'DreadKnight' : 'Exalted'}
        
        self.mapping = basic | advanced | elite | exalted
        
        return self
    
    def transform(self, X, y=None):
        X['classRank'] = X['mainClass'].map(self.mapping)
        return X
    
    
class ToCategory(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.types = {k: 'category' for k in X.select_dtypes(include=['object', 'category']).columns}
        return self
    
    def transform(self, X, y=None):  
        return X.astype(self.types)
    

def train(X_train, X_test, y_train, y_test): 
    hyper_parameter = {
        'objective': 'regression_l1',
        'metric': ['l2','l1'],
        'boosting': 'gbdt',
        'min_data_in_leaf':20,
        'verbose': 1,
        'learning_rate': 0.01,   
        'num_boost_round': 1_000,
        'early_stopping_rounds': 2000,
        'verbose_eval': 500
    }
    
    pipe = make_pipeline(
        DateFeaturesExtractor(),
        ClassRankExtractor(),
        ToCategory(),
        LGBMRegressor(**hyper_parameter)
    )
    
    X_train_transformed = pipe[:-1].fit_transform(X_train.copy(deep=True))
    X_test_transformed = pipe[:-1].transform(X_test.copy(deep=True))
    
    cat_features = list(X_train_transformed.columns[X_train_transformed.dtypes=="category"])
    pipe[-1].fit(X_train_transformed, y_train, eval_set=(X_test_transformed, y_test), categorical_feature=cat_features)
    return pipe


def remove_outlier(df):
    """
    Removes outliers that have a serious impact on the model
    """
    return df[(np.abs(stats.zscore(df['soldPrice'])) < 1)]


def to_x_y(df):
    return df.drop(columns=['soldPrice']), df['soldPrice']


def compute_tsne(df_cv, y_test,  shap_values, pipe):
    # Easy segmentation
    n_quant = 5


    df_cv['predictedSoldPrice'] = pipe[-1].predict(df_cv)
    shap_embedded = TSNE(n_components=2, perplexity=25,random_state=34).fit_transform(shap_values)
    df_cv['TSNE-1'] = shap_embedded[:,0]
    df_cv['TSNE-2'] = shap_embedded[:,1]
    df_cv = df_cv.merge(y_test, left_index=True, right_index=True)
    df_cv['soldPrice (Quintile)'] = pd.qcut(df_cv.soldPrice, n_quant, labels=['Bottom 20%','Middle 20% to 40%','Middle 40% to 60%','Middle 60% to 80%', 'Top 20%'])
    df_cv['Predicted soldPrice (Quintile)'] = pd.qcut(df_cv.predictedSoldPrice, n_quant, labels=['Bottom 20%','Middle 20% to 40%','Middle 40% to 60%','Middle 60% to 80%', 'Top 20%'])
    
    df_cv.to_csv(os.path.join(Path(__file__).parent, 'data/cross_validation.csv'))



if __name__ == "__main__":
    
    df = (
        pd.read_csv(os.path.join(Path(__file__).parent, 'data/tavern_data.csv'), decimal=',')
        .pipe(remove_outlier)
    )
    X, y = to_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    pipe = train(X_train, X_test, y_train, y_test)
    
    joblib.dump(pipe, os.path.join(Path(__file__).parent, 'data/model.joblib'))
    
    df_cv = pipe[:-1].transform(X_test.copy(deep=True))
    # compute SHAP values
    explainer = shap.TreeExplainer(pipe[-1])
    joblib.dump(explainer, os.path.join(Path(__file__).parent, 'data/explainer.joblib'))
    shap_values = explainer.shap_values(df_cv)
    compute_tsne(df_cv, y_test,  shap_values, pipe)