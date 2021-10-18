#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Based Product Recommendation System using Flask and Heroku


import warnings
warnings.filterwarnings('ignore')


import pickle as pkl
from nltk.tokenize import word_tokenize
import xgboost



data  = pkl.load(open('dataset/data.pkl','rb'))
xgbc        = pkl.load(open('models/XGBoost.pkl','rb'))
tfidf      = pkl.load(open('models/tfidf_features.pkl','rb'))
user_recom = pkl.load(open('models/user_recommendation.pkl','rb'))


def sentiment(recom_prod):
    df = data[data.name.isin(recom_prod)]
    tfidf_features = tfidf.transform(df['text'])
    pred_data = xgbc.predict(tfidf_features)
    predictions = [round(value) for value in pred_data]
    df['predict'] = predictions
    output_data = df[df['predict']==1][['name',    'brand', 'categories']].drop_duplicates()[:5].reset_index(drop=True)

    return output_data

def recommendation(user_input):
    recom = user_recom.loc[user_input.lower()].sort_values(ascending=False)[0:20].index
    return recom
