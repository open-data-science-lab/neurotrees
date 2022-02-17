from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import lightgbm as lgb
import math

def sigmoid(x: pd.Series):
  return pd.Series([1 / (1 + math.exp(-elem)) for elem in x.to_list()])

class FeatureGeneratorLGBMSigm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_generators_list = []
        self.lgbc = lgb.LGBMClassifier()

    def fit(self, X, y):
        self.lgbc.fit(X, y)
        trees_df = self.lgbc.booster_.trees_to_dataframe()
        nodes_df = trees_df[trees_df['split_feature'].isna() == False] # if feature is None then it is a leaf
        feature_threshold_df = nodes_df[['split_feature', 'threshold']].reset_index(drop=True)
        nodes_num = len(feature_threshold_df)
        for i in range(nodes_num):
            self.feature_generators_list.append(lambda df: sigmoid(feature_threshold_df.iloc[i][1] - df[f'{feature_threshold_df.iloc[i][0]}']))
        return self

    def transform(self, X: pd.DataFrame):
        generated_df = pd.DataFrame()
        for i, generator in enumerate(self.feature_generators_list):
            generated_df[f'{i}'] = generator(X)
        return generated_df

    def get_lgbm_model(self):
        return self.lgbc