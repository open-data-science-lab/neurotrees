"""
Read docs here https://github.com/microsoft/LightGBM/blob/0688f471fc303af816cd6564107f6d720ebb99ec/python-package/lightgbm/basic.py#L3394
"""
import pandas as pd

from featuregenerator_lgbm import FeatureGeneratorLGBM

df = pd.read_csv("decomposition/datasets/numerical/Breast_cancer_data.csv")
X = df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

feature_generator = FeatureGeneratorLGBM()
feature_generator.fit(X, y)
generated_df = feature_generator.transform(X)

print(generated_df)