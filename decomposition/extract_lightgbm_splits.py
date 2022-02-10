"""
Read docs here https://github.com/microsoft/LightGBM/blob/0688f471fc303af816cd6564107f6d720ebb99ec/python-package/lightgbm/basic.py#L3394
"""
from pathlib import Path
import pandas as pd
import lightgbm as lgb

df = pd.read_csv("decomposition/datasets/numerical/Breast_cancer_data.csv")
X = df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

lgbc = lgb.LGBMClassifier()

print('Starting training...')
lgbc.fit(X_train, y_train)
y_pred = lgbc.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print('Dumped model:')
trees_df = lgbc.booster_.trees_to_dataframe()
nodes_df = trees_df[trees_df['split_feature'].isna() == False] # if feature is None then it is a leaf
feature_threshold_df = nodes_df[['split_feature', 'threshold']].reset_index(drop=True)
nodes_num = len(feature_threshold_df)
feature_generator_list = []
for i in range(nodes_num):
    feature_generator_list.append(lambda df: df[f'{feature_threshold_df.iloc[i][0]}'] <= feature_threshold_df.iloc[i][1])

generated_df = pd.DataFrame()
for i, generator in enumerate(feature_generator_list):
    generated_df[f'{i}'] = generator(df)

print(generated_df)