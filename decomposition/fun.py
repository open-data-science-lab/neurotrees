import pandas as pd
from sklearn.linear_model import LogisticRegression
from featuregenerator_lgbm import FeatureGeneratorLGBM

df = pd.read_csv("decomposition/datasets/numerical/Breast_cancer_data.csv")
X = df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

feature_generator = FeatureGeneratorLGBM()
feature_generator.fit(X_train, y_train)
generated_df_train = feature_generator.transform(X_train)
generated_df_test = feature_generator.transform(X_test)

lr = LogisticRegression(random_state=72).fit(X_train, y_train)
print('LogReg score on usual data:')
print(lr.score(X_test, y_test))

lr.fit(generated_df_train, y_train)
print('LogReg score on generated from LightGBM features:')
print(lr.score(generated_df_test, y_test))