import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from featuregenerator_lgbm_sigm import FeatureGeneratorLGBMSigm
from featuregenerator_lgbm import FeatureGeneratorLGBM

df = pd.read_csv("./datasets/numerical/jungle_chess_2pcs_raw_endgame_complete.csv")
# X = df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
# y = df['diagnosis']
label = 'class'
X = df.drop(columns=[label], axis=1)
y = df[label]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

feature_generator = FeatureGeneratorLGBM()
feature_generator.fit(X_train, y_train)
generated_df_train = feature_generator.transform(X_train)
generated_df_test = feature_generator.transform(X_test)

feature_generator_sigm = FeatureGeneratorLGBMSigm()
feature_generator_sigm.fit(X_train, y_train)
generated_sigm_train = feature_generator_sigm.transform(X_train)
generated_sigm_test = feature_generator_sigm.transform(X_test)

lr = LogisticRegression(random_state=72).fit(X_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(48, 64, 32), max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=72).fit(X_train, y_train)
print('LogReg score on usual data:')
print(lr.score(X_test, y_test))
print('MLP score on usual data:')
print(mlp.score(X_test, y_test))

lr = LogisticRegression(random_state=72).fit(generated_df_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(48, 64, 32), max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=72).fit(generated_df_train, y_train)
print('LogReg score on generated from LightGBM features:')
print(lr.score(generated_df_test, y_test))
print('MLP score on generated from LightGBM features:')
print(mlp.score(generated_df_test, y_test))

lr = LogisticRegression(random_state=72).fit(generated_sigm_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(48, 64, 32), max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=72).fit(generated_sigm_train, y_train)
print('LogReg score on generated from LightGBM sigmoid-features:')
print(lr.score(generated_sigm_test, y_test))
print('MLP score on generated from LightGBM sigmoid-features:')
print(mlp.score(generated_sigm_test, y_test))