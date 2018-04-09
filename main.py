import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def clean_data(data):
    cols_proto = ['GrLivArea', 'LotArea', '1stFlrSF', 'TotalBsmtSF', 'BsmtUnfSF',
                  'YearBuilt', 'GarageArea', 'MoSold', 'YearRemodAdd', 'OverallQual']

    data = data.fillna(data.std())
    return data[cols_proto]
data_train = pd.read_csv('train.csv')

data_Y = data_train[['SalePrice']]
data_X = data_train.drop(['SalePrice'], axis=1)


data_X = clean_data(data_X)

rng = np.random.RandomState(42)

forest = IsolationForest(max_samples=100, random_state=rng)
line_reg = LinearRegression()

scores = cross_val_score(line_reg, data_X, data_Y, cv=10)

line_reg.fit(data_X, data_Y)
print(scores.mean())

X_predict = pd.read_csv('test.csv')

X_predict_clean = clean_data(X_predict)

print(X_predict.describe())

Y_pred = line_reg.predict(X_predict_clean)

data = pd.DataFrame({'Id':X_predict.Id, 'SalePrice': np.reshape(Y_pred, (-1))})

data.set_index('Id').to_csv('my_test.csv')
