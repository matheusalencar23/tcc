from matplotlib import lines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from helpers import binary_to_integer, binary_to_real


arr = [0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,1,0
,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0
,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0
,1,1,0,1,1,0,0]


data = pd.read_csv('./data.csv')
speed = np.asarray(dados.iloc[:, 0])
temperature = np.asarray(data.iloc[:, 1])
fill = np.asarray(data.iloc[:, 2])
thickness = np.asarray(data.iloc[:, 3])
orientation = np.asarray(data.iloc[:, 4])
tensile = np.asarray(data.iloc[:, 5])

x = np.c_[speed, temperature, fill, thickness, orientation]
y = tensile
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=1)

def regr(ind, x, y):
    learning_rate_init = binary_to_real(ind[:25])
    beta_1 = binary_to_real(ind[25:50])
    beta_2 = binary_to_real(ind[50:75])
    epsilon = binary_to_real(ind[75:100])
    hidden_layer_sizes = (
        binary_to_integer(ind[100:106]),
        binary_to_integer(ind[106:112]),
        binary_to_integer(ind[112:]))
    print(learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes)
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=1000, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, early_stopping=False).fit(x, y)
    return regr


ind = np.asarray(arr)
regr_train = regr(ind, x_train, y_train)
regr_test = regr(ind, x_test, y_test)
pred_train = regr_train.predict(x_train)
pred_test = regr_test.predict(x_test)
print('R^2 train: {}'.format(regr_test.score(x_train, y_train)))
print('MSE train: {}'.format(mean_squared_error(pred_train, y_train)))
print('EVS train: {}'.format(explained_variance_score(pred_train, y_train)))
print('R^2 test: {}'.format(regr_test.score(x_test, y_test)))
print('MSE test: {}'.format(mean_squared_error(pred_test, y_test)))
print('EVS test: {}'.format(explained_variance_score(pred_test, y_test)))

tamanho = len(regr_train.loss_curve_) if (len(regr_train.loss_curve_) < len(
    regr_test.loss_curve_)) else len(regr_test.loss_curve_)

plt.plot(regr_train.loss_curve_[:tamanho - 1],
         'g--', linewidth=1, label="Train")
plt.plot(regr_test.loss_curve_[:tamanho - 1],
         'r--', linewidth=1, label="Test")
plt.xlabel('Generation', fontsize=16)
plt.ylabel('Lost', fontsize=16)
plt.grid()
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
