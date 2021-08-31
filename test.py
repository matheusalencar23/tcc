import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.9, random_state=1)

def regr(x, y):
    regr = MLPRegressor(random_state=1, learning_rate_init=0.0130024254322052, shuffle=True,
                        max_iter=100, beta_1=0.7986843287944794, beta_2=0.6038900017738342,
                        epsilon=0.17187657952308655, solver='adam', activation='relu', 
                        learning_rate='constant', hidden_layer_sizes=(52, 30, 59), 
                        n_iter_no_change=10, tol=0.00001, early_stopping=True, 
                        validation_fraction=0.1).fit(x, y)
    return regr


regr_treino = regr(x_treino, y_treino)
regr_teste = regr(x_teste, y_teste)
pred_treino = regr_treino.predict(x_treino)
pred_teste = regr_teste.predict(x_teste)
print('R^2 treino: {}'.format(regr_teste.score(x_treino, y_treino)))
print('MSE treino: {}'.format(mean_squared_error(pred_treino, y_treino)))
print('R^2 teste: {}'.format(regr_teste.score(x_teste, y_teste)))
print('MSE teste: {}'.format(mean_squared_error(pred_teste, y_teste)))

plt.subplot(1, 2, 1)
plt.plot(regr_treino.loss_curve_, 'r')
plt.plot(regr_teste.loss_curve_, 'g')
plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 60, 100), np.linspace(0, 60, 100), 'b')
plt.plot(y_treino, pred_treino, 'go')
plt.plot(y_teste, pred_teste, 'ro')
plt.show()
