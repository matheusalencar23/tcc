import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from geneticalgorithm import geneticalgorithm as ga
import random as rd

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=1)

def conversorBinarioReal(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[i] * (2 ** (- i - 1))
    return v + 0.000000001

def conversorBinarioInteiro(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[len(binario) - i - 1] * 2**(i)
    if v > 0:
        return int(v)
    else:
        return int(1)

def aptidao(x):
    learning_rate_init = conversorBinarioReal(x[18:43])
    momentum = conversorBinarioReal(x[43:])
    hidden_layer_sizes = (
        conversorBinarioInteiro(x[:6]),
        conversorBinarioInteiro(x[6:12]), 
        conversorBinarioInteiro(x[12:18]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init,
                        max_iter=10000, momentum=momentum,
                        solver='adam', activation='relu',
                        hidden_layer_sizes=hidden_layer_sizes).fit(x_treino, y_treino)
    score = regr.score(x_teste, y_teste)
    if score and score > 0:
        return -score
    else:
        return 0

algorithm_param = {'max_num_iteration': 500,
                   'population_size': 10,
                   'mutation_probability': 0.05,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.9,
                   'parents_portion': 0.3,
                   'crossover_type': 'two_point',
                   'max_iteration_without_improv': None}

pop_i = np.array([[0, 1]]*68)

model = ga(function=aptidao, dimension=68, function_timeout=600,
           variable_type='int', variable_boundaries=pop_i, algorithm_parameters=algorithm_param)
model.run()

