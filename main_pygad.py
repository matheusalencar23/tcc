import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, train_size=0.9, test_size=0.1, random_state=1)


def conversorBinarioReal(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[i] * (2 ** (- i - 1))
    if v > 0:
        return v
    else:
        return v + 0.0000000001


def conversorBinarioInteiro(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[len(binario) - i - 1] * 2**(i)
    if v > 0:
        return int(v)
    else:
        return int(1)


MAX_ITER_RN = 5000


def aptidao(x, i):
    learning_rate_init = conversorBinarioReal(x[:25])
    beta_1 = conversorBinarioReal(x[25:50])
    beta_2 = conversorBinarioReal(x[50:75])
    epsilon = conversorBinarioReal(x[75:100])
    hidden_layer_sizes = (
        conversorBinarioInteiro(x[100:106]),
        conversorBinarioInteiro(x[106:112]),
        conversorBinarioInteiro(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init,
                        max_iter=MAX_ITER_RN, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes).fit(x_treino, y_treino)
    score = regr.score(x_teste, y_teste)
    if score and score > 0:
        return score
    return 0


NUM_GERACOES = 1000
TAM_POP = 100
NUM_GENES = 118


def on_start(model):
    print('Algoritmo Genético Iniciado')
    print('Tamanho da população {}'.format(model.pop_size))


def on_fitness(model, aptidoes):
    print('Aptidões')
    print(aptidoes)
    print('\n')


def on_generation(model):
    print("Geração {}/{}".format(model.generations_completed, NUM_GERACOES))


def on_stop(model, aptidoesFinais):
    print('Algoritmo Genético Finalizado')


model = pygad.GA(num_generations=NUM_GERACOES, num_parents_mating=TAM_POP,
                 fitness_func=aptidao, sol_per_pop=TAM_POP,
                 num_genes=NUM_GENES, gene_type=int,
                 init_range_low=0, init_range_high=2,
                 parent_selection_type="tournament", K_tournament=3,
                 keep_parents=0, crossover_type="single_point",
                 crossover_probability=0.9, mutation_type="random", suppress_warnings=True,
                 mutation_probability=0.05, on_start=on_start, on_stop=on_stop,
                 on_generation=on_generation, on_fitness=on_fitness)
model.run()
solution, solution_fitness, solution_idx = model.best_solution()
print("Melhor indivíduo : {}".format(solution))
print("Aptidão do melhor indivíduo = {}".format(solution_fitness))
model.plot_fitness()
