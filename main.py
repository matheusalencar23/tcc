import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad
import os
from datetime import datetime
import csv
from helpers import binary_to_integer, binary_to_real

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(ROOT_DIR, 'data.csv')

MAX_ITER_NN = 100
NUM_GENERATIONS = 1000
NUM_GENES = 118
TRAINING_SIZE = 0.7
POP_SIZE = [100, 500, 1000]
NUM_EXEC = 10

data = pd.read_csv(FILE_DIR)
speed = np.asarray(data.iloc[:, 0])
temperature = np.asarray(data.iloc[:, 1])
fill = np.asarray(data.iloc[:, 2])
thickness = np.asarray(data.iloc[:, 3])
orientation = np.asarray(data.iloc[:, 4])
tensile = np.asarray(data.iloc[:, 5])

x = np.c_[speed, temperature, fill, thickness, orientation]
y = tensile
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=TRAINING_SIZE, random_state=1)

def aptidao(x, i):
    learning_rate_init = binary_to_real(x[:25])
    beta_1 = binary_to_real(x[25:50])
    beta_2 = binary_to_real(x[50:75])
    epsilon = binary_to_real(x[75:100])
    hidden_layer_sizes = (
        binary_to_integer(x[100:106]),
        binary_to_integer(x[106:112]),
        binary_to_integer(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=MAX_ITER_NN, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=25, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_train, y_train)
    score = regr.score(x_test, y_test)
    return score


def predicao(x):
    learning_rate_init = binary_to_real(x[:25])
    beta_1 = binary_to_real(x[25:50])
    beta_2 = binary_to_real(x[50:75])
    epsilon = binary_to_real(x[75:100])
    hidden_layer_sizes = (
        binary_to_integer(x[100:106]),
        binary_to_integer(x[106:112]),
        binary_to_integer(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=MAX_ITER_NN, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=25, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_train, y_train)
    pred = regr.predict(x_test)
    score = regr.score(x_test, y_test)
    mse = mean_squared_error(y_test, pred)
    return pred, learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse


def on_start(model):
    print('----------------------- GA Started {} -------------------------'.format(datetime.today()))
    arquivo = open(os.path.join(ROOT_DIR, "./times.txt"), "a")
    arquivo.write(
        '----------------------- GA Started {} -------------------------\n'.format(datetime.today()))
    arquivo.write('Pop size {}\n'.format(model.pop_size))
    arquivo.close()
    print('Pop size {}'.format(model.pop_size))


def on_generation(model):
    print("Generation {}/{}".format(model.generations_completed, NUM_GENERATIONS))
    solution, solution_fitness, solution_idx = model.best_solution()
    print("Best fitness: {}".format(solution_fitness))
    print(model.last_generation_fitness)


def on_stop(model, aptidoesFinais):
    print('--------------------- GA Finished {} -----------------------'.format(datetime.today()))
    arquivo = open(os.path.join(ROOT_DIR, "./times.txt"), "a")
    arquivo.write(
        '----------------------- GA Finished {} -------------------------\n'.format(datetime.today()))
    arquivo.close()


with open(os.path.join(ROOT_DIR, 'data_table.csv'), 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Pop size', 'Test', 'Best individual', 'Best fitness',
                    'learning_rate_init', 'beta_1', 'beta_2', 'epsilon', 'hidden_layer_sizes', 'r^2', 'mean_squared_error'])
    for tam_pop in POP_SIZE:
        for i in range(NUM_EXEC):
            print("------------------------------------------------- Iteração #{} ------------------------------------------------".format(i + 1))
            model = pygad.GA(num_generations=NUM_GENERATIONS, num_parents_mating=tam_pop,
                                fitness_func=aptidao, sol_per_pop=tam_pop, keep_parents=int(
                                    tam_pop/4),
                                num_genes=NUM_GENES, gene_type=int,
                                init_range_low=0, init_range_high=2,
                                parent_selection_type="tournament", K_tournament=3, crossover_type="two_points",
                                crossover_probability=0.9, mutation_type="random", suppress_warnings=True,
                                mutation_probability=0.05, on_start=on_start, on_stop=on_stop,
                                on_generation=on_generation, stop_criteria="saturate_10")
            model.run()
            solution, solution_fitness, solution_idx = model.best_solution()
            pred, learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse = predicao(
                solution)

            print("Best individual: {}".format(solution))
            print("Best fitness: {}".format(solution_fitness))
            print('learning_rate_init: ', learning_rate_init)
            print('beta_1: ', beta_1)
            print('beta_2: ', beta_2)
            print('epsilon: ', epsilon)
            print('hidden_layer_sizes: ', hidden_layer_sizes)
            print('r^2: ', score)
            print('mean_squared_error: ', mse)
            print("\n")

            plt.subplot(1, 2, 1)
            plt.subplots_adjust(wspace=0.25)
            plt.plot(model.best_solutions_fitness)
            plt.title('Fitness x Generation ({})'.format(tam_pop))
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.subplot(1, 2, 2)
            plt.plot(y_test, pred, 'ro')
            plt.plot(np.linspace(0, 60, 100), np.linspace(0, 60, 100), 'b')
            plt.title('Real x Predict ({})'.format(tam_pop))
            plt.xlabel('Real')
            plt.ylabel('Predição')
            if not os.path.exists(os.path.join(ROOT_DIR, 'images')):
                os.makedirs(os.path.join(ROOT_DIR, 'images'))
            plt.savefig(os.path.join(ROOT_DIR, 'images', '{}_{}.png'.format(tam_pop, i + 1)), format="png")
            # plt.show(block=False)
            plt.close()

            arquivo = open(os.path.join(ROOT_DIR, "tests.txt"), "a")
            arquivo.write(
                '----------------------------------------------------------------------------------------------------------------------\n')
            arquivo.write('Pop size {}\n'.format(tam_pop))
            arquivo.write('Test {}\n'.format(i + 1))
            arquivo.write("Best individual: {}\n".format(solution))
            arquivo.write(
                "Best fitness: {}\n".format(solution_fitness))
            arquivo.write(
                'learning_rate_init: {}\n'.format(learning_rate_init))
            arquivo.write('beta_1: {}\n'.format(beta_1))
            arquivo.write('beta_2: {}\n'.format(beta_2))
            arquivo.write('epsilon: {}\n'.format(epsilon))
            arquivo.write(
                'hidden_layer_sizes: {}\n'.format(hidden_layer_sizes))
            arquivo.write('r^2: {}\n'.format(score))
            arquivo.write('mean_squared_error: {}\n'.format(mse))
            arquivo.write("\n")
            arquivo.close()

            writer.writerow([tam_pop, i + 1, solution, solution_fitness, learning_rate_init,
                            beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse])
