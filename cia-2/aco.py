from tkinter import BEVEL
import numpy as np

np.random.seed(5)
def bceloss(y_true, y_pred):
    num = (1-y_true) * np.log(1 - y_pred + 1e-7)
    den = y_true * np.log(y_pred + 1e-7)
    return -np.mean(num+den, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_ant_colony(num_ants, num_weights):
    return np.random.randn(num_ants, num_weights)

def fitness(ants, X, y, layer):
    num_ants = ants.shape[0]
    loss = np.zeros(num_ants)
    for i in range(num_ants):
        y_pred = np.round(sigmoid(layer(X, ants[i][:11], ants[i][-1])))
        loss[i] = bceloss(y, y_pred)
    return loss

def update_pheromone(pheromone, ants, evaporation_rate, fitness_list):
    num_ants, num_weights = ants.shape
    for i in range(num_ants):
        for j in range(num_weights):
            pheromone[i][j] = (1 - evaporation_rate) * pheromone[i][j] + evaporation_rate * fitness_list[i]
    return pheromone

def select_weights(pheromone, num_weights):
    probabilities = np.sum(pheromone, axis=0) / np.sum(pheromone)
    weights = np.random.choice(np.arange(num_weights), size=num_weights, p=probabilities)
    return weights

def optimize_weights(X, y, num_ants, num_iterations, evaporation_rate, alpha, beta,layer):
    # num_samples, num_features = X.shape
    num_features = num_ants
    ants = init_ant_colony(num_ants, num_features)
    pheromone = np.ones((num_ants, num_features))

    for i in range(num_iterations):
        fitness_list = fitness(ants, X, y, layer)
        pheromone = update_pheromone(pheromone, ants, evaporation_rate, fitness_list)
        selected_weights = select_weights(pheromone, num_features)
        # print(selected_weights)
        for j in range(num_ants):
            delta_weights = alpha * (np.random.randn(num_features) * beta + ants[selected_weights[j]] - ants[j] )
            ants[j] += delta_weights


    return ants[np.argmin(fitness_list)] , np.min(fitness_list)