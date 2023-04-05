import numpy as np

class GeneticAlgorithm:
    def __init__(self, population, y_true, x, layer, n_parents=2,crossover_rate = 0.5, mutation_rate=0.01, n_generations=1000):
        self.population = population
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate

        self.y_true = y_true
        self.x = x
        self.layer = layer

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def fitness_function(self, weights):
        # print(weights)
        y_pred = np.round(self.sigmoid( self.layer(self.x, weights[:11], weights[-1]) ))
        num = (1-self.y_true) * np.log(1 - y_pred + 1e-7)
        den = self.y_true * np.log(y_pred + 1e-7)
        return -np.mean(num+den, axis=0)
    
    def selection(self, population, fitness, n_parents):
        max_ = np.argsort(fitness)[::-1]
        parents = np.zeros((n_parents, population.shape[1]))

        for i in range(n_parents):
            parents[i] = population[max_[i]]

        return parents

    def crossover(self, parents, n_children):
        n_parents = parents.shape[0]
        n_chromosome = parents.shape[1]
        children = np.zeros((n_children, n_chromosome))

        for i in range(n_children):
            parent1 = parents[np.random.randint(n_parents)]
            parent2 = parents[np.random.randint(n_parents)]

            crossover_point = np.random.randint(1, n_chromosome - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            children[i] = child

        return children

    def mutation(self, population, mutation_rate):
        n_population = population.shape[0]
        n_chromosome = population.shape[1]

        for i in range(n_population):
            for j in range(n_chromosome):
                if np.random.rand() < mutation_rate:
                    population[i, j] = np.random.uniform(-1, 1)

    def evolve(self):
        n_population = self.population.shape[0]

        for i in range(self.n_generations):

            fitness = np.zeros(n_population)
            for j in range(n_population):
                fitness[j] = self.fitness_function(self.population[j])

            parents = self.selection(self.population, fitness, self.n_parents)
            children = np.zeros((n_population - self.n_parents, parents.shape[1]))


            if np.random.uniform(0, 1) < self.crossover_rate:
                children = self.crossover(parents, n_population - self.n_parents)

                self.mutation(children, self.mutation_rate)
            else:
                children = parents

            children_fitness = np.zeros(len(children))


            for j in range(len(children)):
                children_fitness[j] = self.fitness_function(children[j])
                

            population = np.concatenate((parents, children), axis=0)
            fitness = np.concatenate((fitness[:self.n_parents], children_fitness), axis=0)

            max_ = np.argsort(fitness)[::-1][:n_population]
            self.population = population[max_]

        best_chromosome = self.population[0]
        best_fitness = self.fitness_function(best_chromosome)
        return best_chromosome, best_fitness


