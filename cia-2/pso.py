import math
import numpy as np
np.random.seed(9)


class particle():
    def __init__(self, dimensions, x_train , y_train,lossfn,dense, c1 = 1, c2 = 1, MAX = 1 ):
        self.position = np.random.uniform(-2,2, size = (dimensions,)) 
        self.best_position = self.position
        self.fitness = f(lossfn, dense,y_train,x_train, self.position[:11] , self.position[11:])
        self.best_fitness = self.fitness
        self.velocity = np.zeros(shape = (dimensions,)) 

        self.c1,self.c2 = c1,c2
        self.MAX = MAX
        self.r1 = np.random.uniform(0,self.MAX)
        self.r2 = np.random.uniform(0,self.MAX)
        self.w = np.random.uniform(0.5,self.MAX)

def generate_swarm(x_train , y_train,lossfn,dense):
    swarm = [particle(12,x_train , y_train,lossfn,dense) for i in range(10)]
    return swarm
def get_globalbest(swarm):
    candidates = [i.best_fitness for i in swarm]
    print(candidates)
    return min(candidates), np.argmin(candidates)
    
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def f(loss_function,model,y_train,x , weights, bias):
    y_pred = np.round(sigmoid(model(x, weights, bias)))
    # y_pred = model(x, weights, bias)

    loss = loss_function(y_train, y_pred)   
    return loss



def update(swarm,global_best,lossfn, dense,y_train,x_train):
    for i in swarm:
        i.velocity = (i.w * i.velocity) + (i.c1 * i.r1 * (i.best_position - i.position)) + (i.c2 * i.r2 * (global_best - i.position))
        i.position = i.position + i.velocity

        i.fitness = f(lossfn, dense,y_train,x_train, i.position[:11] ,i.position[11:])
        
        if i.best_fitness > i.fitness:
            i.best_fitness = i.fitness
            i.best_position  = i.position

def PSO(lossfn, dense,y_train,x_train,threshold = 0, generation = 200 , n_inputs = 1):
    swarm = generate_swarm(x_train , y_train,lossfn,dense)

    # print("ok")
    for t in range(generation):
        fitness_list = [i.fitness for i in swarm]
        global_best, global_best_particle = get_globalbest(swarm)

        if np.average(fitness_list) <= threshold:
            break
        else:
            update(swarm,global_best,lossfn, dense,y_train,x_train)
        fitness_list = [i.fitness for i in swarm]
        global_best, global_best_particle = get_globalbest(swarm)
        # print('Best Fitness Value: ', min(fitness_list))
        
    # print('Global Best Position: ', global_best_particle)
    # print('Best Fitness Value: ', min(fitness_list))
    # print('Average Particle Best Fitness Value: ', np.average(fitness_list))
    # print('Number of Generation: ', t)

    return swarm[global_best_particle]