import numpy as np
from deap import base, creator, tools, algorithms
import pickle


rR = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Number of points
num_points = 18

# Read model
filename = "model/eta_cT_model.sav"
model = pickle.load(open(filename, 'rb'))

# Function to predict efficiency
def predict_efficiency(cR_beta_array, external_params):
    cR = cR_beta_array[:, 0]
    beta = cR_beta_array[:, 1]
    input_data = np.hstack([external_params, rR, cR, beta]).reshape(1, -1)
    return (model.predict(input_data))[0]

# DEAP Genetic Algorithm Configuration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Attribute generator: Generate initial values close to the original
def generate_initial_cR_beta(initial_data):
    return [np.random.uniform(cR - 0.01, cR + 0.01) for cR in initial_data[:, 0]] + \
           [np.random.uniform(beta - 1.0, beta + 1.0) for beta in initial_data[:, 1]]

# Register individual creation
def setup_toolbox(initial_data):
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: generate_initial_cR_beta(initial_data))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evaluate(individual, external_params):
    cR_beta_array = np.array(individual).reshape(2, -1).T
    efficiency = predict_efficiency(cR_beta_array, external_params)
    if efficiency is None or len(efficiency) == 0:
        return -float("inf"),  # Return a very low fitness value if the prediction fails
    
    return float(efficiency[0]), 

# Define mutation with constraints
def constrained_mutation(individual, initial_data, indpb=0.2):
    for i in range(num_points):
        # Mutate c/R
        if np.random.rand() < indpb:
            individual[i] = np.clip(individual[i] + np.random.uniform(-0.005, 0.005),
                                    initial_data[i, 0] - 0.01,
                                    initial_data[i, 0] + 0.01)
        # Mutate beta
        if np.random.rand() < indpb:
            individual[num_points + i] = np.clip(individual[num_points + i] + np.random.uniform(-0.5, 0.5),
                                                 initial_data[i, 1] - 1.0,
                                                 initial_data[i, 1] + 1.0)
    return individual,


# Run Genetic Algorithm
def run_ga(external_params, initial_data):

    # Register functions for DEAP
    setup_toolbox(initial_data)
    toolbox.register("evaluate", evaluate, external_params=external_params)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", constrained_mutation, indpb=0.2, initial_data = initial_data)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=50, stats=stats, halloffame=hof, verbose=False)
    return hof[0], evaluate(hof[0], external_params)
    
# Optimization function
def optimize(external_params, initial_data):
    initial_data = np.array(initial_data)
    best_individual, best_efficiency = run_ga(external_params, initial_data)
    cR = best_individual[:18]
    beta = best_individual[18:]
    input_data = np.hstack([external_params, rR, cR, beta]).reshape(1,-1)
    pred = model.predict(input_data)  
    ct = pred[0,1]
    N = external_params[4] / 60
    D = external_params[1]
    J = external_params[3]
    rho = 1.225
    v = J * N * D
    T = ct * 0.5 * rho * v**2 * np.pi * D ** 2 / 4

    best_individual = np.column_stack((cR, beta))

    return best_individual, best_efficiency, v, T
