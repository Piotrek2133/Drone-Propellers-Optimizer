import numpy as np
from deap import base, creator, tools, algorithms
import pickle

# Initial data for r/R, c/R, and beta
initial_data = [
    (0.15, 0.114, 28.24),
    (0.20, 0.134, 31.87),
    (0.25, 0.157, 32.26),
    (0.30, 0.177, 31.32),
    (0.35, 0.194, 29.65),
    (0.40, 0.208, 27.57),
    (0.45, 0.218, 25.24),
    (0.50, 0.225, 22.97),
    (0.55, 0.228, 20.94),
    (0.60, 0.228, 19.19),
    (0.65, 0.223, 17.69),
    (0.70, 0.214, 16.31),
    (0.75, 0.202, 15.07),
    (0.80, 0.185, 13.94),
    (0.85, 0.165, 12.89),
    (0.90, 0.139, 11.86),
    (0.95, 0.100, 11.04),
    (1.00, 0.060, 10.23),
]

# Convert to numpy array for easier manipulation
initial_data = np.array(initial_data)
rR = initial_data[:, 0]

# Number of points
num_points = len(initial_data)

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
def generate_initial_cR_beta():
    return [np.random.uniform(cR - 0.01, cR + 0.01) for cR in initial_data[:, 1]] + \
           [np.random.uniform(beta - 1.0, beta + 1.0) for beta in initial_data[:, 2]]

# Register individual creation
toolbox.register("individual", tools.initIterate, creator.Individual, generate_initial_cR_beta)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evaluate(individual, external_params):
    cR_beta_array = np.array(individual).reshape(2, -1).T
    efficiency = predict_efficiency(cR_beta_array, external_params)
    if efficiency is None or len(efficiency) == 0:
        return -float("inf"),  # Return a very low fitness value if the prediction fails
    
    return float(efficiency[0]), 

# Define mutation with constraints
def constrained_mutation(individual, indpb=0.2):
    for i in range(num_points):
        # Mutate c/R
        if np.random.rand() < indpb:
            individual[i] = np.clip(individual[i] + np.random.uniform(-0.005, 0.005),
                                    initial_data[i, 1] - 0.01,
                                    initial_data[i, 1] + 0.01)
        # Mutate beta
        if np.random.rand() < indpb:
            individual[num_points + i] = np.clip(individual[num_points + i] + np.random.uniform(-0.5, 0.5),
                                                 initial_data[i, 2] - 1.0,
                                                 initial_data[i, 2] + 1.0)
    return individual,

# Register functions for DEAP
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", constrained_mutation, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run Genetic Algorithm
def run_ga(external_params):
    toolbox.register("evaluate", evaluate, external_params=external_params)
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=50, stats=stats, halloffame=hof, verbose=True)
    return hof[0], evaluate(hof[0], external_params)
    
# Optimization function
def optimize(external_params):
    best_individual, best_efficiency = run_ga(external_params)
    cR = best_individual[:18]
    beta = best_individual[18:]
    rR = initial_data[:,0]
    input_data = np.hstack([external_params, rR, cR, beta]).reshape(1,-1)
    pred = model.predict(input_data)  
    ct = pred[0,1]
    B = external_params[0]
    D = external_params[1]
    J = external_params[3]
    rho = 1.225
    v = J * B * D
    T = ct * 0.5 * rho * v**2 * np.pi * D ** 2 / 4
    best_individual  = np.array(best_individual).reshape(num_points, 2)
    #print("Optimal c/R and beta:", np.array(best_individual).reshape(num_points, 2))
    #print("Predicted Efficiency:", best_efficiency)
    return best_individual, best_efficiency, v, T

