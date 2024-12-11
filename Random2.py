import time
import numpy as np
import pandas as pd
import random
import customhys as mh
from customhys import metaheuristic as mh
from joblib import Parallel, delayed
import multiprocessing
import warnings
from customhys import benchmark_func as bf
import scipy.io

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")

# Cargar los datos precomputados
mat_data = scipy.io.loadmat('precomputed_data.mat')
precomputed_data = mat_data['precomputedData']

def desnormalize_vector(vector, min_val=1, max_val=256):
    desnormalized_vector = min_val + 0.5 * (vector + 1) * (max_val - min_val)
    return desnormalized_vector

def renyi(th1, th2, H2D, level):
    alpha = 0.5
    Fitness = np.full(level, np.inf)
    epsilon = 1e-8

    for i in range(level):
        with np.errstate(divide='ignore'):
            if i == 0:
                pk = np.sum(H2D[0:th2[i], 0:th1[i]])
                pk = max(pk, epsilon)
                fitness_value = (1 / (1 - alpha)) * np.log(np.sum((H2D[0:th2[i], 0:th1[i]] / pk) ** alpha))
            else:
                pk = np.sum(H2D[th2[i - 1] + 1:th2[i], th1[i - 1] + 1:th1[i]])
                pk = max(pk, epsilon)
                fitness_value = (1 / (1 - alpha)) * np.log(
                    np.sum((H2D[th2[i - 1] + 1:th2[i], th1[i - 1] + 1:th1[i]] / pk) ** alpha))

        Fitness[i] = fitness_value

    Fitness = np.nansum(Fitness)
    return Fitness

def psegmen_precomputed(x1, x2, x3, x4, image_index):
    global precomputed_data
    H2D = precomputed_data[0, image_index]['H2D']
    Th1 = [round(x1), round(x2)]
    Th2 = [round(x3), round(x4)]
    fobj = renyi
    Fobj = -fobj(Th1, Th2, H2D, 2)
    if abs(Fobj) == float('inf'):
        Fobj = 10000
    return Fobj

class P1(bf.BasicProblem):
    def __init__(self, variable_num, image_index):
        super().__init__(variable_num)
        self.variable_num = variable_num
        self.max_search_range = np.array([256, 256, 256, 256])
        self.min_search_range = np.array([1, 1, 1, 1])
        self.func_name = 'P1'
        self.image_index = image_index

    def set_image_index(self, image_index):
        self.image_index = image_index

    def get_func_val(self, variables, *args):
        fcost = psegmen_precomputed(variables[0], variables[1], variables[2], variables[3], image_index=self.image_index)
        return fcost

# Espacio Heurístico
heuristic_space = [
    ('central_force_dynamic', {'gravity': 0.001, 'alpha': 0.01, 'beta': 1.5, 'dt': 1.0}, 'all'),
    ('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 1, 'factor': 1.0}, 'greedy'),
    ('firefly_dynamic', {'distribution': 'uniform', 'alpha': 1.0, 'beta': 1.0, 'gamma': 100.0}, 'all'),
    ('genetic_crossover', {'pairing': 'tournament_2_100', 'crossover': 'uniform', 'mating_pool_factor': 0.4}, 'all'),
    ('genetic_mutation', {'scale': 1.0, 'elite_rate': 0.1, 'mutation_rate': 0.25, 'distribution': 'uniform'}, 'greedy'),
    ('gravitational_search', {'gravity': 1.0, 'alpha': 0.02}, 'all'),
    ('random_flight', {'scale': 1.0, 'distribution': 'levy', 'beta': 1.5}, 'greedy'),
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
    ('random_sample', {}, 'all'),
    ('random_search', {'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'all'),
    ('swarm_dynamic',
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'all')
]

# Generación Aleatoria de Población
def generate_random_population(heuristic_space_list, pop_size, num_operators):
    population = []
    for _ in range(pop_size):
        selected_operators = random.sample(heuristic_space_list, num_operators)
        population.append(selected_operators)
    return population

def evaluate_sequence_P1(sequence, num_agents, num_iterations, num_replicas, prob):
    def run_metaheuristic(prob):
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        best_position, f_best = met.get_solution()
        hist_values = met.historical
        return f_best, best_position, hist_values['fitness'], hist_values['position']

    num_cores = multiprocessing.cpu_count()
    all_fitness_values = []
    all_positions = []

    results_parallel = Parallel(n_jobs=num_cores)(delayed(run_metaheuristic)(prob) for _ in range(num_replicas))

    fitness_values = [result[0] for result in results_parallel]
    positions = [result[1] for result in results_parallel]
    all_fitness_values.extend(fitness_values)
    all_positions.extend(positions)

    median_fitness = np.median(fitness_values)
    q1 = np.percentile(fitness_values, 25)
    q3 = np.percentile(fitness_values, 75)
    iqr = q3 - q1
    metric = median_fitness + iqr
    return metric
def random_search_optimization(heuristic_space_list, num_generations, pop_size, num_operators, num_agents, num_iterations, num_replicas):
    best_sequence = None
    best_performance = float('inf')
    worst_performance = float('-inf')
    performance_history = []

    with open('p1_10img.txt', 'w') as file:
        for gen in range(num_generations):
            print(f"Generation {gen + 1}")
            gen_start_time = time.time()

            population = generate_random_population(heuristic_space_list, pop_size, num_operators)
            generation_performance = []
            generation_sequences = []

            for seq_index, seq in enumerate(population):
                image_performances = []

                for image_index in range(10):  # Loop over both images
                    fun = P1(4, image_index=image_index)
                    prob = fun.get_formatted_problem()

                    performance = evaluate_sequence_P1(seq, num_agents, num_iterations, num_replicas, prob)
                    image_performances.append(performance)

                avg_performance = np.mean(image_performances)
                generation_performance.append(avg_performance)
                generation_sequences.append(seq)

                if avg_performance < best_performance:
                    best_performance = avg_performance
                    best_sequence = seq

                if avg_performance > worst_performance:
                    worst_performance = avg_performance
                    worst_sequence = seq

            performance_history.append((gen + 1, best_performance, best_sequence))

            file.write(f"Generation {gen + 1}\n")
            file.write(f"Best Sequence: {best_sequence}\n")
            file.write(f"Best Performance: {best_performance}\n")
            file.write(f"Worst Sequence: {worst_sequence}\n")


            file.write(f"Worst Performance: {worst_performance}\n\n")

            total_time = time.time() - gen_start_time
            print(f"Total time for generation {gen + 1}: {total_time:.2f} seconds")

        file.write("Performance History:\n")
        for gen, perf, seq in performance_history:
            file.write(f"Generation {gen}: Performance: {perf}, Sequence: {seq}\n")

    return best_sequence, best_performance, performance_history

if __name__ == "__main__":
    num_generations = 10
    pop_size = 4
    num_operators = 3
    num_agents = 30
    num_iterations = 60
    num_replicas = 30
    best_sequence, best_performance, performance_history = random_search_optimization(heuristic_space, num_generations, pop_size, num_operators, num_agents, num_iterations, num_replicas)
    print("Best Sequence:", best_sequence)
    print("Best Performance:", best_performance)
