"""
sea.py

"""


import numpy as np
import random
from operator import itemgetter


# --------------------------------------------------

# ----- Simple Evolutionary Algorithm - Binary

def sea_bin(size_cromo, num_generations, population_size, mutation_method, mutation_prob, crossover_method, crossover_prob, parent_sel_method, survival_sel_method, fitness_func, seed=None, immigrants_insertion=0):
    # Inicialize population
    populacao = gera_pop_bin(population_size, size_cromo, seed)
    # Evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    # Save Values
    fitness_best_gen = [best_pop_maximization(populacao)[1]]
    fitness_average_gen = [average_pop_bin(populacao)]
    for g in range(num_generations):
        # Parents selection
        mate_pool = parent_sel_method(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, population_size - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = crossover_method(indiv_1, indiv_2, crossover_prob)
            progenitores.extend(filhos)
        # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation_method(cromo, mutation_prob)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = survival_sel_method(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        # Immigrant Insertion
        if immigrants_insertion == 1:
            populacao = random_immigrants(0.2, populacao, 1, size_cromo)
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        if immigrants_insertion == 2:
            populacao = elitist_immigrants(0.2, populacao, 1, mutation_prob)
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        # Save Values
        fitness_best_gen.append(best_pop_maximization(populacao)[1])
        fitness_average_gen.append(average_pop_bin(populacao))
    return best_pop_maximization(populacao), fitness_best_gen, fitness_average_gen


# Initialize population
def gera_pop_bin(size_pop, size_cromo, seed):
    if seed is None:
        return [(gera_indiv_bin(size_cromo), 0) for i in range(size_pop)]
    else:
        random.seed(seed)
        pop = [(gera_indiv_bin(size_cromo), 0) for i in range(size_pop)]
        random.seed()
        return pop

def gera_indiv_bin(size_cromo):
    # Random initialization
    indiv = [random.randint(0, 1) for i in range(size_cromo)]
    return indiv

# Variation operator: Binary Mutation
def muta_bin(indiv, prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i], prob_muta)
    return cromo

def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random.random()
    if value < prob_muta:
        g ^= 1
    return g

# Variation Operator: One-Point Crossover
def one_point_cross(indiv_1, indiv_2, prob_cross):
    value = random.random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pos = random.randint(0, len(cromo_1))
        f1 = cromo_1[0:pos] + cromo_2[pos:]
        f2 = cromo_2[0:pos] + cromo_1[pos:]
        return (f1, 0), (f2, 0)
    else:
        return indiv_1, indiv_2

# Parents Selection: Tournament
def tour_sel_maximization(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour_maximization(pop, t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def tour_maximization(population, size):
    """Maximization Problem. Deterministic"""
    pool = random.sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]

# Survivals Selection: Elitism
def sel_survivors_elite_maximization(elite):
    """Maximization"""
    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

# Auxiliary
def best_pop_maximization(populacao):
    """Maximization."""
    populacao.sort(key=itemgetter(1), reverse=True)
    return populacao[0]

def average_pop_bin(populacao):
    return sum([indiv[1] for indiv in populacao])/len(populacao)


# ----- Simple Evolutionary Algorithm - Float

def sea_float(domain, num_generations, population_size, mutation_method, mutation_prob, crossover_method, crossover_prob, parent_sel_method, survival_sel_method, fitness_func, seed=None, immigrants_insertion=0):
    # Inicialize population
    populacao = gera_pop_float(population_size, domain, seed)
    # Evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    # Save Values
    best_overall = [best_pop_minimization(populacao), 0]
    fitness_best_gen = [best_pop_minimization(populacao)[1]]
    fitness_average_gen = [average_pop_float(populacao)]
    for gen in range(num_generations):
        # Parents selection
        mate_pool = parent_sel_method(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, population_size - 1):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = crossover_method(indiv_1, indiv_2, crossover_prob)
            progenitores.extend(filhos)
            if len(progenitores) == population_size:
                break
        # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation_method(cromo, mutation_prob, domain)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = survival_sel_method(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        # Immigrant Insertion
        if immigrants_insertion == 1:
            populacao = random_immigrants(0.2, populacao, 2, domain)
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        if immigrants_insertion == 2:
            populacao = elitist_immigrants(0.2, populacao, 2, mutation_prob, domain)
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        # Save Values
        new_best = best_pop_minimization(populacao)
        if new_best[1] < best_overall[0][1]:
            best_overall[0] = new_best
            best_overall[1] = gen
        fitness_best_gen.append(best_pop_minimization(populacao)[1])
        fitness_average_gen.append(average_pop_float(populacao))
    return best_pop_minimization(populacao), fitness_best_gen, fitness_average_gen, best_overall


# Initialize population
def gera_pop_float(size_pop, domain, seed):
    if seed is None:
        return [(gera_indiv_float(domain), 0) for i in range(size_pop)]
    else:
        random.seed(seed)
        pop = [(gera_indiv_float(domain), 0) for i in range(size_pop)]
        random.seed()
        return pop

def gera_indiv_float(domain):
    return [random.uniform(domain[i][0], domain[i][1]) for i in range(len(domain))]

# Variation Operator: Uniform Mutation
def muta_uni(indiv, prob_muta, domain):
    cromo = indiv[:]
    for i in range(len(cromo)):
        cromo[i] = muta_uni_gene(cromo[i], prob_muta, domain[i])
    return cromo

def muta_uni_gene(gene, prob_muta, domain_gene):
    value = np.random.random()
    new_gene = gene
    if value < prob_muta:
        new_gene = np.random.uniform(domain_gene[0], domain_gene[1])
    return new_gene

# Variation Operator: Arithmetical Crossover
def arithmetical_cross(alpha):
    def arithmetical(indiv_1, indiv_2, prob_cross):
        size = len(indiv_1[0])
        value = random.random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            f1 = [None] * size
            f2 = [None] * size
            for i in range(size):
                f1[i] = alpha * cromo_1[i] + (1 - alpha) * cromo_2[i]
                f2[i] = (1 - alpha) * cromo_1[i] + alpha * cromo_2[i]
            return (f1, 0), (f2, 0)
        return indiv_1, indiv_2

    return arithmetical

# Parents Selection: Tournament
def tour_sel_minimization(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour_minimization(pop, t_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament

def tour_minimization(population, size):
    """Minimization Problem.Deterministic"""
    pool = random.sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]

# Survivals Selection: Elitism
def sel_survivors_elite_minimization(elite):
    def elitism(parents, offspring):
        """Minimization."""
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism

# Auxiliary
def best_pop_minimization(populacao):
    """Minimization."""
    populacao.sort(key=itemgetter(1))
    return populacao[0]

def average_pop_float(populacao):
    return sum([fit for cromo,fit in populacao])/len(populacao)


# ----- Immigrants Insertion

# Random Immigrants
def random_immigrants(replacement_rate, population, problem_type, indiv):
    if problem_type == 1: # bin
        population.sort(key=itemgetter(1), reverse=True)
    if problem_type == 2: # float
        population.sort(key=itemgetter(1))

    new_population = population[:]
    num_replace = int(len(population) * replacement_rate)
    for i in range(num_replace):
        if problem_type == 1:  # bin
            new_population[-1 - i] = (gera_indiv_bin(indiv), 0)
        if problem_type == 2:  # float
            new_population[-1 - i] = (gera_indiv_float(indiv), 0)

    return new_population


# Elitist Immigrants
def elitist_immigrants(replacement_rate, population, problem_type, muta_prob, domain=None):
    if problem_type == 1:  # bin
        population.sort(key=itemgetter(1), reverse=True)
    if problem_type == 2:  # float
        population.sort(key=itemgetter(1))

    best = population[0]
    new_population = population[:]
    num_replace = int(len(population) * replacement_rate)
    for i in range(num_replace):
        temp = best[:]
        if problem_type == 1:  # bin
            temp = muta_bin(temp[0], muta_prob)
        if problem_type == 2:  # float
            temp = muta_uni(temp[0], muta_prob, domain)

        new_population[-1 - i] = (temp, 0)

    return new_population
