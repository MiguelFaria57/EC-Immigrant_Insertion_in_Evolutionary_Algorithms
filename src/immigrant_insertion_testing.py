"""
immigrant_insertion_testing.py

"""


from sea import *
from benchmarks import fitness_onemax, fitness_jb, fitness_quartic, fitness_rastrigin
from utils import *
from random import sample


# --------------------------------------------------

def test_immigrant_insertion_bin(problem_name, problem_fitness):
    print("\n" + problem_name)
    num_runs = 30
    seeds = sample(range(1, 1000), num_runs)
    save_seeds("output/test_immigrants_insertion/seeds_" + problem_name, seeds)
    size_cromo = 100
    num_generations = 1000
    population_size = 100
    mutation_method = muta_bin
    mutation_prob = 0.05
    crossover_method = one_point_cross
    crossover_prob = 0.9
    parent_sel_method = tour_sel_maximization(5)
    survival_sel_method = sel_survivors_elite_maximization(0.2)
    fitness_func = problem_fitness
    immigrants_insertion_method = ("default", "random-immigrants", "elitist-immigrants")

    last_best = [[] for _ in range(len(immigrants_insertion_method))]
    mean_average = [[] for _ in range(len(immigrants_insertion_method))]
    for r, s in enumerate(seeds):
        print("Run: %s | Seed: %s" %(str(r), str(s)))

        best_indiv = [[] for _ in range(len(immigrants_insertion_method))]
        best_gen = [[] for _ in range(len(immigrants_insertion_method))]
        average_gen = [[] for _ in range(len(immigrants_insertion_method))]
        for iim in range(len(immigrants_insertion_method)):
            best_final, fitness_best_gen, fitness_average_gen = sea_bin(size_cromo, num_generations, population_size, mutation_method, mutation_prob, crossover_method, crossover_prob, parent_sel_method, survival_sel_method, fitness_func, s, immigrants_insertion_method)
            best_indiv[iim] = best_final
            best_gen[iim] = fitness_best_gen
            average_gen[iim] = fitness_average_gen
            last_best[iim].append(fitness_best_gen[-1])
            mean_average[iim].append(sum(fitness_average_gen) / len(fitness_average_gen))

            filename = ("output/test_immigrants_insertion/problems/", "test_" + problem_name + "_" + immigrants_insertion_method[iim] + "_r-" + str(r))
            save_plot_best_and_average(problem_name, filename, fitness_best_gen, fitness_average_gen, None, 0)
            save_best_and_average(filename, fitness_best_gen, fitness_average_gen)
            save_best_overall(filename, best_final)

        filename = ("output/test_immigrants_insertion/problems/", "test_" + problem_name + "_all" + "_r-" + str(r))
        save_plot_algorithms(problem_name, filename, best_gen[0], best_gen[1], best_gen[2], 0, 1)
        save_algorithms(filename, best_gen[0], best_gen[1], best_gen[2])
    filename = ("output/test_immigrants_insertion/runs/", "test_" + problem_name + "_all")
    save_plot_algorithms(problem_name, filename, last_best[0], last_best[1], last_best[2], 0, 0)
    save_algorithms(filename, last_best[0], last_best[1], last_best[2])


def test_immigrant_insertion_float(problem_name, problem_fitness, problem_domain):
    print("\n" + problem_name)
    num_runs = 30
    seeds = sample(range(1, 1000), num_runs)
    save_seeds("output/test_immigrants_insertion/seeds_"+problem_name, seeds)
    dimensions = 25
    domain = [problem_domain] * dimensions
    num_generations = 1000
    population_size = 100
    mutation_method = muta_uni
    mutation_prob = 0.05
    crossover_method = arithmetical_cross(0.5)
    crossover_prob = 0.9
    parent_sel_method = tour_sel_minimization(5)
    survival_sel_method = sel_survivors_elite_minimization(0.2)
    fitness_func = problem_fitness
    immigrants_insertion_method = ("default", "random-immigrants", "elitist-immigrants")

    last_best = [[] for _ in range(len(immigrants_insertion_method))]
    mean_average = [[] for _ in range(len(immigrants_insertion_method))]
    best = [[] for _ in range(len(immigrants_insertion_method))]
    for r, s in enumerate(seeds):
        print("Run: %s | Seed: %s" %(str(r), str(s)))

        best_indiv = [[] for _ in range(len(immigrants_insertion_method))]
        best_gen = [[] for _ in range(len(immigrants_insertion_method))]
        average_gen = [[] for _ in range(len(immigrants_insertion_method))]
        for iim in range(len(immigrants_insertion_method)):
            best_final, fitness_best_gen, fitness_average_gen, best_overall = sea_float(domain, num_generations, population_size, mutation_method, mutation_prob, crossover_method, crossover_prob, parent_sel_method, survival_sel_method, fitness_func)
            best_indiv[iim] = best_overall
            best_gen[iim] = fitness_best_gen
            average_gen[iim] = fitness_average_gen
            last_best[iim].append(fitness_best_gen[-1])
            mean_average[iim].append(sum(fitness_average_gen) / len(fitness_average_gen))
            best[iim].append(best_overall[0][1])

            filename = ("output/test_immigrants_insertion/problems/", "test_" + problem_name + "_" + immigrants_insertion_method[iim] + "_r-" + str(r))
            save_plot_best_and_average(problem_name, filename, fitness_best_gen, fitness_average_gen, best_overall, 0)
            save_best_and_average(filename, fitness_best_gen, fitness_average_gen)
            save_best_overall(filename, best_overall)

        filename = ("output/test_immigrants_insertion/problems/", "test_" + problem_name + "_all" + "_r-" + str(r))
        save_plot_algorithms(problem_name, filename, best_gen[0], best_gen[1], best_gen[2], 0, 1)
        save_algorithms(filename, best_gen[0], best_gen[1], best_gen[2])
    filename = ("output/test_immigrants_insertion/runs/", "test_" + problem_name + "_all")
    save_plot_algorithms(problem_name, filename, best[0], best[1], best[2], 0, 0)
    save_algorithms(filename, best[0], best[1], best[2])



########################################################################################################################

if __name__ == "__main__":
    test_immigrant_insertion_bin("OneMax", fitness_onemax)
    test_immigrant_insertion_bin("JoaoBrandaosNumbers", fitness_jb)
    test_immigrant_insertion_float("QuarticFunction", fitness_quartic, [-1.28, 1.28])
    test_immigrant_insertion_float("RastriginFunction", fitness_rastrigin, [-5.12, 5.12])

    pass
