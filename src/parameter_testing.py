"""
parameter_testing.py

"""


from sea import *
from benchmarks import fitness_onemax, fitness_jb, fitness_quartic, fitness_rastrigin
from utils import *


# --------------------------------------------------

def test_problem_bin(problem_name, problem_fitness):
    size_cromo = 100
    num_generations = 1000
    population_size = 100
    mutation_method = muta_bin
    mutation_prob = (0.05, 0.1, 0.15)
    crossover_method = one_point_cross
    crossover_prob = (0.7, 0.8, 0.9)
    parent_sel_method = tour_sel_maximization(5)
    survival_sel_method = sel_survivors_elite_maximization(0.2)
    fitness_func = problem_fitness

    for mp in mutation_prob:
        for cp in crossover_prob:
            best_final, fitness_best_gen, fitness_average_gen = sea_bin(size_cromo, num_generations, population_size, mutation_method, mp, crossover_method, cp, parent_sel_method, survival_sel_method, fitness_func)

            display_best(best_final)
            filename = ("output/test_parameters/problems/", "test_" + problem_name + "_mp-" + str(mp) + "_cp-" + str(cp))
            save_plot_best_and_average(problem_name, filename, fitness_best_gen, fitness_average_gen, None, 0)
            save_best_and_average(filename, fitness_best_gen, fitness_average_gen)
            save_best_overall(filename, best_final)


def test_problem_float(problem_name, problem_fitness, problem_domain):
    dimensions = 25
    domain = [problem_domain] * dimensions
    num_generations = 1000
    population_size = 100
    mutation_method = muta_uni
    mutation_prob = (0.05, 0.1, 0.15)
    crossover_method = arithmetical_cross(0.5)
    crossover_prob = (0.7, 0.8, 0.9)
    parent_sel_method = tour_sel_minimization(5)
    survival_sel_method = sel_survivors_elite_minimization(0.2)
    fitness_func = problem_fitness

    for mp in mutation_prob:
        for cp in crossover_prob:
            best_final, fitness_best_gen, fitness_average_gen, best_overall = sea_float(domain, num_generations, population_size, mutation_method, mp, crossover_method, cp, parent_sel_method, survival_sel_method, fitness_func)

            display_best(best_final)
            display_best(best_overall[0], best_overall[1])
            filename = ("output/test_parameters/problems/", "test_" + problem_name + "_mp-" + str(mp) + "_cp-" + str(cp))
            save_plot_best_and_average(problem_name, filename, fitness_best_gen, fitness_average_gen, best_overall, 0)
            save_best_and_average(filename, fitness_best_gen, fitness_average_gen)
            save_best_overall(filename, best_overall)


def test_runs_bin(problem_name, problem_fitness):
    num_runs = 30
    size_cromo = 100
    num_generations = 1000
    population_size = 100
    mutation_method = muta_bin
    mutation_prob = (0.05, 0.1, 0.15)
    crossover_method = one_point_cross
    crossover_prob = (0.7, 0.8, 0.9)
    parent_sel_method = tour_sel_maximization(5)
    survival_sel_method = sel_survivors_elite_maximization(0.2)
    fitness_func = problem_fitness

    for mp in mutation_prob:
        for cp in crossover_prob:
            last_best = []
            mean_average = []
            for i in range(num_runs):
                print("MP: ", mp, " | CP: ", cp, " | Run: ",  i)
                best_final, fitness_best_gen, fitness_average_gen = sea_bin(size_cromo,num_generations,population_size,mutation_method,mp,crossover_method,cp,parent_sel_method,survival_sel_method,fitness_func)
                last_best.append(fitness_best_gen[-1])
                mean_average.append(sum(fitness_average_gen)/len(fitness_average_gen))

            filename = ("output/test_parameters/runs/", "test_" + problem_name + "_mp-" + str(mp) + "_cp-" + str(cp))
            save_plot_runs(filename, last_best, mean_average, None, 0)
            save_runs(filename, last_best, mean_average, None)


def test_runs_float(problem_name, problem_fitness, problem_domain):
    num_runs = 30
    dimensions = 25
    domain = [problem_domain] * dimensions
    num_generations = 1000
    population_size = 100
    mutation_method = muta_uni
    mutation_prob = (0.05, 0.1, 0.15)
    crossover_method = arithmetical_cross(0.5)
    crossover_prob = (0.7, 0.8, 0.9)
    parent_sel_method = tour_sel_minimization(5)
    survival_sel_method = sel_survivors_elite_minimization(0.2)
    fitness_func = problem_fitness


    for mp in mutation_prob:
        for cp in crossover_prob:
            last_best = []
            mean_average = []
            best = []
            for i in range(num_runs):
                print("MP: ", mp, " | CP: ", cp, " | Run: ",  i)
                best_final, fitness_best_gen, fitness_average_gen, best_overall = sea_float(domain,num_generations,population_size,mutation_method,mp,crossover_method,cp,parent_sel_method,survival_sel_method,fitness_func)
                last_best.append(fitness_best_gen[-1])
                mean_average.append(sum(fitness_average_gen)/len(fitness_average_gen))
                best.append(best_overall[0][1])

            filename = ("output/test_parameters/runs/", "test_" + problem_name + "_mp-" + str(mp) + "_cp-" + str(cp))
            save_plot_runs(filename, last_best, mean_average, best, 0)
            save_runs(filename, last_best, mean_average, best)



########################################################################################################################

if __name__ == "__main__":
    #test_problem_bin("OneMax", fitness_onemax)
    #test_problem_bin("JoaoBrandaosNumbers", fitness_jb)
    #test_problem_float("QuarticFunction", fitness_quartic, [-1.28, 1.28])
    #test_problem_float("RastriginFunction", fitness_rastrigin, [-5.12, 5.12])

    test_runs_bin("OneMax", fitness_onemax)
    test_runs_bin("JoaoBrandaosNumbers", fitness_jb)
    test_runs_float("QuarticFunction", fitness_quartic, [-1.28, 1.28])
    test_runs_float("RastriginFunction", fitness_rastrigin, [-5.12, 5.12])

    pass
