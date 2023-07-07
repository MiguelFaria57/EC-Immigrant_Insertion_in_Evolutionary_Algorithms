"""
utils.py

"""


import matplotlib.pyplot as plt
import pickle


# --------------------------------------------------

def display_best(best, gen=None):
    if gen is None:
        print('Chromo: %s\nFitness: %s\n' %(best[0], best[1]))
    else:
        print('Chromo: %s\nFitness: %s\nGen: %s\n' %(best[0], best[1], gen))


def save_plot_best_and_average(problem_name, filename, best, average, best_overall, show):
    lim_inf = 0
    lim_sup = 0
    if problem_name == "OneMax":
        lim_inf = round(max(best) - max(best)/10)
        lim_sup = round(max(best) + 2)
    if problem_name == "JoaoBrandaosNumbers":
        lim_inf = round(max(best) - max(best)/2)
        lim_sup = round(max(best) + 2)
    if problem_name == "QuarticFunction":
        lim_inf = round(best_overall[0][1]-0.1, 2)
        lim_sup = round(best_overall[0][1]+0.5, 2)
    if problem_name == "RastriginFunction":
        lim_inf = round(best_overall[0][1]-0.1, 2)
        lim_sup = round(best_overall[0][1]+5, 2)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
    generations = list(range(len(best)))
    plt.suptitle(filename[1])
    ax1.set_title('Performance over generations', fontsize=10)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.plot(generations, best, label='Best')
    ax1.plot(generations,average,label='Average')
    ax1.legend(loc='best')
    ax1.annotate(str(round(best[-1],5)), xy=(generations[-1], best[-1]), xytext=(generations[-1] + 0.2, best[-1]))
    ax1.annotate(str(round(average[-1],5)), xy=(generations[-1], average[-1]), xytext=(generations[-1] + 0.2, average[-1]))

    ax2.set_ylim(lim_inf, lim_sup)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.plot(generations, best, label='Best')
    ax2.plot(generations, average, label='Average')
    ax2.legend(loc='best')
    ax2.annotate(str(round(best[-1],5)), xy=(generations[-1], best[-1]), xytext=(generations[-1] + 0.2, best[-1]))
    ax2.annotate(str(round(average[-1],5)), xy=(generations[-1], average[-1]), xytext=(generations[-1] + 0.2, average[-1]))
    if best_overall is not None and round(best[-1], 5) !=  round(best_overall[0][1], 5):
        ax2.annotate(str(round(best_overall[0][1],5)), xy=(best_overall[1], best_overall[0][1]), xytext=(best_overall[1] + 0.2, best_overall[0][1]))
    if show:
        plt.show()
    plt.savefig(filename[0] + "images/" + filename[1] + ".png")
    plt.close()

def save_best_and_average(filename, best, average):
    with open(filename[0] + "values/" + filename[1] + ".txt", 'w') as f:
        f.write(','.join(map(str, best)) + '\n')
        f.write(','.join(map(str, average)) + '\n')

def read_best_and_average(filename):
    with open(filename[0] + "values/" + filename[1] + ".txt", 'r') as f:
        lines = f.readlines()
    best = list(map(float, lines[0].strip().split(',')))
    average = list(map(float, lines[1].strip().split(',')))
    return best, average

def save_best_overall(filename, best_overall):
    with open(filename[0] + "values/" + filename[1] + ".pkl", 'wb') as f:
        pickle.dump(best_overall, f)

def read_best_overall(filename):
    with open(filename[0] + "values/" + filename[1] + ".pkl", 'rb') as f:
        best_overall = pickle.load(f)
    return best_overall


def save_plot_runs(filename, best, average, best_overall, show):
    if best_overall is None:
        plt.figure(figsize=(10, 6))
        runs = list(range(len(best)))
        plt.suptitle(filename[1])
        plt.title('Performance over runs', fontsize=10)
        plt.xlabel('Run')
        plt.ylabel('Fitness')
        plt.plot(runs, best, label='Last Best')
        plt.plot(runs, average, label='Mean of Averages')
        plt.legend(loc='best')
        plt.annotate(str(round(best[-1], 5)), xy=(runs[-1], best[-1]), xytext=(runs[-1] + 0.2, best[-1]))
        plt.annotate(str(round(average[-1], 5)), xy=(runs[-1], average[-1]),xytext=(runs[-1] + 0.2, average[-1]))
    else:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
        runs = list(range(len(best)))
        plt.suptitle(filename[1])
        ax1.set_title('Performance over runs', fontsize=10)
        ax1.set_xlabel('Run')
        ax1.set_ylabel('Fitness')
        ax1.plot(runs, best, label='Last Best')
        ax1.plot(runs, average, label='Mean of Averages')
        if best_overall != best:
            ax1.plot(runs, best_overall, label='Best Overall')
        ax1.legend(loc='best')
        ax1.annotate(str(round(best[-1], 5)), xy=(runs[-1], best[-1]), xytext=(runs[-1] + 0.2, best[-1]))
        ax1.annotate(str(round(average[-1], 5)), xy=(runs[-1], average[-1]), xytext=(runs[-1] + 0.2, average[-1]))
        if round(best[-1], 5) != round(best_overall[-1], 5):
            plt.annotate(str(round(best_overall[-1], 5)), xy=(runs[-1], best_overall[-1]), xytext=(runs[-1] + 0.2, best_overall[-1]))
        ax2.set_ylim(round(min(best) - 0.08, 2), round(max(best) + 0.08, 2))
        text_legend = []
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Fitness')
        ax2.plot(runs, best, label='Last Best')
        text_legend += ['Last Best | Mean: {:.5f}'.format(sum(best) / len(best))]
        ax2.plot(runs, average, label='Mean of Averages')
        if best_overall != best:
            ax2.plot(runs, best_overall, label='Best Overall')
            text_legend += ['Best Overall | Mean: {:.5f}'.format(sum(best_overall) / len(best_overall))]
        ax2.legend(loc='best', labels=text_legend)
        ax2.annotate(str(round(best[-1], 5)), xy=(runs[-1], best[-1]), xytext=(runs[-1] + 0.2, best[-1]))
        ax2.annotate(str(round(average[-1], 5)), xy=(runs[-1], average[-1]), xytext=(runs[-1] + 0.2, average[-1]))
        if round(best[-1], 5) != round(best_overall[-1], 5):
            plt.annotate(str(round(best_overall[-1], 5)), xy=(runs[-1], best_overall[-1]), xytext=(runs[-1] + 0.2, best_overall[-1]))
    if show:
        plt.show()
    plt.savefig(filename[0] + "images/" + filename[1] + ".png")
    plt.close()


def save_runs(filename, best, average, best_overall):
    with open(filename[0] + "values/" + filename[1] + ".txt", 'w') as f:
        f.write(','.join(map(str, best)) + '\n')
        f.write(','.join(map(str, average)) + '\n')
        if best_overall is not None:
            f.write(','.join(map(str, best_overall)) + '\n')


def save_plot_algorithms(problem_name, filename, algo0, algo1, algo2, show, plot_type):
    if plot_type == 0:
        plt.figure(figsize=(10, 6))
        generations = list(range(len(algo1)))
        plt.suptitle(filename[1])
        plt.title('Performance over runs', fontsize=10)
        plt.xlabel('Run')
        plt.ylabel('Fitness')
        plt.plot(generations, algo0, label='Default')
        plt.plot(generations, algo1, label='Random Immigrants')
        plt.plot(generations, algo2, label='Elitist Immigrants')
        text_legend = ['Default | Mean: {:.5f}'.format(sum(algo0)/len(algo0)), 'Random Immigrants | Mean: {:.5f}'.format(sum(algo1)/len(algo1)), 'Elitist Immigrants | Mean: {:.5f}'.format(sum(algo2)/len(algo2))]
        plt.legend(loc='best', labels=text_legend)
        plt.annotate(str(round(algo0[-1], 5)), xy=(generations[-1], algo0[-1]), xytext=(generations[-1] + 0.2, algo0[-1]))
        plt.annotate(str(round(algo1[-1], 5)), xy=(generations[-1], algo1[-1]), xytext=(generations[-1] + 0.2, algo1[-1]))
        plt.annotate(str(round(algo2[-1], 5)), xy=(generations[-1], algo2[-1]), xytext=(generations[-1] + 0.2, algo2[-1]))

    elif plot_type == 1:
        lim_inf = 0
        lim_sup = 0
        max_val = max(algo0 + algo1 + algo2)
        min_val = min(algo0 + algo1 + algo2)
        if problem_name == "OneMax":
            lim_inf = round(max_val - max_val/10)
            lim_sup = round(max_val + 2)
        if problem_name == "JoaoBrandaosNumbers":
            lim_inf = round(max_val - max_val/2)
            lim_sup = round(max_val + 2)
        if problem_name == "QuarticFunction":
            lim_inf = round(min_val - 0.1, 2)
            lim_sup = round(min_val + 0.5, 2)
        if problem_name == "RastriginFunction":
            lim_inf = round(min_val - 0.1, 2)
            lim_sup = round(min_val + 5, 2)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
        generations = list(range(len(algo0)))
        plt.suptitle(filename[1])
        ax1.set_title('Performance over generations', fontsize=10)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.plot(generations, algo0, label='Default')
        ax1.plot(generations, algo1, label='Random Immigrants')
        ax1.plot(generations, algo2, label='Elitist Immigrants')
        ax1.legend(loc='best')

        ax2.set_ylim(lim_inf, lim_sup)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.plot(generations, algo0, label='Default')
        ax2.plot(generations, algo1, label='Random Immigrants')
        ax2.plot(generations, algo2, label='Elitist Immigrants')
        if (problem_name == "QuarticFunction" or problem_name == "RastriginFunction") and (min(algo0)!=algo0[-1] or min(algo1)!=algo1[-1] or min(algo2)!=algo2[-1]):
            text_legend = ['Default | Best: {:.5f}'.format(min(algo0)), 'Random Immigrants | Best: {:.5f}'.format(min(algo1)), 'Elitist Immigrants | Best: {:.5f}'.format(min(algo2))]
            ax2.legend(loc='best', labels=text_legend)
        else:
            ax2.legend(loc='best')
        ax2.annotate(str(round(algo0[-1], 5)), xy=(generations[-1], algo0[-1]), xytext=(generations[-1] + 0.2, algo0[-1]))
        ax2.annotate(str(round(algo1[-1], 5)), xy=(generations[-1], algo1[-1]), xytext=(generations[-1] + 0.2, algo1[-1]))
        ax2.annotate(str(round(algo2[-1], 5)), xy=(generations[-1], algo2[-1]), xytext=(generations[-1] + 0.2, algo2[-1]))

    if show:
        plt.show()
    plt.savefig(filename[0] + "images/" + filename[1] + ".png")
    plt.close()

def save_algorithms(filename, algo0, algo1, algo2):
    with open(filename[0] + "values/" + filename[1] + ".txt", 'w') as f:
        f.write(','.join(map(str, algo0)) + '\n')
        f.write(','.join(map(str, algo1)) + '\n')
        f.write(','.join(map(str, algo2)) + '\n')

def read_algorithms(filename):
    with open(filename[0] + "values/" + filename[1] + ".txt", 'r') as f:
        lines = f.readlines()
    algo0 = list(map(float, lines[0].strip().split(',')))
    algo1 = list(map(float, lines[1].strip().split(',')))
    algo2 = list(map(float, lines[2].strip().split(',')))
    return algo0, algo1, algo2


def save_seeds(filename, seeds):
    with open(filename + ".txt", 'w') as f:
        f.write("Run Seed\n")
        for r, s in enumerate(seeds):
            f.write(str(r) + " " + str(s) + "\n")
