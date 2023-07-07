"""
statistical_analysis.py

"""


import numpy as np
import scipy.stats as st
from utils import *
import warnings


# --------------------------------------------------

def ks_test(data):
    """
    Kolmogorov-Smirnov
    """
    if np.std(data) == 0:
        return 1.0, 1.0
    else:
        norm_data = (data - np.mean(data)) / (np.std(data) / np.sqrt(len(data)))
        return st.kstest(norm_data, 'norm')

def wilcoxon_test(data1, data2):
    """
    Wilcoxon
    non parametric
    two samples
    dependent
    """
    return st.wilcoxon(data1, data2)

def wilcoxon_test_effect_size(statistic, num_s, num_obs):
    mean = num_s * (num_s + 1) / 4
    std = np.sqrt(num_s * (num_s + 1) * (2 * num_s + 1) / 24)
    z_score = (statistic - mean) / std
    return z_score / np.sqrt(num_obs)

def t_test_dep(data1, data2):
    """
    Dependent T Test
    parametric
    two samples
    dependent
    """
    t, pval = st.ttest_rel(data1, data2)
    return t, pval


# --------------------------------------------------

def statistical_analysis(problem_name):
    print("\n\n##### " + problem_name + " #####\n")
    alpha = 0.05
    algorithm_labels = ("Default", "Random Immigrants", "Elitist Immigrants")
    filename = ("output/test_immigrants_insertion/runs/", "test_" + problem_name + "_all")
    algorithms = read_algorithms(filename)

    if all(x == algorithms[0] for x in algorithms):
        print("Data is equal. There are no differences.")
        return

    if problem_name == "JoaoBrandaosNumbers": # Verify if error that occurs with this problem is correct
        for a in algorithms:
            if 0 not in a:
                #print("There are no zeros in the data. Ignore the error message.")
                warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros. Switching to normal approximation.")
            else:
                print("There are zeros in the data. Handle the error message.")

    print("##### KS Test - Verify if the algorithms follow a normal distribution"
          "\n\tH0: Follow normal distribution"
          "\n\tH1: Does not follow normal distribution\n")
    normal_distribution = False
    ks_t = [[] for _ in range(len(algorithms))]
    for t in range(len(ks_t)):
        ks_t[t] = ks_test(algorithms[t])
        statistic = ks_t[t][0]
        p_value = ks_t[t][1]
        if p_value < alpha:
            print(algorithm_labels[t] + ":\n\tstatistic: " + str(round(statistic, 5)) + " | p-value: " + '{:.3e}'.format(p_value))
            print("\t" + '{:.3e}'.format(p_value) + " < " + str(alpha) + " -> Reject")
        else:
            print(algorithm_labels[t] + ":\n\tstatistic: " + str(round(statistic, 5)) + " | p-value: " + str(round(p_value, 5)))
            print("\t" + str(round(p_value, 5)) + " >= " + str(alpha), " -> Accept")
            normal_distribution = True

    if not normal_distribution:
        print("\n##### Wilcoxon Test - Verify if the algorithms using immigrant insertion differ from the default")
    else:
        print("\n##### Dependent T Test - Verify if the algorithms using immigrant insertion differ from the default")
    print("\tH0: Algorithms are equal\n\tH1: Algorithms differ\n")
    test = [[] for _ in range(len(algorithms)-1)]
    for t in range(1, 1+len(test)):
        if not normal_distribution:
            test[t-1] = wilcoxon_test(algorithms[0], algorithms[t])
        else:
            test[t-1] = t_test_dep(algorithms[0], algorithms[t])
        statistic = round(test[t-1][0], 5)
        p_value = round(test[t-1][1], 5)
        effect_size = round(wilcoxon_test_effect_size(statistic, 30, 60), 5)
        print(algorithm_labels[t] + ":\n\tstatistic: " + str(statistic) + " | p-value: " + str(p_value) + " | effect_size: " + str(effect_size))
        if p_value < alpha:
            print("\t" + str(p_value), " < ", str(alpha), " -> Reject")
        else:
            print("\t" + str(p_value), " >= ", str(alpha), " -> Accept")



########################################################################################################################

if __name__ == "__main__":
    statistical_analysis("OneMax")
    statistical_analysis("JoaoBrandaosNumbers")
    statistical_analysis("QuarticFunction")
    statistical_analysis("RastriginFunction")

    pass
