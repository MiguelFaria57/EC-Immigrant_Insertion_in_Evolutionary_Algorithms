"""
benchmarks.py

"""


import math
import random


# --------------------------------------------------

def phenotype(indiv):
    return indiv

def phenotype_jb(indiv):
    fen = [i+1 for i in range(len(indiv)) if indiv[i] == 1]
    return fen

# --------------------------------------------------

# ----- One Max

def fitness_onemax(indiv):
    return evaluate_onemax(phenotype(indiv))

def evaluate_onemax(indiv):
    return sum(indiv)


# ----- João Brandão’s Numbers

def fitness_jb(indiv):
    return evaluate_jb(phenotype_jb(indiv), len(indiv))

def evaluate_jb(indiv, comp):
    alfa = 1.0
    beta = 1.1
    return alfa * len(indiv) - beta * violations(indiv,comp)

def violations(indiv,comp):
    v = 0
    for elem in indiv:
        limite = min(elem-1,comp-elem)
        vi = 0
        for j in range(1,limite+1):
            if ((elem - j) in indiv) and ((elem+j) in indiv):
                vi += 1
        v += vi
    return v


# --------------------------------------------------

# ----- Quatic Function

def fitness_quartic(indiv):
    return quartic(phenotype(indiv))

def quartic(indiv):
    """
    quartic function (DeJong F4)
    domain = [-1.28, 1.28]
    minimum 0 at x = 0
    """
    return sum([(i+1) * indiv[i]**4 for i in range(len(indiv))]) + math.fabs(random.gauss(0, 1)) # sum([(i + 1) * x for i, x in enumerate(indiv)]) + uniform(0,1)


# ----- Rastrigin Function

def fitness_rastrigin(indiv):
    return rastrigin(phenotype(indiv))

def rastrigin(indiv):
    """
    rastrigin function
    domain = [-5.12, 5.12]
    minimum at (0,...,0)
    """
    n = len(indiv)
    A = 10
    return A * n + sum([x**2 - A * math.cos(2 * math.pi * x) for x in indiv])
