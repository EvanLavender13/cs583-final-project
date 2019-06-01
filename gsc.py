import cv2
import numpy as np
from deap import tools


def cxSeamUniform(ind1, ind2):
    ind1_pivot = to_balanced_ternary(ind1.pop(ind1.pivot))
    ind2_pivot = to_balanced_ternary(ind2.pop(ind2.pivot))

    pivot_gene_size = max(len(ind1_pivot), len(ind2_pivot))

    ind1_pivot_gene = [0] * pivot_gene_size
    ind1_pivot_gene[-len(ind1_pivot): 10] = ind1_pivot

    ind2_pivot_gene = [0] * pivot_gene_size
    ind2_pivot_gene[-len(ind2_pivot): 10] = ind2_pivot

    prob = ind1.fitness.values[0] / (ind1.fitness.values[0] + ind2.fitness.values[0])

    tools.cxUniform(ind1, ind2, indpb=prob)
    tools.cxUniform(ind1_pivot_gene, ind2_pivot_gene, indpb=prob)

    ind1.insert(ind1.pivot, to_integer(ind1_pivot_gene))
    ind2.insert(ind2.pivot, to_integer(ind2_pivot_gene))


def cxSeamTwoPoint(ind1, ind2):
    ind1_pivot = to_balanced_ternary(ind1.pop(ind1.pivot))
    ind2_pivot = to_balanced_ternary(ind2.pop(ind2.pivot))

    pivot_gene_size = max(len(ind1_pivot), len(ind2_pivot)) + 1

    ind1_pivot_gene = [0] * pivot_gene_size
    ind1_pivot_gene[-len(ind1_pivot): 10] = ind1_pivot

    ind2_pivot_gene = [0] * pivot_gene_size
    ind2_pivot_gene[-len(ind2_pivot): 10] = ind2_pivot

    tools.cxTwoPoint(ind1, ind2)
    tools.cxTwoPoint(ind1_pivot_gene, ind2_pivot_gene)

    ind1.insert(ind1.pivot, to_integer(ind1_pivot_gene))
    ind2.insert(ind2.pivot, to_integer(ind2_pivot_gene))


def cxSeamOnePoint(ind1, ind2):
    ind1_pivot = to_balanced_ternary(ind1.pop(ind1.pivot))
    ind2_pivot = to_balanced_ternary(ind2.pop(ind2.pivot))

    pivot_gene_size = max(len(ind1_pivot), len(ind2_pivot)) + 1

    ind1_pivot_gene = [0] * pivot_gene_size
    ind1_pivot_gene[-len(ind1_pivot): 10] = ind1_pivot

    ind2_pivot_gene = [0] * pivot_gene_size
    ind2_pivot_gene[-len(ind2_pivot): 10] = ind2_pivot

    tools.cxOnePoint(ind1, ind2)
    tools.cxOnePoint(ind1_pivot_gene, ind2_pivot_gene)

    ind1.insert(ind1.pivot, to_integer(ind1_pivot_gene))
    ind2.insert(ind2.pivot, to_integer(ind2_pivot_gene))


def to_integer(ternary):
    decimal = 0

    n = len(ternary)
    for i in range(n):
        decimal += (ternary[i] * (3 ** (n - i - 1)))

    return decimal


def to_balanced_ternary(n):
    if n == 0:
        return [0]

    ternary = []
    while n:
        ternary.insert(0, [0, 1, -1][n % 3])
        n = int(-~n / 3)

    return ternary


def get_fitness(individual, energy_map):
    seam = construct_seam(individual)

    m, n = energy_map.shape[: 2]

    sum = 1.0
    for coordinate in seam:
        if coordinate[1] < 0 or coordinate[1] > n - 3:
            return 1.0,

        sum += energy_map[coordinate[0]][coordinate[1]]

    fitness = sum / m
    fitness = 1 / (fitness ** 2)

    # print(sum, fitness)

    return fitness,


def get_energy_map_scharr(image):
    energy_map = np.absolute(cv2.Scharr(image, -1, 1, 0)) + np.absolute(cv2.Scharr(image, -1, 0, 1))

    cv2.imwrite("energy_map.jpg", energy_map)

    return energy_map


def get_energy_map_sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    energy_map = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)

    cv2.imwrite("energy_map.jpg", energy_map)

    return energy_map


def construct_seam(individual):
    """
    Constructs a seam from the given individual of the form:

        seam = {(i, f_v(i))} from i = 1, m

    Args:
        pivot: pivot position
        individual: genes represented as a list of integers

    Returns:
        seam coordinates

    """

    return np.array([(i, f_v(individual, i)) for i in range(len(individual))])


def f_v(individual, index):
    """
    Function used to construct a seam from an individual

    Args:
        pivot: pivot position
        individual: genes represented as a list of integers
        index: the index of the gene to transform

    Returns:
        corresponding coordinate of a gene

    """
    pivot = individual.pivot

    if index == pivot:
        return individual[index]
    elif index > pivot:
        return individual[index] + f_v(individual, index - 1)
    elif index < pivot:
        return individual[index] + f_v(individual, index + 1)
