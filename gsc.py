import cv2
import numpy as np


def generate_population(pivot, m, size):
    return np.array([generate_individual(pivot, m) for i in range(size)])


def generate_individual(pivot, m):
    individual = np.random.randint(low=-1, high=2, size=m)
    individual[pivot] = np.random.randint(low=0, high=m)

    return individual


def get_fitness(pivot, individual, energy_map):
    seam = construct_seam(pivot, individual)

    m = energy_map.shape[0]

    sum = 0
    for coordinate in seam:
        if coordinate[1] < 0 or coordinate[1] > m - 1:
            return 0

        sum += energy_map[coordinate[0]][coordinate[1]]

    fitness = (m - sum) / m

    return fitness


def get_energy_map(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    cv2.imwrite("grad_x.jpg", grad_x)
    cv2.imwrite("grad_y.jpg", grad_y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    energy_map = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imwrite("energy_map.jpg", energy_map)

    return energy_map


def construct_seam(pivot, individual):
    """
    Constructs a seam from the given individual of the form:

        seam = {(i, f_v(i))} from i = 1, m

    Args:
        pivot: pivot position
        individual: genes represented as a list of integers

    Returns:
        seam coordinates

    """

    return np.array([(i, f_v(pivot, individual, i)) for i in range(len(individual))])


def f_v(pivot, individual, index):
    """
    Function used to construct a seam from an individual

    Args:
        pivot: pivot position
        individual: genes represented as a list of integers
        index: the index of the gene to transform

    Returns:
        corresponding coordinate of a gene

    """

    if index == pivot:
        return individual[index]
    elif index > pivot:
        return individual[index] + f_v(pivot, individual, index - 1)
    elif index < pivot:
        return individual[index] + f_v(pivot, individual, index + 1)
