import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

import multiprocessing

from gsc import get_energy_map, get_fitness, construct_seam, cxSeam

if __name__ == "__main__":
    image = sys.argv[1]

    image = cv2.imread(image, cv2.IMREAD_COLOR)
    target_image = np.copy(image)

    ROWS = image.shape[0]
    COLS = image.shape[1]

    target_shape = (ROWS, COLS - 400)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("grayscale.jpg", grayscale)

    energy_map = get_energy_map(grayscale) / 255.0

    print("ROWS=", ROWS)
    print("COLS=", COLS)

    POP_SIZE = 6
    MUTPB = 0.1
    NUM_GENS = 5

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    pool = multiprocessing.Pool()

    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    toolbox.register("rand_int", np.random.randint, low=-1, high=2)
    toolbox.register("rand_m", np.random.randint, low=0, high=target_image.shape[1] - 1)
    toolbox.register("rand_p", np.random.randint, low=0, high=target_image.shape[0] - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_int, n=target_image.shape[0])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxSeam)
    toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=MUTPB)
    toolbox.register("select", tools.selRoulette, k=POP_SIZE)

    while target_image.shape[:2] != target_shape:
        toolbox.register("evaluate", get_fitness, energy_map=energy_map)

        # Initialize population and pivot position value
        population = toolbox.population(n=POP_SIZE)
        for individual in population:
            pivot = toolbox.rand_p(high=target_image.shape[0] - 1)
            individual[pivot] = toolbox.rand_m(high=target_image.shape[1] - 1)
            individual.pivot = pivot

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for g in range(NUM_GENS):

            # Select next generation
            offspring = toolbox.select(population, k=POP_SIZE)

            # Clone them
            offspring = list(map(toolbox.clone, offspring))

            # Mate them
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            # Mutate them
            for mutant in offspring:
                # Need to keep track of pivot value
                pivot_val = mutant[mutant.pivot]
                mutant[mutant.pivot] = 0
                toolbox.mutate(mutant)
                mutant.pivot = toolbox.rand_p(high=target_image.shape[0] - 1)
                mutant[mutant.pivot] = pivot_val
                del mutant.fitness.values

            # Evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

        u_max = np.max([individual.fitness.values[0] for individual in population])

        S = [individual for individual in population if individual.fitness.values[0] == u_max]

        for individual in S:
            m, n = target_image.shape[: 2]

            print("m, n=", m, n)

            seam = construct_seam(individual)

            output = np.zeros(shape=(m, n - 1, 3))
            output_energy = np.zeros(shape=(m, n - 1))

            for row in range(m):
                col = seam[row][1]
                output[row, :, 0] = np.delete(target_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(target_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(target_image[row, :, 2], [col])

                output_energy[row] = np.delete(energy_map[row], [col])

            target_image = np.copy(output)
            energy_map = np.copy(output_energy)

        cv2.imwrite("seam_removed.jpg", target_image)

