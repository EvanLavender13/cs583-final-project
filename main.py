import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

from gsc import get_energy_map, get_fitness, construct_seam, f_v, cxSeam

if __name__ == "__main__":
    image = sys.argv[1]

    image = cv2.imread(image, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("grayscale.jpg", grayscale)

    ENERGY_MAP = get_energy_map(grayscale) / 255.0
    ROWS = ENERGY_MAP.shape[0]
    COLS = ENERGY_MAP.shape[1]
    PIVOT = np.random.randint(low=0, high=ROWS)

    print("PIVOT=", PIVOT)
    print(ENERGY_MAP.sum())
    print(ENERGY_MAP.max())

    POP_SIZE = 10
    CXPB = 0.1
    MUTPB = 0.1

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("rand_int", np.random.randint, low=-1, high=2)
    toolbox.register("rand_m", np.random.randint, low=0, high=COLS)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_int, n=ROWS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxSeam)
    toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=MUTPB)
    toolbox.register("select", tools.selRoulette, k=POP_SIZE)
    toolbox.register("evaluate", get_fitness, energy_map=ENERGY_MAP)

    # Initialize population and pivot position value
    population = toolbox.population(n=POP_SIZE)
    for individual in population:
        pivot = np.random.randint(low=0, high=ROWS)
        individual[pivot] = toolbox.rand_m()
        individual.pivot = pivot

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    NUM_GENS = 25

    stats = tools.Statistics(key=lambda indv: indv.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()

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
            mutant.pivot += np.random.randint(low=-2, high=2)
            mutant[mutant.pivot] = toolbox.rand_m()
            del mutant.fitness.values

        # Evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

        record = stats.compile(population)
        logbook.record(gen=g, **record)

    u_fit = np.mean([individual.fitness.values[0] for individual in population])

    print("u_fit=", u_fit)

    S = [individual for individual in population if individual.fitness.values[0] >= u_fit]

    print(len(S))

    k = 5

    m, n = image.shape[: 2]

    # output = np.zeros(shape=(m, n - 1, 3))
    output = image

    for individual in S:
        seam = construct_seam(individual)

        for row in range(m):
            col = seam[row][1]
            # output[row, :, 0] = np.delete(image[row, :, 0], [col])
            # output[row, :, 1] = np.delete(image[row, :, 1], [col])
            # output[row, :, 2] = np.delete(image[row, :, 2], [col])
            output[row][col] = 0.0

        cv2.imwrite("seam_removed.jpg", output)

    gen = logbook.select("gen")
    fit_max = logbook.select("avg")

    fig, ax = plt.subplots()
    line = ax.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness", color="b")

    plt.show()
