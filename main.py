import multiprocessing
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

from gsc import get_energy_map, get_fitness, construct_seam, cxSeam

if __name__ == "__main__":
    image = sys.argv[1]

    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_image = np.copy(image)

    print(image.shape, target_image.shape)

    ROWS = image.shape[0]
    COLS = image.shape[1]

    target_shape = (ROWS, COLS - 75)

    print("ROWS=", ROWS)
    print("COLS=", COLS)

    POP_SIZE = 6
    MUTPB = 0.5
    NUM_GENS = 5
    SELECTION = "tournament"

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    pool = multiprocessing.Pool()

    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    toolbox.register("gene_value", np.random.randint, low=-1, high=2)
    toolbox.register("pivot_value", np.random.randint, low=0)
    toolbox.register("pivot_index", np.random.randint, low=0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene_value, n=ROWS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxSeam)
    toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=MUTPB)

    if SELECTION == "roulette":
        toolbox.register("select", tools.selRoulette, k=POP_SIZE)
    elif SELECTION == "tournament":
        toolbox.register("select", tools.selTournament, k=POP_SIZE, tournsize=3)

    fig, (img_plot, tar_plot) = plt.subplots(1, 2, figsize=(12, 4))
    img_plot.axis("off")
    tar_plot.axis("off")
    plt.ion()

    img_data = img_plot.imshow(image)
    tar_data = tar_plot.imshow(target_image)

    while target_image.shape[:2] != target_shape:
        grayscale = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        energy_map = get_energy_map(grayscale) / 255.0

        # Need to register evaluate function to take in updated energy map
        toolbox.register("evaluate", get_fitness, energy_map=energy_map)

        # Initialize population and pivot position value
        population = toolbox.population(n=POP_SIZE)
        for individual in population:
            pivot = toolbox.pivot_index(high=target_image.shape[0] - 1)
            individual[pivot] = toolbox.pivot_value(high=target_image.shape[1] - 1)
            individual.pivot = pivot

        # Evaluate the entire population
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for g in range(NUM_GENS):

            # Select next generation
            offspring = toolbox.select(population, k=POP_SIZE)

            # Clone them
            offspring = list(toolbox.map(toolbox.clone, offspring))

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

                if np.random.random() < MUTPB:
                    mutant.pivot = toolbox.pivot_index(high=target_image.shape[0] - 1)
                    mutant[mutant.pivot] = toolbox.pivot_value(high=target_image.shape[1] - 1)
                else:
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

        S = [individual for individual in population if individual.fitness.values[0] == u_max and u_max > 1.0]

        for individual in S:
            m, n = target_image.shape[: 2]

            print("m, n=", m, n)

            seam = construct_seam(individual)

            output = np.zeros(shape=(m, n - 1, 3))

            for row in range(m):
                col = seam[row][1]

                output[row, :, 0] = np.delete(target_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(target_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(target_image[row, :, 2], [col])

                col_diff = image.shape[1] - target_image.shape[1]
                image[row, col + col_diff] = [255, 0, 0]

            target_image = np.copy(output)

        img_data.set_data(image.astype(np.int))
        tar_data.set_data(
            np.hstack([target_image.astype(np.int),
                       np.zeros(shape=(ROWS, COLS - target_image.shape[1], 3), dtype=np.uint8)]))

        plt.draw()
        plt.pause(0.00001)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite("seams_removed.jpg", target_image)
    cv2.imwrite("seams_shown.jpg", image)
