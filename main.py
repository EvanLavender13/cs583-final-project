import argparse
import logging
import multiprocessing

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools

from util.gsc import get_energy_map_sobel, get_energy_map_scharr, cxSeamOnePoint, cxSeamTwoPoint, cxSeamUniform, \
    get_fitness, construct_seam

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description="Genetic Seam Carving")

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument("target_shape", type=int, nargs=2, help="Target image shape in 'row col' format")
    parser.add_argument("output", type=str, help="Output image")

    parser.add_argument("pop_size", type=int, help="Population size")
    parser.add_argument("num_gens", type=int, help="Number of generations")
    parser.add_argument("mut_pb", type=float, help="Mutation probability")

    parser.add_argument("--energy", type=str, choices=["sobel", "scharr"], default="sobel", help="Energy map gradient")

    parser.add_argument("--selection", type=str, choices=["roulette", "tournament"], default="roulette",
                        help="Selection operator")
    parser.add_argument("--crossover", type=str, choices=["onepoint", "twopoint", "uniform"], default="onepoint",
                        help="Crossover operator")
    parser.add_argument("--mutation", type=str, choices=["uniform", "shuffle", "flipbit"], default="uniform",
                        help="Mutation operator")

    parser.add_argument("--display", action="store_true", help="Display visualization")
    parser.add_argument("--verbose", action="store_true", help="Display information")

    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_image = np.copy(image)

    ROWS = image.shape[0]
    COLS = image.shape[1]

    # Vertical seams only, for now
    target_shape = (ROWS, args.target_shape[1])

    # Get genetic algorithm parameters
    pop_size = args.pop_size
    num_gens = args.num_gens
    mut_pb = args.mut_pb

    energy = args.energy

    # Select genetic operators
    selection = args.selection
    crossover = args.crossover
    mutation = args.mutation

    # Stuff to look at
    display = args.display
    verbose = args.verbose

    logging.info("Carving %s to size %s and saving result to %s" % (args.input, target_shape, args.output))
    logging.info("Evolving populations of size %s for %s generations with a mutation probability of %s" %
                 (pop_size, num_gens, mut_pb))
    logging.info("Energy function: %s" % energy)
    logging.info("Selection operator: %s" % selection)
    logging.info("Crossover operator: %s" % crossover)
    logging.info("Mutation operator: %s" % mutation)

    # Select energy map function
    if energy == "sobel":
        get_energy_map = get_energy_map_sobel
    elif energy == "scharr":
        get_energy_map = get_energy_map_scharr

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # For multi-core support
    # Calls to toolbox.map will be multi-threaded (or something)
    pool = multiprocessing.Pool()

    # Set up some handy dandy tools
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    toolbox.register("gene_value", np.random.randint, low=-1, high=2)
    toolbox.register("pivot_value", np.random.randint, low=0)
    toolbox.register("pivot_index", np.random.randint, low=0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene_value, n=ROWS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register selection operator based on args
    if selection == "roulette":
        toolbox.register("select", tools.selRoulette, k=pop_size)
    elif selection == "tournament":
        toolbox.register("select", tools.selTournament, k=pop_size, tournsize=3)

    # Register crossover operator based on args
    if crossover == "onepoint":
        toolbox.register("mate", cxSeamOnePoint)
    elif crossover == "twopoint":
        toolbox.register("mate", cxSeamTwoPoint)
    elif crossover == "uniform":
        toolbox.register("mate", cxSeamUniform)

    # Register mutation operator based on args
    if mutation == "uniform":
        toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=mut_pb)
    elif mutation == "shuffle":
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mut_pb)
    elif mutation == "flipbit":
        toolbox.register("mutate", tools.mutFlipBit, indpb=mut_pb)

    if display:
        plt.figure(figsize=(8, 4))
        plt.axis("off")
        plt.ion()

        img_data = plt.imshow(image)

    # Go until we have the right shape
    while target_image.shape[:2] != target_shape:
        # Convert to grayscale and get energy map
        grayscale = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        energy_map = get_energy_map(grayscale) / 255.0

        # Need to re-register evaluate function to take in updated energy map
        toolbox.register("evaluate", get_fitness, energy_map=energy_map)

        # Initialize population and pivot values
        population = toolbox.population(n=pop_size)
        for individual in population:
            pivot = toolbox.pivot_index(high=target_image.shape[0] - 1)
            individual[pivot] = toolbox.pivot_value(high=target_image.shape[1] - 1)
            individual.pivot = pivot

        # Evaluate the entire population
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for g in range(num_gens):
            # Select next generation
            offspring = toolbox.select(population, k=pop_size)

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

                if np.random.random() < mut_pb:
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

        if verbose:
            logging.info("Maximum fitness: %s" % u_max)

        S = [individual for individual in population if individual.fitness.values[0] == u_max and u_max > 1.0]

        for individual in S:
            m, n = target_image.shape[: 2]

            seam = construct_seam(individual)

            output = np.zeros(shape=(m, n - 1, 3))

            if display:
                display_output = np.copy(target_image)

            for row in range(m):
                col = seam[row][1]

                output[row, :, 0] = np.delete(target_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(target_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(target_image[row, :, 2], [col])

                if display:
                    display_output[row, col] = [255, 0, 0]

            target_image = np.copy(output)

        if display:
            img_data.set_data(np.hstack([display_output.astype(np.int),
                                         np.zeros(shape=(ROWS, COLS - target_image.shape[1], 3), dtype=np.uint8)]))

            plt.draw()
            plt.pause(0.00001)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(args.output, target_image)
