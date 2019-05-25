import sys

import cv2
import numpy as np

from gsc import get_energy_map, generate_population, get_fitness

if __name__ == "__main__":
    image = sys.argv[1]

    image = cv2.imread(image, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(grayscale.max(), image.max())

    print(image.shape)

    energy_map = get_energy_map(grayscale) / 255.0

    m = energy_map.shape[0]

    pivot = np.random.randint(low=0, high=m)

    population = generate_population(pivot, m, 10)

    print(population.shape)

    for individual in population:
        print(get_fitness(pivot, individual, energy_map))
