import logging
import os
import pickle

import numpy as np
from measure_image_similarity import structural_sim, pixel_sim, sift_sim, earth_movers_distance

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    OUTPUT_IMAGE = "./output.jpg"

    pop_file = "pop_data.pickle"
    gen_file = "gen_data.pickle"

    open(pop_file, "wb")
    open(gen_file, "wb")

    pop_data_dict = {}
    gen_data_dict = {}

    input_image = "../images/waterfall.jpg"

    logging.info("********** POPULATION SIZE **********")
    pop_pickle = pickle.load(open(pop_file, "rb"))

    if not pop_pickle:
        for pop in [3, 6, 10, 25, 100]:
            logging.info("***** Population size %s *****" % pop)

            avg_struct_sim = []
            avg_pix_sim = []
            avg_sif_sim = []
            avg_emd = []

            for i in range(10):
                logging.info("Run #%s" % (i + 1))

                os.system("python3 ../main.py %s 263 316 %s %s 10 0.05" % (input_image, OUTPUT_IMAGE, pop))

                avg_struct_sim.append(structural_sim(input_image, OUTPUT_IMAGE))
                avg_pix_sim.append(pixel_sim(input_image, OUTPUT_IMAGE))
                avg_sif_sim.append(sift_sim(input_image, OUTPUT_IMAGE))
                avg_emd.append(earth_movers_distance(input_image, OUTPUT_IMAGE))

            pop_data_dict[pop] = {"structural_similarity": np.mean(avg_struct_sim),
                                  "pixel_similarity": np.mean(avg_pix_sim),
                                  "sift_similarity": np.mean(avg_sif_sim), "earth_movers_distance": np.mean(avg_emd)}

            print(pop, pop_data_dict[pop])

        # Dump pop data
        pickle.dump(pop_data_dict, open(pop_file, "wb"))

    logging.info("********** NUMBER OF GENERATIONS **********")
    gen_pickle = pickle.load(open(gen_file, "rb"))

    if not gen_pickle:
        for gen in [5, 10, 25, 50, 100]:
            logging.info("***** %s Generations *****" % gen)

            avg_struct_sim = []
            avg_pix_sim = []
            avg_sif_sim = []
            avg_emd = []

            for i in range(10):
                logging.info("Run #%s" % (i + 1))

                os.system("python3 ../main.py %s 263 316 %s 10 %s 0.05" % (input_image, OUTPUT_IMAGE, gen))

                avg_struct_sim.append(structural_sim(input_image, OUTPUT_IMAGE))
                avg_pix_sim.append(pixel_sim(input_image, OUTPUT_IMAGE))
                avg_sif_sim.append(sift_sim(input_image, OUTPUT_IMAGE))
                avg_emd.append(earth_movers_distance(input_image, OUTPUT_IMAGE))

            gen_data_dict[gen] = {"structural_similarity": np.mean(avg_struct_sim),
                                  "pixel_similarity": np.mean(avg_pix_sim),
                                  "sift_similarity": np.mean(avg_sif_sim), "earth_movers_distance": np.mean(avg_emd)}

            print(gen, gen_data_dict[gen])

        # Dump pop data
        pickle.dump(gen_data_dict, open(gen_file, "wb"))
