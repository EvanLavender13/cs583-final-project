import logging
import os

import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

from measure_image_similarity import earth_movers_distance, structural_sim, pixel_sim, sift_sim


def mean_squared_error(image1, image2):
    error = np.sum((image1.astype(np.float) - image2.astype(np.float)) ** 2)
    error /= np.float(image1.shape[0] * image1.shape[1])

    return error


def compare_images(image1, image2):
    m = mean_squared_error(image1, image2)
    s = ssim(image1, image2, multichannel=True)

    return m, s


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    mountain_dir = "output/mountain"
    small_tower_dir = "output/small_tower"
    waterfall_dir = "output/waterfall"
    whale_dir = "output/whale"

    test_image = "images/mountain.jpg"

    for subdir in ["/pop/", "/gen/", "/mutpb/", "/scharr/"]:
        logging.info("********** Mountain image - " + subdir)
        for file in os.listdir(mountain_dir + subdir):
            comp_image = mountain_dir + subdir + file
            logging.info("Image file %s" % (mountain_dir + subdir + file))

            struct_sim = structural_sim(test_image, comp_image)
            pix_sim = pixel_sim(test_image, comp_image)
            sif_sim = sift_sim(test_image, comp_image)
            emd = earth_movers_distance(test_image, comp_image)

            logging.info((struct_sim, pix_sim, sif_sim, emd))
            logging.info("")

    test_image = "output/sc_images/sc_small_tower.jpg"

    for subdir in ["/pop/", "/gen/", "/mutpb/"]:
        logging.info("********** Small Tower image - " + subdir)
        for file in os.listdir(small_tower_dir + subdir):
            comp_image = small_tower_dir + subdir + file
            logging.info("Image file %s" % (small_tower_dir + subdir + file))

            struct_sim = structural_sim(test_image, comp_image)
            pix_sim = pixel_sim(test_image, comp_image)
            sif_sim = sift_sim(test_image, comp_image)
            emd = earth_movers_distance(test_image, comp_image)

            logging.info((struct_sim, pix_sim, sif_sim, emd))
            logging.info("")

    test_image = "output/sc_images/sc_waterfall.jpg"

    for subdir in ["/pop/", "/gen/", "/mutpb/", "/selection/"]:
        logging.info("********** Waterfall image - " + subdir)
        for file in os.listdir(waterfall_dir + subdir):
            comp_image = waterfall_dir + subdir + file
            logging.info("Image file %s" % (waterfall_dir + subdir + file))

            struct_sim = structural_sim(test_image, comp_image)
            pix_sim = pixel_sim(test_image, comp_image)
            sif_sim = sift_sim(test_image, comp_image)
            emd = earth_movers_distance(test_image, comp_image)

            logging.info((struct_sim, pix_sim, sif_sim, emd))
            logging.info("")

    test_image = "output/sc_images/sc_whale.jpg"

    for subdir in ["/pop/", "/gen/", "/mutpb/", "/crossover/", "/mutation/"]:
        logging.info("********** Whale image - " + subdir)
        for file in os.listdir(whale_dir + subdir):
            comp_image = whale_dir + subdir + file
            logging.info("Image file %s" % (whale_dir + subdir + file))

            struct_sim = structural_sim(test_image, comp_image)
            pix_sim = pixel_sim(test_image, comp_image)
            sif_sim = sift_sim(test_image, comp_image)
            emd = earth_movers_distance(test_image, comp_image)

            logging.info((struct_sim, pix_sim, sif_sim, emd))
            logging.info("")
