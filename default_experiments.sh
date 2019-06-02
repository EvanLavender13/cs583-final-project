#!/usr/bin/env bash

mkdir -p output

# Varied pop size, constant gens, constant mut rate, default stuff
###########################################################################
mkdir -p output/small_tower
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/05_10_default.jpg 5 10 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_10_default.jpg 10 10 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/25_10_default.jpg 25 10 0.05

mkdir -p output/waterfall
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/05_10_default.jpg 5 10 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_10_default.jpg 10 10 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/25_10_default.jpg 25 10 0.05

mkdir -p output/mountain
time python3 main.py images/moutain.jpg 269 304 output/moutain/05_10_default.jpg 5 10 0.05
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_10_default.jpg 10 10 0.05
time python3 main.py images/moutain.jpg 269 304 output/moutain/25_10_default.jpg 25 10 0.05

mkdir -p output/whale
time python3 main.py images/whale.jpg 340 408 output/whale/05_10_default.jpg 5 10 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/10_10_default.jpg 10 10 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/25_10_default.jpg 25 10 0.05
###########################################################################

# Constant pop size, varied number of gens, constant mut rate, default stuff
###########################################################################
mkdir -p output/small_tower
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_20_default.jpg 10 20 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_30_default.jpg 10 30 0.05

mkdir -p output/waterfall
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_20_default.jpg 10 20 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_30_default.jpg 10 30 0.05

mkdir -p output/mountain
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_20_default.jpg 10 20 0.05
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_30_default.jpg 10 30 0.05

mkdir -p output/whale
time python3 main.py images/whale.jpg 340 408 output/whale/10_20_default.jpg 10 20 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/10_30_default.jpg 10 30 0.05
###########################################################################

# Constant pop size, constant number of gens, varied mut rate, default stuff
###########################################################################
mkdir -p output/small_tower
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_10_1_default.jpg 10 10 0.1
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_10_15_default.jpg 10 10 0.15
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/10_10_25_default.jpg 10 10 0.25

mkdir -p output/waterfall
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_10_1_default.jpg 10 10 0.1
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_10_15_default.jpg 10 10 0.15
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/10_10_25_default.jpg 10 10 0.25

mkdir -p output/mountain
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_10_1_default.jpg 10 10 0.1
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_10_15_default.jpg 10 10 0.15
time python3 main.py images/moutain.jpg 269 304 output/moutain/10_10_25_default.jpg 10 10 0.25

mkdir -p output/whale
time python3 main.py images/whale.jpg 340 408 output/whale/10_10_1_default.jpg 10 10 0.1
time python3 main.py images/whale.jpg 340 408 output/whale/10_10_15_default.jpg 10 10 0.15
time python3 main.py images/whale.jpg 340 408 output/whale/10_10_25_default.jpg 10 10 0.25
###########################################################################
