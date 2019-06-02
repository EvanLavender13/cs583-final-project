#!/usr/bin/env bash

mkdir -p output

mkdir -p output/small_tower
mkdir -p output/waterfall
mkdir -p output/mountain
mkdir -p output/whale

# Varied pop size, constant gens, constant mut rate, default stuff
###########################################################################
mkdir -p output/small_tower/pop
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/pop/05.jpg 5 10 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/pop/10.jpg 10 10 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/pop/25.jpg 25 10 0.05

mkdir -p output/waterfall/pop
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/pop/05_.jpg 5 10 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/pop/10.jpg 10 10 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/pop/25.jpg 25 10 0.05

mkdir -p output/mountain/pop
time python3 main.py images/mountain.jpg 269 304 output/mountain/pop/05.jpg 5 10 0.05
time python3 main.py images/mountain.jpg 269 304 output/mountain/pop/10.jpg 10 10 0.05
time python3 main.py images/mountain.jpg 269 304 output/mountain/pop/25.jpg 25 10 0.05

mkdir -p output/whale/pop
time python3 main.py images/whale.jpg 340 408 output/whale/pop/05.jpg 5 10 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/pop/10.jpg 10 10 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/pop/25.jpg 25 10 0.05
###########################################################################

# Constant pop size, varied number of gens, constant mut rate, default stuff
###########################################################################
mkdir -p output/small_tower/gen
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/gen/20.jpg 10 20 0.05
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/gen/30.jpg 10 30 0.05

mkdir -p output/waterfall/gen
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/gen/20.jpg 10 20 0.05
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/gen/30.jpg 10 30 0.05

mkdir -p output/mountain/gen
time python3 main.py images/mountain.jpg 269 304 output/mountain/gen/20.jpg 10 20 0.05
time python3 main.py images/mountain.jpg 269 304 output/mountain/gen/30.jpg 10 30 0.05

mkdir -p output/whale/gen
time python3 main.py images/whale.jpg 340 408 output/whale/gen/20.jpg 10 20 0.05
time python3 main.py images/whale.jpg 340 408 output/whale/gen/30.jpg 10 30 0.05
###########################################################################

# Constant pop size, constant number of gens, varied mut rate, default stuff
###########################################################################
mkdir -p output/small_tower/mutpb
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/mutpb/1.jpg 10 10 0.1
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/mutpb/15.jpg 10 10 0.15
time python3 main.py images/small_tower.jpg 242 286 output/small_tower/mutpb/25.jpg 10 10 0.25

mkdir -p output/waterfall/mutpb
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/mutpb/1.jpg 10 10 0.1
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/mutpb/15.jpg 10 10 0.15
time python3 main.py images/waterfall.jpg 263 316 output/waterfall/mutpb/25.jpg 10 10 0.25

mkdir -p output/mountain/mutpb
time python3 main.py images/mountain.jpg 269 304 output/mountain/mutpb/1.jpg 10 10 0.1
time python3 main.py images/mountain.jpg 269 304 output/mountain/mutpb/15.jpg 10 10 0.15
time python3 main.py images/mountain.jpg 269 304 output/mountain/mutpb/25.jpg 10 10 0.25

mkdir -p output/whale/mutpb
time python3 main.py images/whale.jpg 340 408 output/whale/mutpb/1.jpg 10 10 0.1
time python3 main.py images/whale.jpg 340 408 output/whale/mutpb/15.jpg 10 10 0.15
time python3 main.py images/whale.jpg 340 408 output/whale/mutpb/25.jpg 10 10 0.25
###########################################################################

# Mountain with Scharr
###########################################################################
mkdir -p output/mountain/scharr
time python3 main.py images/mountain.jpg 340 408 output/mountain/scharr/10.jpg 10 10 0.05 --scharr
time python3 main.py images/mountain.jpg 340 408 output/mountain/scharr/20.jpg 10 20 0.05 --scharr
time python3 main.py images/mountain.jpg 340 408 output/mountain/scharr/30.jpg 10 30 0.05 --scharr
###########################################################################

# Waterfall with tournament selection
###########################################################################
mkdir -p output/waterfall/selection
time python3 main.py images/waterfall.jpg 340 408 output/waterfall/selection/10_twopoint.jpg 10 10 0.05 --selection tournament
time python3 main.py images/waterfall.jpg 340 408 output/waterfall/selection/20_twopoint.jpg 10 20 0.05 --selection tournament
time python3 main.py images/waterfall.jpg 340 408 output/waterfall/selection/30_twopoint.jpg 10 30 0.05 --selection tournament
###########################################################################

# Whale with crossover variants
###########################################################################
mkdir -p output/whale/crossover
time python3 main.py images/whale.jpg 340 408 output/whale/crossover/10_twopoint.jpg 10 10 0.05 --crossover twopoint
time python3 main.py images/whale.jpg 340 408 output/whale/crossover/20_twopoint.jpg 10 20 0.05 --crossover twopoint
time python3 main.py images/whale.jpg 340 408 output/whale/crossover/30_twopoint.jpg 10 30 0.05 --crossover twopoint

time python3 main.py images/whale.jpg 340 408 output/whale/crossover/10_uniform.jpg 10 10 0.05 --crossover uniform
time python3 main.py images/whale.jpg 340 408 output/whale/crossover/20_uniform.jpg 10 20 0.05 --crossover uniform
time python3 main.py images/whale.jpg 340 408 output/whale/crossover/30_uniform.jpg 10 30 0.05 --crossover uniform
###########################################################################

# Whale with mutation variants
###########################################################################
mkdir -p output/whale/mutation
time python3 main.py images/whale.jpg 340 408 output/whale/mutation/10_shuffle.jpg 10 10 0.05 --mutation shuffle
time python3 main.py images/whale.jpg 340 408 output/whale/mutation/20_shuffle.jpg 10 20 0.05 --mutation shuffle
time python3 main.py images/whale.jpg 340 408 output/whale/mutation/30_shuffle.jpg 10 30 0.05 --mutation shuffle

time python3 main.py images/whale.jpg 340 408 output/whale/mutation/10_flipbit.jpg 10 10 0.05 --mutation flipbit
time python3 main.py images/whale.jpg 340 408 output/whale/mutation/20_flipbit.jpg 10 20 0.05 --mutation flipbit
time python3 main.py images/whale.jpg 340 408 output/whale/mutation/30_flipbit.jpg 10 30 0.05 --mutation flipbit
###########################################################################
