# Genetic Seam Carving
> Genetic Seam Carving: A Genetic Algorithm Approach for Content-Aware Image Retargeting

Implementation of the genetic seam carving algorithm described in the references below. Genetic Seam Carving is an evolutionary algorithm for content-aware image resizing. 

![](https://github.com/EvanLavender13/cs583-final-project/blob/master/output/gifs/waterfall.gif)

## Requirements

Python 3

```sh
pip install -r requirements.txt
```

## Usage
```
usage: main.py [-h] [--energy {sobel,scharr}]
               [--selection {roulette,tournament}]
               [--crossover {onepoint,twopoint,uniform}]
               [--mutation {uniform,shuffle,flipbit}] [--display] [--verbose]
               input target_shape target_shape output pop_size num_gens mut_pb

Genetic Seam Carving

positional arguments:
  input                 Input image
  target_shape          Target image shape in 'row col' format
  output                Output image
  pop_size              Population size
  num_gens              Number of generations
  mut_pb                Mutation probability

optional arguments:
  -h, --help            show this help message and exit
  --energy {sobel,scharr}
                        Energy map gradient
  --selection {roulette,tournament}
                        Selection operator
  --crossover {onepoint,twopoint,uniform}
                        Crossover operator
  --mutation {uniform,shuffle,flipbit}
                        Mutation operator
  --display             Display visualization
  --verbose             Display information
```

## Examples

```sh
python3 main.py small_tower.jpg 242 286 smaller_tower.jpg 5 10 0.05
```

![](https://github.com/EvanLavender13/cs583-final-project/blob/master/images/small_tower.jpg) ![](https://github.com/EvanLavender13/cs583-final-project/blob/master/output/small_tower/pop/05.jpg)

```sh
python3 main.py whale.jpg 340 408 whale_out.jpg 10 30 0.05 --crossover uniform
```

![](https://github.com/EvanLavender13/cs583-final-project/blob/master/images/whale.jpg) 
![](https://github.com/EvanLavender13/cs583-final-project/blob/master/output/whale/crossover/30_uniform.jpg)

## References
- [Genetic Seam Carving: A Genetic Algorithm Approach for Content-Aware Image Retargeting](https://www.researchgate.net/publication/277132230_Genetic_Seam_Carving_A_Genetic_Algorithm_Approach_for_Content-Aware_Image_Retargeting)
- [An improved Genetic Algorithms-based Seam Carving method](https://www.researchgate.net/publication/299533436_An_improved_Genetic_Algorithms-based_Seam_Carving_method)
