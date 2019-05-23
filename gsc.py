def fitness(individual):
    seam = construct_seam(individual)


def construct_seam(individual):
    """
    Constructs a seam from the given individual of the form:

        seam = {(i, f_v(i))} from i = 1, m

    Args:
        individual: tuple consisting of pivot position and genes

    Returns:
        seam coordinates

    """
    pivot = individual[0]
    genes = individual[1]

    return [(i + 1, f_v(genes, pivot, i)) for i in range(len(genes))]


def f_v(genes, pivot, index):
    """
    Function used to construct a seam from an individual

    seam = {(i, f_v(i))} from i = 1, m

    Args:
        genes: list of integer values
        pivot: pivot position
        index: the index of the gene to transform

    Returns:
        corresponding coordinate of a gene

    """

    if index == pivot:
        return genes[index]
    elif index > pivot:
        return genes[index] + f_v(genes, pivot, index - 1)
    elif index < pivot:
        return genes[index] + f_v(genes, pivot, index + 1)
