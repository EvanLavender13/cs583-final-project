from gsc import construct_seam

if __name__ == "__main__":
    c1 = (2, [0, -1, 2, 0])
    c2 = (0, [3, 1, -1, 0])
    c3 = (0, [2, -1, -1, -1])

    print(construct_seam(c1))  # [(1, 1), (2, 1), (3, 2), (4, 2)]
    print(construct_seam(c2))  # [(1, 3), (2, 4), (3, 3), (4, 3)]
    print(construct_seam(c3))  # [(1, 2), (2, 1), (3, 0), (4, -1)] INVALID

