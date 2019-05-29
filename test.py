from gsc import to_balanced_ternary, to_integer

if __name__ == "__main__":
    dec = 2343

    tern = to_balanced_ternary(dec)

    print(dec, tern)

    dec = to_integer(tern)

    print(dec, tern)
