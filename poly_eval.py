from numpy.polynomial.polynomial import Polynomial as P


if __name__ == "__main__":
    p = P([1,2,3])
    # order is ^0 ^1 ^2 ^3...

    y = p(4)
    # == 1 + 2*4 + 3*16
    # == 57
    print(y)

