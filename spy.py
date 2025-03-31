import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse


def spy_plot(matrix):
    # Create the spy plot
    plt.figure(figsize=(6, 6))
    plt.spy(matrix, markersize=5, color='b')

    # Add title and labels
    plt.title("Spy Plot of a Random Sparse Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Create a random sparse matrix
    size = 20  # Matrix size (20x20)
    density = 0.2  # Fraction of nonzero elements

    # Generate a random sparse matrix
    matrix = scipy.sparse.random(size, size, density=density, format='csr')
    
    spy_plot(matrix)
