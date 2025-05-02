import matplotlib.pyplot as plt
import numpy as np
import lib
import sys



# Plotting a basic histogram
def hist(data):
    plt.hist(data, bins=30, color="skyblue", edgecolor="black")

    # Adding labels and title
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Basic Histogram")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    data = lib.interpret(sys.argv[1])
    
