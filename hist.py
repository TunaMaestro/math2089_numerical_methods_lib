import matplotlib.pyplot as plt
import numpy as np
import lib
import sys

data = lib.interpret(sys.argv[1])


# Plotting a basic histogram
plt.hist(data, bins=30, color="skyblue", edgecolor="black")

# Adding labels and title
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Basic Histogram")

# Display the plot
plt.show()
