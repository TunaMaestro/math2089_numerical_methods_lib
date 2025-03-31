import numpy as np
import scipy.stats as stats
import pylab 

def pplot(xs):
    p = stats.probplot(xs, dist="norm", plot=pylab)
    pylab.show()
    return p

