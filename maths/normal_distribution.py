import random as rd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.widgets import Cursor

summe = 0
n_wuerfeln= 10000

# Return random number between 1 and 6
def wuerfel():
    return rd.randint(1, 6)

def wuerfeln():
    a = wuerfel()
    b = wuerfel()
    summe = a + b 
    return summe

ergebnis = []
for i in range(n_wuerfeln):
    ergebnis.append(wuerfeln())
    
mu = np.mean(ergebnis)
std = np.std(ergebnis)

# Plot the histogram.
plt.hist(ergebnis, bins=11, density=True, alpha=0.5, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.show()