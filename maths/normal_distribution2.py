import random as rd
import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 1, 0.1 # mean and standard deviation
I = np.random.normal(mu, sigma, 10000)
R= 50


plt.subplots(figsize=(15,6 ))
plt.subplot(121)
count, bins, ignored = plt.hist(I, bins=1000, density=True)
#PDF
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
title = "I in Ampere: mu = %.2f,  sigma = %.2f" % (mu, sigma)
plt.ylabel("$f_I(i)$", fontsize=14)
plt.title(title, fontsize=18)

## Power
plt.subplot(122)
P = I**I * R
count, p_bins, ignored = plt.hist(P, bins=1000, density=True)
title = "P in Watt"
plt.title(title, fontsize=18)
plt.ylabel("$f_P(p)$", fontsize=14)
plt.xlim(35,70)
plt.show()