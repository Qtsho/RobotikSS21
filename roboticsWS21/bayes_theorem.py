import numpy as np
import matplotlib.pyplot as plt


def BayesTheorem (pA, pBgA, pBgnA):
    pnotA = 1 -pA
    pB  = pBgA *pA + pBgnA*pnotA
    posterior = (pBgA*pA)/pB
    return posterior
pZleer_leer = 0.7
pZleer_voll = 0.2
pLeer = 0.5
posterior1 = BayesTheorem (pLeer, pZleer_leer,pZleer_voll)
print ("P1(leer| z= leer)", posterior1)

pZ2leer_leer = 0.5
pZ2leer_voll = 3/5

posterior2 = BayesTheorem (posterior1, pZ2leer_leer,pZ2leer_voll)
print ("P2(leer| z= leer)", posterior2)
posterior3 = BayesTheorem (posterior1, 0.9,0.2)
print ("P3(leer| z= leer)", posterior3)