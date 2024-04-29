import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from time import time

    
def fun_response(x,M1,M2,index,a1,a2,h1,h2,**params):
    α1 = fun_attack_rate(x,1,**params)
    α2 = fun_attack_rate(x,2,**params)
    if index == 1:
        numerator = a1*α1*M1
    elif index == 2:
        numerator = a2*α2*M2
    denominator = 1 + a1*α1*h1*M1 + a2*α2*h2*M2
    return numerator/denominator

def fun_attack_rate(x, index, α1_of_1, α2_of_1, s1, s2, **params):
    if index == 1:
        θ_1 = - np.log(1/α1_of_1 - 1)/(1-s1)
        return 1/(1 + np.exp(- θ_1 * (x - s1)))
    elif index == 2:
        θ_2 = - np.log(1/α2_of_1 - 1)/(1-s2)
        return 1/(1 + np.exp(- θ_2 * (x - s2)))
    
