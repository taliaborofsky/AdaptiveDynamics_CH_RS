
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from fun_response_funs import *


def fun_fitness(x, M1, M2, **params):
    '''
    this is a subordinate's fitness of being in a group of size x
    @inputs:
    x - grp size
    M1 - pop size of big prey
    M2 - pop size of small prey
    params - a dictionary at least including: b1, b2, r, γ, a1, a2, h1, h2, α1_of_1, 
                                                    α2_of_1, s1, s2
    @returns:
    fitness of a subordinate, a float (or array if one of the parameter inputs is an array)

    @example:
    >>fun_fitness(x=np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2))
    array([0.24166667, 0.45833333, 0.53055556])
    '''
    fitnesses = fitness_from_prey(x, M1, M2,**params)
    total_sub_fitness = np.sum(fitnesses,0)
    return total_sub_fitness


def fitness_from_prey(x, M1, M2,b1, b2, r, γ,**params):
    '''
    portion of inclusive fitness from each prey type, stored in an array, after potentially unequal sharing
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    b1 - big prey  conversion (prey --> pred)
    b2 - small prey conversion (prey --> pred)
    r - relatedness between group members
    γ - extent of reproductive skew (portion of subordinate's food donated to dominant)
    params - dictionary of other parameters, which must at least contain 
             a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2

    @returns:
    np.array([<inclusive fitness from big prey>, <inclusive fitness from small prey>])
    (so the rows correspond to prey types

    @example
    >>fitness_from_prey(x= np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2))
    array([[0.08333333, 0.41666667, 0.52777778],
       [0.15833333, 0.04166667, 0.00277778]])
       
    '''
    
    wgroup = np.array([b1*fun_response(x,M1,M2,1,**params), b2*fun_response(x,M1,M2,2,**params)])
    try:
        if x > 1:
            repro_exchange = (1-γ)*(1-r) + r*x
            return 1/x *(wgroup) * repro_exchange
        else:
            return wgroup
    except ValueError:
        repro_exchange = np.ones(np.shape(x))
        repro_exchange[x>1] = (1-γ)*(1-r) + r*x[x>1]
        return (1/x) * wgroup * repro_exchange
def fun_fitness_from_big_prey(x, M1, M2, **params):
    '''
    portion of inclusive fitness from big prey type. calls fitness_from_prey
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    b1 - big prey  conversion (prey --> pred)
    b2 - small prey conversion (prey --> pred)
    params - dictionary of other parameters, which must at least contain 
             r, γ, a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2

    @returns:
    <inclusive fitness from big prey>
    '''
    return fitness_from_prey(x, M1, M2, **params)[0]
def fun_fitness_from_small_prey(x, M1, M2, **params):
    '''
    portion of inclusive fitness from small prey type. calls fitness_from_prey
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    b1 - big prey  conversion (prey --> pred)
    b2 - small prey conversion (prey --> pred)
    params - dictionary of other parameters, which must at least contain 
             r, γ, a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2

    @returns:
    <inclusive fitness from small prey>
    '''
    return fitness_from_prey(x, M1, M2, **params)[1]
