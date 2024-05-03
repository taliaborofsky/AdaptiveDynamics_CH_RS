
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
                                                    α2_of_1, s1, s2, limited_portions
                                                    (and must also have b0 if limited_portions = True)
    @returns:
    fitness of a subordinate, a float (or array if one of the parameter inputs is an array)

    @example:
    >>fun_fitness(x=np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2, 
                                                    limited_portions = False))
    array([0.24166667, 0.45833333, 0.53055556])
    '''
    fitnesses = fitness_from_prey(x, M1, M2,**params)
    total_sub_fitness = np.sum(fitnesses,0)
    return total_sub_fitness



def per_capita_fitness_from_prey(x,M1,M2,b1,b2,limited_portions, **params):
    '''
    portion of direct fitness from each prey type without any skew, stored in an array
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    b1 - big prey  conversion (prey --> pred)
    b2 - small prey conversion (prey --> pred)
    limited_portions - True or False, whether predators can only eat a limited amount or not
    params - dictionary of other parameters, which must at least contain 
             a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2

    @returns:
    np.array([<inclusive fitness from big prey>, <inclusive fitness from small prey>])
    (so the rows correspond to prey types

    @example
    >>per_capita_fitness_from_prey(x= np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, 
                                                        h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2, 
                                                    limited_portions = False))
    array([[0.08333333, 0.41666667, 0.52777778],
       [0.15833333, 0.04166667, 0.00277778]])
    
    >>per_capita_fitness_from_prey(x= np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, 
                                                        h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2,b0 = 0.05, 
                                                    limited_portions = True))
    array([[0.00416667, 0.04166667, 0.07916667],
       [0.07916667, 0.04166667, 0.00277778]])
    '''
    # set portion size, need to account for x being an array
    conversion_big, conversion_small = conversion_prey(x,b1,b2,limited_portions,**params)

        
    w_per_capita = np.array([conversion_big*fun_response(x,M1,M2,1,**params), 
                         conversion_small*fun_response(x,M1,M2,2,**params)])
    return w_per_capita
    
def fitness_from_prey(x, M1, M2,b1, b2, r, γ,limited_portions,**params):
    '''
    portion of inclusive fitness from each prey type, stored in an array, after potentially unequal sharing
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    r - relatedness between group members
    γ - extent of reproductive skew (portion of subordinate's food donated to dominant)
    params - dictionary of other parameters, which must at least contain 
             a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2, b1, b2, limited_portions

    @returns:
    np.array([<inclusive fitness from big prey>, <inclusive fitness from small prey>])
    (so the rows correspond to prey types

    @example
    >>fitness_from_prey(x= np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, 
                                                        h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2, 
                                                    limited_portions = False))
    array([[0.08333333, 0.41666667, 0.52777778],
       [0.15833333, 0.04166667, 0.00277778]])
    
    >>fitness_from_prey(x= np.array([1,2,3]), M1=10, M2=10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, 
                                                        h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2,b0 = 0.05, 
                                                    limited_portions = True))
    array([[0.00416667, 0.04166667, 0.07916667],
       [0.07916667, 0.04166667, 0.00277778]])
    '''
    # set portion size, need to account for x being an array

        
    w_per_capita = per_capita_fitness_from_prey(x,M1,M2, **params)
    try:
        if x > 1:
            repro_exchange = (1-γ)*(1-r) + r*x
            return w_per_capita * repro_exchange
        else:
            return w_per_capita
    except ValueError:
        repro_exchange = np.ones(np.shape(x))
        repro_exchange[x>1] = (1-γ)*(1-r) + r*x[x>1]
        return  w_per_capita * repro_exchange

def conversion_prey(x,b1,b2,limited_portions,**params):
    '''
    Finds the number of predators produced per prey item caught, for big prey and small prey
    @inputs:
    x - pred group size, could be vector
    b1 - number preds produced from big prey if completely eaten
    b2 - number preds produced from small prey if completely eaten
    limited_portions- whether there's a limit of how much food one predator can eat
    params - dic of parameters used by rest of model
    
    @returns
    conversion_big, conversion_small (both floats)

    @examples
    >> conversion_prey(x=2,b1=1,b2=0.1,limited_portions=True,**dict(b0 = 0.1))
    (0.1, 0.05)
    >> conversion_prey(x=[1,2],b1=1,b2=0.1,limited_portions=True,**dict(b0 = 0.1))
    (array([0.1, 0.1]), array([0.1 , 0.05]))
    >> conversion_prey(x=2,b1=1,b2=0.1,limited_portions=False,**dict(b0 = 0.1))
    (0.5, 0.05)
    '''
    if limited_portions == True:
        b0 = params["b0"]
        try:
            conversion_big = b0 if b1/x > b0 else b1/x
            conversion_small = b0 if b2/x > b0 else b2/x 
        except (ValueError,TypeError): # if x is a list (-->TypeError) or an ndarray (-->ValueError)
            x = np.array(x)
            conversion_big = np.ones(np.shape(x))
            conversion_small = conversion_big.copy()
            conversion_big = b1/x
            conversion_big[conversion_big>b0] = b0
            conversion_small = b2/x
            conversion_small[conversion_small>b0] = b0
    else:
        conversion_big = b1/x
        conversion_small = b2/x
    return conversion_big, conversion_small
    
        
        
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
             r, γ, a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2, limited_portions
             (and must also have b0 if limited_portions = True)

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
             r, γ, a1, a2, h1, h2, α1_of_1, α2_of_1, s1, s2, limited_portions
             (and must also have b0 if limited_portions = True)

    @returns:
    <inclusive fitness from small prey>

    @example
    >> fun_fitness_from_small_prey(1, 10, 10, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, 
    >>                                    h1=0.5, h2=0.5, α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2,
                                          limited_portions = False))
    0.15833333333333333
    '''
    return fitness_from_prey(x, M1, M2, **params)[1]
