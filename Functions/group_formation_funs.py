import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from fun_response_funs import *
from fitness_funs import *

            

def group_formation_model_separate(t, f_of_x_vec,p,M1,M2, params):
    '''
    the full system of balance equations for x = 1,2,3,...,x_max
    @inputs:
    t - time, necessary for running with solve_ivp
    f_of_x_vec - a vector of the number of groups of size 1, 2, 3, ..., x_max (maximum group size)
    p - population size of predators
    M1 - population size of big prey
    M2 - population size of small prey
    params - is a dictionary of the parameters that must contain: 
            b1, b2,r, γ, a1, a2, h1, h2, α1_of_15, α2_of_1, s1, s2, limited_portions, 
            τx, δ, d
            (and b0 if limited_portions = False)
    @ returns
    df_dt for x = 1, 2, ..., xmax
    '''
    x_max = params['x_max']; τx = params['τx']; δ = params['δ']; d = params['d']
    f_of_x_vec = np.append(f_of_x_vec,0) # so can find dfdt at x = x_max
    def f(x):
        return f_of_x_vec[x-1]
    def S(x,y):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def ψ(x):
        f_of_1 = f_of_x_vec[0]
        if x== 1 and f_of_1 >=1:
            return (f_of_1 - 1) * S(2,1)
        elif x <= x_max - 1:
            return f_of_1*S(x+1,1)
        else:
            return 0
    def ϕ(x):
        return x*S(1,x) if x <= x_max else 0
    
    x_max = params['x_max']; τx = params['τx']; δ = params['δ']; d = params['d']
    xvec = np.arange(1,x_max+1,1)
    # it \tau_x > 0make population matrix = birth matrix + death matrix
    fitnessvec = fun_fitness(xvec, M1, M2, **params)
    dfdt_vec = np.zeros(x_max)
    
    for x in xvec:
        if x == 1:
            dfdt = (2*f(2)*ϕ(2) + np.sum([f(y) * ϕ(y) for y in range(3,x_max+1)]) \
                    - sum([f(y-1)*ψ(y-1) for y in range(2,x_max+1)]))/τx
        elif x == 2:
            dfdt = (-f(2)*ϕ(2) - f(2)*ψ(2) + 0.5*f(1)*ψ(1) + f(3)*ϕ(3))/τx
        else:
            dfdt = (-f(x)*ϕ(x) - f(x) * ψ(x) + f(x-1)*ψ(x-1) + f(x+1)*ϕ(x+1))/τx
        
        dfdt_vec[x-1] = dfdt
    return dfdt_vec
    

def fun_1_death(x, τx, δ):
    '''
    The probability of AT LEAST one death in a group of size x over time τ_x
    @inputs:
    x - grp size
    τx - group evolution time constant
    δ - death rate
    params - dictionary of parameters from the ret of the model, not really needed...
    @output:
    float

    @example
    >> fun_1-death(x=1, τx = 0.01, δ = 0.1)
    0.0010000000000000009
    >> funfdgf_1_death(np.array([1,2,3]), 0.01, 0.1)
    array([0.001   , 0.001999, 0.002997])sxz
    
    '''
    return 1 - (1 - δ*τx)**x

def fun_death_y_to_x(x, y, τx, δ, x_max):
    '''
    The probability a group of size y shrinks to a group of size x because y - x individuals die, 
    works for for x < y, y <= x_max

    @inputs:
    x = group size after death, is the shape of what is being returned
    y = original group size, y > x, y <= x_max
    δ = death rate
    x_max = maximum group size
    params = dictionary of other parameters used in the model

    @output:
    float between 0 and 1 (note for τx small, fun_death_y_to_x(x,y,**params) \approx 0 if x < y-1

    @example
    >>fun_death_y_to_x(x=2, y=3, **dict(τx=0.01, δ=0.1, x_max=10))
    0.0029940030000000003
    '''
    if isinstance(y, np.ndarray):
        to_return = np.zeros(y.shape)
        notzero = x<y
        y = y[notzero]
        if isinstance(x, np.ndarray):
            x = x[notzero]
        to_return[notzero] = nchoosek(y,y-x) * (δ*τx)**(y-x)*(1-δ*τx)**x
        return to_return
    else:
        if x < y:
            return nchoosek(y,y-x) * (δ*τx)**(y-x)*(1-δ*τx)**x
            
def nchoosek(n,k):
    '''
    n choose k
    n!/(k!(n-k)!)
    @inputs:
    n and k are integers, but can handle np.arrays
    @returns:
    positive integer (or array if inputs are arrays
    @example
    >> nchoosek(3,1)
    3.0
    >> nchoosek(np.array([3,2]),1)
    array([3.,2.])
    '''
    return sp.special.factorial(n)/(sp.special.factorial(k)*sp.special.factorial(n-k))

def fun_leave_group(x, fitnessvec, x_max, d):
    '''
    The probability an individual leaves a group of size x.
    This is ϕ(x) in the text
    @inputs
    x - current grp size (before leaving)
    fitnessvec = vector of fitnesses for each group size
    x_max - parameter, maximum group size
    d = parameter determining steepness of best response function

    @ example:
    >> fitnessvec = array([0.24166667, 0.45833333, 0.53055556])
    >> fun_leave_group(xvec=[1,2,3], fitnessvec, x_max=3, d=100)
    array([0.5       , 0.03915869, 0.01923075])
    '''
    # deciding between being alone and staying in group of size x
    return best_response_fun_given_fitness(1,x,fitnessvec,d)

def best_response_fun_given_fitness(x,y,fitnessvec, d):
    '''
    Compares W(x) to W(y) to "decide" on group size y or x
    @inputs
    x - potentially new group size
    y - current grp size
    fitnessvec - vector of fitnesses fro x = 1, 2, ..., xmax
    d - steepness, or sensitivity, of best response function
    params - dictionary of params used by the rest of the model 
    @returns:
    float between 0 and 1
    
    '''
    W_of_x = fitnessvec[x-1]
    W_of_y = fitnessvec[y-1]
    return W_of_x**d/(W_of_x**d + W_of_y**d)
    
def best_response_fun(x,y,M1,M2, d, **params):
    '''
    Compares W(x) to W(y) to "decide" on group size y or x
    @inputs
    x - potentially new group size
    y - current grp size
    M1 - big prey pop size
    M2 - small prey pop size
    d - steepness, or sensitivity, of best response function
    params - dictionary of params used by the rest of the model, 
            but must include all params relevant to functional responses and 
            inclusive fitness 
    @returns:
    float between 0 and 1

    @example
    >> best_response_fun(x=2,y=3,M1=10,M2=10, d=100, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
    >>                                              α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2) )
    4.4162891945392386e-07
    >> 1 - best_response_fun(x=3,y=2,M1=10,M2=10, d=100, **dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
    >>                                              α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2) )
    4.4162891943422267e-07
    
    '''
    W_of_x = fun_fitness(x,M1,M2, **params)
    W_of_y = fun_fitness(y, M1, M2, **params)
    return W_of_x**d/(W_of_x**d + W_of_y**d)
