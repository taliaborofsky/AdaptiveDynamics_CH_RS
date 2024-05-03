import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from fun_response_funs import *
from fitness_funs import *


def group_formation_model(t, f_of_x_vec,p,M1,M2,params):
    '''
    the full system of balance equations for x = 2,3,...,x_max
    @inputs:
    t - time, necessary for running with solve_ivp
    f_of_x_vec - a vector of the number of groups of size 2, 3, ..., x_max (maximum group size)
    p - population size of predators
    M1 - population size of big prey
    M2 - population size of small prey
    params - is a dictionary of the parameters that must contain: x_max (max group size), 
            τx (time scale of group size evolution), δ (pred death rate), 
            d (steepness of best response fun)
    '''
    x_max = params["x_max"]
    dfdt = [fun_dfdt(f_of_x_vec, x, p, M1, M2, **params) for x in range(2,x_max+1)]
    #for x in range(2,x_max+1):
    #    dfdt[x-2] = fun_dfdt(f_of_x_vec, x, p, M1, M2, **params)
    return dfdt


def fun_dfdt(f_of_x_vec, x, p, M1, M2,  τx, δ, x_max, **params):
    '''
    fun_dfdt: This calculates the change in the distribution f(x) wrt time for x >= 2
        τx df/dt = -xf(x)ϕ(x) - f(1) f(x) ψ(x) - f(x) D(x) 
                + f(x+1)ϕ(x+1) + sum_{y=x+1}^{x_max} f(y) D(y)
        but slightly different for x = 2, x = x_max
    f(x) is the number of groups of size x
    @inputs
    f_of_x_vec - vector of f(x) for x = 2, .., x_max.
    x - grp size, must be >= 2
    p - pred pop size
    M1 - big prey pop size
    M2 - small prey pop size
    τx - group size change time scale
    δ - pred death rate
    x_max - max grp size
    params - dictionary of params used by rest of model

    @returns
    float dfdt

    @example
    >> fun_dfdt(f_of_x_vec = [0,0], x=2, p=100, M1=0, M2=100, **dict(τx= 0.01, δ=0, x_max=3, 
    >>                                                          b1=1,b2=0.1,r=0, γ=0, a1=1, 
    >>                                                          a2=1, h1=0.5, h2=0.5, 
    >>                                              α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2, d = 100))
    6.209899910586194e-26
    
    '''
    # get f(x), f(1), and f(x+1)
    
    def f(y):
        #if y == 1: # recursive
        #    return p - sum([z*f(z) for z in range(2,x_max+1)]) # this is recursively designed
        #if y >= 2 and y <= x_max:
        #    return f_of_x_vec[y-2]
        #else:
        #    return 0
        return(fun_f_of_x(y, f_of_x_vec, p, x_max, **params))

    def D_tot(y):
        '''
        probability there is at least one death in group of size y
        '''
        return fun_1_death(y, τx, δ, **params)
    def D(z1,z2):
        # probability group of size z2 shrinks to group of size z1 for z1 \leq x_max  
        return fun_death_y_to_x(z1,z2, τx, δ, x_max, **params)
    def ϕ(y):
        # probability individual leaves group of size y for y <= x_max
        return fun_leave_group(y, M1, M2, x_max, **params) 
    def ψ(y):
        return fun_join_group(y, M1, M2, x_max, **params)
    


    # if x = 2, τ_x df_dx, group formation is different. it is 1/2 f(1)^2 ψ(1) instead of f(x-1)f(1) ψ(x-1)
    # if x = x_max, the group cannot grow larger and there are no larger groups that can shrink to be that size
    individual_leaves = x*f(x) * ϕ(x)
    grows_to_larger_group = f(1)*f(x) * ψ(x) if x<x_max else 0
    death_in_group = f(x) * D_tot(x)
    
    if x == 2:
        #join_smaller_grp = f(1)*f(x-1)*ψ(x-1)
        join_smaller_grp = (1/2)*f(1)*(f(1)-1)*ψ(1)
    else:
        join_smaller_grp = f(1)*f(x-1)*ψ(x-1)
            
    #join_smaller_grp = f(1)*f(x-1)*ψ(x-1)
    larger_grp_shrinks = (x+1)*f(x+1)*ϕ(x+1) if x < x_max else 0
    death_in_larger_grp = sum([f(y)*D(x,y) for y in range(x+1,x_max+1)]) if x < x_max else 0
        
    #dfdt_times_taux = x*f_of_x*fun_leave_group(x) - 
    
    
    return 1/τx * (-individual_leaves - grows_to_larger_group - death_in_group  
                  + join_smaller_grp + larger_grp_shrinks + death_in_larger_grp)


def fun_f_of_x(x, f_of_x_vec, p, x_max,**params):
    '''
    f(x)...find number of groups of size x
    @inputs
    x - grp size
    f_of_x_vec - vector of [f(2), f(3), ..., f(x_max)]
    p - pred pop size
    x_max - param, max grp size
    params - dictionary of params used by the rest of the model
    @ returns
    integer

    @examples
    >>fun_f_of_x(1,[1],p=3,x_max=2,**dict())
    1
    >>fun_f_of_x(1,[0,1],p=3,x_max=3,**dict())
    0
    
    '''
    if x == 1:
            return p - sum([z*f_of_x_vec[z-2] \
                            for z in range(2,x_max+1)]) 
    if x >= 2 and x <= x_max:
        return f_of_x_vec[x-2]
    else:
        return 0
    

def fun_leave_group(x, M1, M2, x_max, **params):
    '''
    The probability an individual leaves a group of size x.
    This is ϕ(x) in the text
    @inputs
    x - current grp size (before leaving)
    M1 - big prey pop size
    M2 - small prey pop size
    x_max - parameter, maximum group size
    params - dictionary of params used by the rest of the model
    '''
    # deciding between being alone and staying in group of size x
    return best_response_fun(1,x,M1,M2,**params)


def fun_join_group(x, M1, M2, x_max, **params):
    '''
    The probability an individual joins a group of size x.
    This is ψ(x) in the text
    @inputs
    x - current grp size (before joining)
    M1 - big prey pop size
    M2 - small prey pop size
    x_max - parameter, maximum group size
    params - dictionary of params used by the rest of the model
    '''
    # deciding between switching from being alone to being in a group of size x + 1
    return best_response_fun(x+1,1,M1,M2,**params)

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


def fun_death_y_to_x(x, y, τx, δ, x_max, **params):
    '''
    The probability a group of size y shrinks to a group of size x because y - x individuals die, 
    works for for x < y, y <= x_max

    @inputs:
    x = group size after death
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
    
    return nchoosek(y,y-x) * (δ*τx)**(y-x)*(1-δ*τx)**x

def fun_1_death(x, τx, δ, **params):
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
    >> fun_1-death(x=1, **dict(τx = 0.01, δ = 0.1))
    0.0010000000000000009
    >> fun_1_death(3, 0.01, 0.1, **dict())
    0.002997000999999999
    
    '''
    return 1 - (1 - δ*τx)**x


    
    
