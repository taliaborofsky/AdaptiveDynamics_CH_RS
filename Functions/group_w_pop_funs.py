import numpy as np
import scipy as sp
from fitness_funs_non_dim import *


def group_formation_model_non_dim(T, F_of_x_vec,P,N1,N2, params):
    '''
    the full system of balance equations for x = 1,2,3,...,x_max, non-dimensionalized
    @inputs:
    T - scaled time, necessary for running with solve_ivp
    F_of_x_vec - a vector of the (scaled) number of groups of 
            size 1, 2, 3, ..., x_max (maximum group size)
    P, N1, N2 - scaled population size of predators, big prey, small prey, respectivel
    if_groups_change = Bool, True --> preds can join/leave groups. 
                            False --> only birth/death affect group sizes
    params - is a dictionary of the parameters that must contain: 
            β1, β2, A1, H1, H2, η1, η2, α1_of_1, α2_of_1, s1, s2, limited_portions, 
            Tx, d, ξ, r, γ, pop_process
    @ returns
    dF_dT for x = 1, 2, ..., xmax
    '''
    x_max = params['x_max']; Tx = params['Tx']; 
    η1 = params['η1']; η2 = params['η2']; tildeδ = 1 - η1 - η2
    d = params['d']; ξ = params['ξ']
    F_of_x_vec = np.append(F_of_x_vec,0) # so can find dfdt at x = x_max
    
    def F(x):
        return F_of_x_vec[x-1]
    def S(x,y):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def ψ(x):
        F_of_1 = F_of_x_vec[0]
        if x== 1 and F_of_1 >=1:
            return ( ξ * F_of_1 - 1) * S(2,1)
        elif x <= x_max - 1:
            return ξ * F_of_1*S(x+1,1)
        else:
            return 0
    def fun_deaths(x):
        return tildeδ * Tx * ( - x*F(x) + (x+1) * F(x+1) )
    def ϕ(x):
        return x*S(1,x) if x <= x_max else 0
    
    xvec = np.arange(1,x_max+1,1)
    # it \tau_x > 0make population matrix = birth matrix + death matrix
    fitnessvec = fitness_from_prey_non_dim(xvec, N1, N2, **params)
    dFdT_vec = np.zeros(x_max)
    include_pop_process = 1 if params['pop_process'] == True else 0
    for x in xvec:
        if x == 1:
            Q_1 = 2*F(2)*ϕ(2) + np.sum([F(y) * ϕ(y) for y in range(3,x_max+1)]) \
                    - sum([F(y-1)*ψ(y-1) for y in range(2,x_max+1)])
            tilde_w_vec = per_capita_fitness_from_prey_non_dim(xvec, N1, N2, **params)
            births = Tx * np.sum(F_of_x_vec[:-1] * xvec * tilde_w_vec)
            dFdT = (Q_1 \
                    + include_pop_process*(births + fun_deaths(1)))/Tx
        elif x == 2:
            Q_2 = -F(2)*ϕ(2) - F(2)*ψ(2) + 0.5*F(1)*ψ(1) + F(3)*ϕ(3)
            dFdT = (Q_2 \
                    + include_pop_process*fun_deaths(2))/Tx
        else:
            Q_x = -F(x)*ϕ(x) - F(x) * ψ(x) + F(x-1)*ψ(x-1) + F(x+1)*ϕ(x+1)
            dFdT = (Q_x \
                    + include_pop_process*fun_deaths(x))/Tx
        
        dFdT_vec[x-1] = dFdT
    return dFdT_vec
    

def fun_leave_group(x, fitnessvec, x_max, d):
    '''
    The probability an individual leaves a group of size x. ϕ(x) in the text
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
    W_min = min(W_of_x, W_of_y)
    W_max = max(W_of_x, W_of_y)
    numerator = (W_of_x/W_max)**d
    denominator = 1 + (W_min/W_max)**d # this adjustment helps avoid dividing by zero from numpy rounding
    return numerator/denominator
    
    # if W_of_x**d + W_of_y**d < 1e-100: # note that then at this point it will be 
    #                                     #rounded to 0, 
    #                                     # but since W_of_y**d must be less than 1e-300, 
    #                                     # can approximate as 1 - (W(y)/W(x))**d
    #     # option 1: both W_of_x and W_of_y are really really small
    #     # option 2: both are really small, but one is much bigger than the other
    #     # scale the smaller 1:
    #     W_min = min(W_of_x, W_of_y)
    #     W_max = max(W_of_x, W_of_y)
    #     numerator = (W_of_x/W_max)**d
    #     denominator = 1 + (W_min/W_max)**d
    #     return numerator/denominator
    # else:
    #     return W_of_x**d/(W_of_x**d + W_of_y**d)
    
def best_response_fun(x,y, N1,N2, d, **params):
    '''
    Compares W(x) to W(y) to "decide" on group size y or x
    @inputs
    x - potentially new group size
    y - current grp size
    N1, N2 - big prey and small prey scaled pop size
    d - steepness, or sensitivity, of best response function
    params - dictionary of params used by the rest of the model
    @returns:
    float between 0 and 1
    
    '''
    W_of_x = fitness_from_prey_non_dim(x, N1, N2, **params)
    W_of_y = fitness_from_prey_non_dim(y, N1, N2, **params)
    return W_of_x**d/(W_of_x**d + W_of_y**d)

    
def check_at_equilibrium(final_distribution, P, N1, N2,pop_process,**params):
    '''
    check dF(x)/dT \approx 0
    @ returns: array dFdT_, and 1 if at equilibrium or 0 if not
    '''
    T = 1 # this doesn't matter
    dFdT_ = group_formation_model_non_dim(T, final_distribution,P,N1,N2, params)
    at_equilibrium = np.abs(dFdT_) > 1e-10
    if sum(at_equilibrium) > 0: # at least one df(x)/dt is not zero
        return dFdT_, 0
    else:
        return dFdT_, 1

def model_one_x(T, initialstate, x, params):
    initialstate = np.array(initialstate)
    initialstate[np.abs(initialstate)<1e-11] = 0
    P, N1, N2 = initialstate
    F_of_x_vec = np.zeros(params['x_max'])
    F_of_x = P/x
    F_of_x_vec[x-1] = F_of_x
    dPdT = fun_dPdT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    return [dPdT, dN1dT, dN2dT]


def model_one_x_evolve(T, initialstate, params):
    '''
    Model where all predators approximated as in groups of same size
    tracks the evolution of the number of helpers.....group members - 1 
            (so the number of subordinates, perhaps)
    initialstate = P, N1, N2, y (scaled pop size of preds, big prey, and small prey, and 
                    num. of helpers)

    @returns:
    [dPdT, dN1dT, dN2dT, dydT]
    '''
    initialstate = np.array(initialstate)
    initialstate[np.abs(initialstate)<1e-11] = 0
    P, N1, N2, y= initialstate
    # y is the number of subordinates. x, group size, is 1 + y
    F_of_x_vec = np.zeros(params['x_max'])
    F_of_x = P/(1+y)
    F_of_x_vec[y] = F_of_x
    dPdT = fun_dPdT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dydT = fun_dydT_non_dim(N1, N2, y, **params)
    return [dPdT, dN1dT, dN2dT, dydT]

def full_model(T, initialstate, arg, params):
    
    # i put arg there as a place holder because somehow makes ivp_solver work
    
    initialstate = np.array(initialstate)

    # this helps for numpy issues
    initialstate[np.abs(initialstate)<1e-11] = 0
    
    P,N1,N2 = initialstate[0:3]
    F_of_x_vec = initialstate[3:]
    dPdT = fun_dPdT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dFdT_vec = group_formation_model_non_dim(T, F_of_x_vec,P,N1,N2, params)
    # if if_groups_change:
    #     dFdT_vec = group_formation_model_non_dim(T, F_of_x_vec,P,N1,N2, 
    #                                              if_groups_change, params)
    # else:
    #     x = np.argwhere(F_of_x_vec>0)[0][0] + 1
    #     dFdT_vec = np.zeros(params['x_max'])
    #     dFdT_vec[x-1] = dPdT/x
    

    return [dPdT, dN1dT, dN2dT, *dFdT_vec]
def fun_dydT_non_dim(N1, N2, y Tx, **params):
    W_of_x = fitness_from_prey_non_dim(1+y, N1, N2, **params)
    W_of_1 = fitness_from_prey_non_dim(1, N1, N2, **params)

    return (W_of_x - W_of_1)/Tx
    
    
def fun_dPdT_non_dim(P, N1, N2, F_of_x_vec, η1, η2, β1, β2, **params):
    '''
    the equation for dPdT, the change in predator population size versus time, 
    non-dimensionalized. 

    @inputs
    P, N1, N2 - nondimensionalized predator, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params that must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η1, η2 - scaled growth rates of big prey, small prey
    β1, β2 - scaled profitability of hunting big prey, small prey
    '''
    x_vec = np.arange(1,params['x_max']+1,1)
    tildeY1_of_x = fun_response_non_dim(x_vec,N1,N2,1,**params)
    tildeY2_of_x = fun_response_non_dim(x_vec,N1,N2,2,**params)
    tildeδ = 1 - η1 - η2
    total_fitness_per_x = β1 * tildeY1_of_x + β2 * tildeY2_of_x
    return np.sum(F_of_x_vec * total_fitness_per_x) - tildeδ*P

def fun_dN1dT_non_dim(P, N1, N2, F_of_x_vec, η1, A1, **params):
    '''
    dN1dT, the change in big prey pop size versus time, non-dim'ed
    @inputs:
    P, N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η1 - scaled growth rate of big prey
    A1 - scaled attack rate of big prey
    '''
    x_vec = np.arange(1,params['x_max']+1,1)

    tildeY1_of_x = fun_response_non_dim(x_vec,N1,N2,1,**params)
    return η1*N1*(1-N1) - A1 * np.sum(F_of_x_vec * tildeY1_of_x)

def fun_dN2dT_non_dim(P, N1, N2, F_of_x_vec, η2, A1, **params):
    '''
    dN2dT, the change in small prey pop size versus time, non-dim'ed
    @inputs:
    P, N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η2 - scaled growth rate of small prey
    A1 - scaled attack rate of big prey
    '''
    A2 = 1 - A1
    x_vec = np.arange(1,params['x_max']+1,1)

    tildeY2_of_x = fun_response_non_dim(x_vec,N1,N2,2,**params)
    
    return η2*N2*(1-N2) - A2 * np.sum(F_of_x_vec * tildeY2_of_x)



def mean_group_size_membership(F_of_x_vec, x_max, P):
    '''
    average group size any individual is in
    
    # columns of F_of_x_vec should be = x_max
    this is not the same as the average group size
    '''
    x_vec = np.arange(1,x_max+1,1)
    frequency_in_group_size_x = (F_of_x_vec*x_vec).T/P
    vec_to_sum = x_vec*frequency_in_group_size_x.T
    return vec_to_sum.sum(1)
