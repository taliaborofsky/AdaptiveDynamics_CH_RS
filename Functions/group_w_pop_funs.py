import numpy as np
import scipy as sp
from fitness_funs_non_dim import *


def group_formation_model_non_dim(T, F_of_x_vec,N1,N2, params):
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
    include_pop_process = 1 if params['pop_process'] == True else 0

    def F(x):
        return F_of_x_vec[x-1]
    def S(x,y):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def ψ(x):
        # rate of joining
        F_of_1 = F_of_x_vec[0]
        if x== 1 and F_of_1 >=1:
            return ξ *F_of_1 * S(2,1)#( ξ *F_of_1 - 1) * S(2,1)
        elif x == 1 and F_of_1 < 1:
            return ξ * F_of_1 * S(2,1)
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

    # births
    π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params)
    births_vec = Tx*F_of_x_vec[:-1]* π_vec
    births_vec = np.append(births_vec,0) # so can calculate births at x_max

    # balance equations
    for x in xvec:
        if x == 1:
            Q_1 = 2*F(2)*ϕ(2) + np.sum([F(y) * ϕ(y) for y in range(3,x_max+1)]) \
                    - sum([F(y-1)*ψ(y-1) for y in range(2,x_max+1)])
            births1 = births_vec[x_max-1] - births_vec[0]
            dFdT = (Q_1 + include_pop_process*(births1 + fun_deaths(1)))/Tx
        elif x == 2:
            Q_2 = -F(2)*ϕ(2) - F(2)*ψ(2) + 0.5*F(1)*ψ(1) + F(3)*ϕ(3)
            births2 = births_vec[0] - births_vec[1]
            dFdT = (Q_2 + include_pop_process*(births2+fun_deaths(2)))/Tx
        else:
            Q_x = -F(x)*ϕ(x) - F(x) * ψ(x) + F(x-1)*ψ(x-1) + F(x+1)*ϕ(x+1)
            
            birthsx = births_vec[x-2] - births_vec[x-1] if x < x_max else births_vec[x-2]
            dFdT = (Q_x + include_pop_process*(birthsx+fun_deaths(x)))/Tx
        
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
    if W_max > 0:
        numerator = (W_of_x/W_max)**d
    else:
        return 0
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

    W_min = min(W_of_x, W_of_y)
    W_max = max(W_of_x, W_of_y)
    if W_max > 0:
        numerator = (W_of_x/W_max)**d
    else:
        return 0
    denominator = 1 + (W_min/W_max)**d

    return W_of_x**d/(W_of_x**d + W_of_y**d)

    
def check_at_equilibrium(final_distribution, P, N1, N2,**params):
    '''
    check dF(x)/dT \approx 0
    @ returns: array dFdT_, and 1 if at equilibrium or 0 if not
    '''
    T = 1 # this doesn't matter
    dFdT_ = group_formation_model_non_dim(T, final_distribution,N1,N2, params)
    not_at_equilibrium = np.abs(dFdT_) > 1e-8
    if sum(not_at_equilibrium) > 0: # at least one df(x)/dt is not zero
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
    dN1dT = fun_dN1dT_non_dim(N1, N2, F_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(N1, N2, F_of_x_vec, **params)
    return [dPdT, dN1dT, dN2dT]



def full_model(T, initialstate, arg, params):
    '''
    gets the time derivatives for P, N1, N2, F(1), F(2), ..., F(xm)
    @inputs
    T is just used by fsolve, not needed
    intiialstate = [P,N1,N2,*F_of_x]
    arg is a dummy because fsolve gets weird if there is only 1 arg?
    params is dictionary of params
    @ returns [dPdT, dN1dT, dN2dT, *dFdT_vec]
    '''
    # i put arg there as a place holder because somehow makes ivp_solver work
    
    initialstate = np.array(initialstate)

    # this helps for numpy issues
    initialstate[np.abs(initialstate)<1e-11] = 0
    
    P,N1,N2 = initialstate[0:3]
    F_of_x_vec = initialstate[3:]
    dPdT = fun_dPdT_non_dim(P, N1, N2, F_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(N1, N2, F_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(N1, N2, F_of_x_vec, **params)
    dFdT_vec = group_formation_model_non_dim(T, F_of_x_vec,N1,N2, params)
    # if if_groups_change:
    #     dFdT_vec = group_formation_model_non_dim(T, F_of_x_vec,N1,N2, 
    #                                              if_groups_change, params)
    # else:
    #     x = np.argwhere(F_of_x_vec>0)[0][0] + 1
    #     dFdT_vec = np.zeros(params['x_max'])
    #     dFdT_vec[x-1] = dPdT/x
    

    return [dPdT, dN1dT, dN2dT, *dFdT_vec]

def nullclines_no_P(initialstate, params):
    '''
    returns the nullclines for N1, N2, F(1), F(2), ..., F(x_max)
    such that N1, N2 \neq 0
    @inputs
    initialstate = [N1, N2, F(1), ..., F(x_max)], type ndarray
    params = dictionary of params
    '''
    N1 = initialstate[0]
    N2 = initialstate[1]
    F_of_x_vec = initialstate[2:]

    # get P
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    P = np.sum(xvec*F_of_x_vec)


    
    

    N1_null, N2_null = N_nullclines(N1, N2, F_of_x_vec, xvec, **params)
    dFdT_vec = group_formation_model_non_dim(0, F_of_x_vec,N1,N2, params) # I put 0 for T
    
    return [N1_null, N2_null, *dFdT_vec]

def N_nullclines(N1, N2, F_of_x_vec, xvec, η1, η2, A1, H1, H2, **params):
    '''
    dN1dT, dN2dT, the change in prey pop size versus time, non-dim'ed, divided by N_i
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    '''

    A2 = 1 - A1
    α1 = fun_alpha1(xvec,**params) 
    α2 = fun_alpha2(xvec,**params) 

    # prey nonzero nullclines
    Y1_no_N = α1/(1 + H1*α1*N1 + H2*α2*N2)
    Y2_no_N = α2/(1 + H1*α1*N1 + H2*α2*N2)

    N1_null = η1 * (1-N1) - A1 * np.sum(F_of_x_vec * Y1_no_N)
    N2_null = η2 * (1-N2) - A2 * np.sum(F_of_x_vec * Y2_no_N)
    
    return N1_null, N2_null
    
    
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
    Y1_of_x = fun_Y1(x_vec,N1,N2,**params)
    Y2_of_x = fun_Y2(x_vec,N1,N2,**params)
    tildeδ = 1 - η1 - η2
    total_fitness_per_x = β1 * Y1_of_x + β2 * Y2_of_x
    return np.sum(F_of_x_vec * total_fitness_per_x) - tildeδ*P

def fun_dN1dT_non_dim(N1, N2, F_of_x_vec, η1, A1, **params):
    '''
    dN1dT, the change in big prey pop size versus time, non-dim'ed
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η1 - scaled growth rate of big prey
    A1 - scaled attack rate of big prey
    '''
    x_vec = np.arange(1,params['x_max']+1,1)

    Y1_of_x = fun_Y1(x_vec,N1,N2,**params)
    return η1*N1*(1-N1) - A1 * np.sum(F_of_x_vec * Y1_of_x)

def fun_dN2dT_non_dim(N1, N2, F_of_x_vec, η2, A1, **params):
    '''
    dN2dT, the change in small prey pop size versus time, non-dim'ed
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    F_of_x_vec - array of F(1), F(2), ... , F(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η2 - scaled growth rate of small prey
    A1 - scaled attack rate of big prey
    '''
    A2 = 1 - A1
    x_vec = np.arange(1,params['x_max']+1,1)

    Y2_of_x = fun_Y2(x_vec,N1,N2,**params)
    
    return η2*N2*(1-N2) - A2 * np.sum(F_of_x_vec * Y2_of_x)



def mean_group_size_membership(F_of_x_vec, x_max, P):
    '''
    average group size any individual is in
    
    # columns of F_of_x_vec should be = x_max
    this is not the same as the average group size
    '''
    x_vec = np.arange(1,x_max+1,1)
    frequency_in_group_size_x = (F_of_x_vec*x_vec).T/P
    vec_to_sum = x_vec*frequency_in_group_size_x.T
    if len(vec_to_sum.shape) > 1:
        return vec_to_sum.sum(1)
    else:
        return sum(vec_to_sum)
