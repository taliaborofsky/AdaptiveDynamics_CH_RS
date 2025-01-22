import numpy as np
import scipy as sp
from fitness_funs_non_dim import *
from scipy.integrate import solve_ivp

def group_formation_model_non_dim(T, g_of_x_vec,N1,N2, params):
    '''
    the full system of balance equations for x = 1,2,3,...,x_max, non-dimensionalized
    @inputs:
    T - scaled time, necessary for running with solve_ivp
    g_of_x_vec - a vector of the (scaled) number of groups of 
            size 1, 2, 3, ..., x_max (maximum group size)
    P, N1, N2 - scaled population size of predators, big prey, small prey, respectivel
    if_groups_change = Bool, True --> preds can join/leave groups. 
                            False --> only birth/death affect group sizes
    params - is a dictionary of the parameters that must contain: 
            β1, β2, A, H1, H2, η1, η2, α1_of_1, α2_of_1, s1, s2, limited_portions, 
            Tx, d, r, γ, pop_process
    @ returns
    df_dT for x = 1, 2, ..., xmax
    '''
    x_max = params['x_max']; Tx = params['Tx']; 
    d = params['d']; 
    g_of_x_vec = np.append(g_of_x_vec,0) # so can find dgdT at x = x_max

    # fix negative values (solve_ivp can overshoot if step sizes too big)
    g_of_x_vec[g_of_x_vec<0] = 0
    
    def g(x):
        return g_of_x_vec[x-1]
    def S(x,y):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def J(x):
        # rate of joining
        g_of_1 = g_of_x_vec[0]
        if x== 1 and g_of_1 >=1:
            return g_of_1 * S(2,1)#( ξ *g_of_1 - 1) * S(2,1)
        elif x == 1 and g_of_1 < 1:
            return g_of_1 * S(2,1)
        elif x <= x_max - 1:
            return g_of_1*S(x+1,1)
        else:
            return 0
    def fun_deaths(x):
        return tildeδ * Tx * ( - x * g(x) + (x+1) * g(x+1) )
    def D(x):
        # rate of leaving/dispersing
        return x*S(1,x) if x <= x_max else 0
    
    xvec = np.arange(1,x_max+1,1)
    # it \tau_x > 0make population matrix = birth matrix + death matrix
    fitnessvec = per_capita_fitness_from_prey_non_dim(xvec, N1, N2, **params) # fitness_from_prey_non_dim(xvec, N1, N2, **params)
    dgdT_vec = np.zeros(x_max)

    # births and deaths
    if params['pop_process']:
        η1 = params['η1']; η2 = params['η2']; tildeδ = 1 - η1 - η2
        π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params) #would fitnessvec*xvec also work...?
        births_vec = Tx*g_of_x_vec[:-1]* π_vec
        births_vec = np.append(births_vec,0) # so can calculate births at x_max
        deaths_vec = [fun_deaths(x) for x in range(1,x_max+1)]
    else:
        π_vec = np.zeros(fitnessvec.shape)
        births_vec = π_vec.copy()
        births_vec = np.append(births_vec,0) # so can calculate births at x_max
        deaths_vec = π_vec.copy()

    # balance equations
    for x in xvec:
        if x == 1:
            Q_1 = 2*g(2)*D(2) + np.sum([g(y) * D(y) for y in range(3,x_max+1)]) \
                    - sum([g(y-1)*J(y-1) for y in range(2,x_max+1)])
            births1 = births_vec[x_max-1] - births_vec[0]
            dgdT = (Q_1 + births1 + deaths_vec[0])/Tx
        elif x == 2:
            Q_2 = -g(2)*D(2) - g(2)*J(2) + 0.5*g(1)*J(1) + g(3)*D(3)
            births2 = births_vec[0] - births_vec[1]
            dgdT = (Q_2 + births2 + deaths_vec[1])/Tx
        else:
            Q_x = -g(x)*D(x) - g(x) * J(x) + g(x-1)*J(x-1) + g(x+1)*D(x+1)
            
            birthsx = births_vec[x-2] - births_vec[x-1] if x < x_max else births_vec[x-2]
            dgdT = (Q_x + birthsx + deaths_vec[x-1])/Tx
        
        dgdT_vec[x-1] = dgdT


    return dgdT_vec
    

def fun_leave_group(x, fitnessvec, x_max, d):
    '''
    The probability an individual leaves a group of size x. D(x) in the text
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
        return 0.5
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
        return 0.5
    denominator = 1 + (W_min/W_max)**d

    return W_of_x**d/(W_of_x**d + W_of_y**d)

    

def model_one_x(T, initialstate, x, params):
    '''
    Find the time derivatives of P, N1, N2, with x fixed
    initialstate = P, N1, N2

    Here I need non-dimed F
    '''
    initialstate = np.array(initialstate)
    initialstate[np.abs(initialstate)<1e-11] = 0
    P, N1, N2 = initialstate
    g_of_x_vec = np.zeros(params['x_max'])
    g_of_x = P/x
    g_of_x_vec[x-1] = g_of_x
    dPdT = fun_dPdT_non_dim(P, N1, N2, g_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(N1, N2, g_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(N1, N2, g_of_x_vec, **params)
    return [dPdT, dN1dT, dN2dT]






def full_model(T, initialstate, arg, params):
    '''
    removed P!
    gets the time derivatives for N1, N2, g(1), g(2), ..., g(xm)
    @inputs
    T is just used by fsolve, not needed
    intiialstate = [N1,N2,*g_of_x]
    arg is a dummy because fsolve gets weird if there is only 1 arg?
    params is dictionary of params
    @ returns [dN1dT, dN2dT, *dgdT_vec]
    '''
    # i put arg there as a place holder because somehow makes ivp_solver work
    
    initialstate = np.array(initialstate)

    # this helps for numpy issues
    # initialstate[np.abs(initialstate)<1e-11] = 0

    #solve_ivp can overshoot, so anywhere that's negative really should be 0
    # initialstate[initialstate<0] = 0
    
    N1,N2 = initialstate[0:2]
    g_of_x_vec = initialstate[2:]
    #dPdT = fun_dPdT_non_dim(P, N1, N2, g_of_x_vec, **params)
    dN1dT = fun_dN1dT_non_dim(N1, N2, g_of_x_vec, **params)
    dN2dT = fun_dN2dT_non_dim(N1, N2, g_of_x_vec, **params)
    dgdT_vec = group_formation_model_non_dim(T, g_of_x_vec,N1,N2, params)
    # if if_groups_change:
    #     dgdT_vec = group_formation_model_non_dim(T, g_of_x_vec,N1,N2, 
    #                                              if_groups_change, params)
    # else:
    #     x = np.argwhere(g_of_x_vec>0)[0][0] + 1
    #     dgdT_vec = np.zeros(params['x_max'])
    #     dgdT_vec[x-1] = dPdT/x
    

    return [dN1dT, dN2dT, *dgdT_vec]

    

def fun_dPdT_non_dim(P, N1, N2, g_of_x_vec, η1, η2, β1, β2, **params):
    '''
    the equation for dPdT, the change in predator population size versus time, 
    non-dimensionalized. 

    @inputs
    P, N1, N2 - nondimensionalized predator, big prey, and small prey pop sizes
    g_of_x_vec - array of g(1), g(2), ... , g(x_max)
    params - dic of params that must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η1, η2 - scaled growth rates of big prey, small prey
    β1, β2 - scaled profitability of hunting big prey, small prey
    '''
    x_vec = np.arange(1,params['x_max']+1,1)
    tildef1_of_x = fun_f1(x_vec,N1,N2,**params)
    tildef2_of_x = fun_f2(x_vec,N1,N2,**params)
    tildeδ = 1 - η1 - η2
    total_fitness_per_x = β1 * tildef1_of_x + β2 * tildef2_of_x
    return np.sum(g_of_x_vec * total_fitness_per_x) - tildeδ*P

def fun_dN1dT_non_dim(N1, N2, g_of_x_vec, η1, **params):
    '''
    dN1dT, the change in big prey pop size versus time, non-dim'ed
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    g_of_x_vec - array of g(1), g(2), ... , g(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η1 - scaled growth rate of big prey
    '''
    if N1 > 0:
        x_vec = np.arange(1,params['x_max']+1,1)

        tildef1_of_x = fun_f1(x_vec,N1,N2,**params)
        return η1*N1*(1-N1) - np.sum(g_of_x_vec * tildef1_of_x)
    else:
        return 0

def fun_dN2dT_non_dim(N1, N2, g_of_x_vec, η2, **params):
    '''
    dN2dT, the change in small prey pop size versus time, non-dim'ed
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    g_of_x_vec - array of g(1), g(2), ... , g(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    η2 - scaled growth rate of small prey
    '''
    if N2 > 0:
        x_vec = np.arange(1,params['x_max']+1,1)

        f2_of_x = fun_f2(x_vec,N1,N2,**params)
    
        return η2*N2*(1-N2) -  np.sum(g_of_x_vec * f2_of_x)
    else:
        return 0



def mean_group_size_membership(g_of_x_vec, x_max, p):
    '''
    average group size any individual is in
    
    # columns of g_of_x_vec should be = x_max
    this is not the same as the average group size
    '''
    x_vec = np.arange(1,x_max+1,1)
    numerator = x_vec*(g_of_x_vec*x_vec)
    if isinstance(p, np.ndarray):
        mask = (p > 1e-10) & (np.all(g_of_x_vec.T>0, axis = 0))
        numerator = numerator.sum(1)
        ans = p.copy()
        ans[mask] = numerator[mask]/p[mask]
        ans[~mask] = np.ones(ans[~mask].shape)
        return ans
        
    else:
        if p < 1e-10 and np.all(np.array(g_of_x_vec)>0):
            return 1
        else:
            ans_to_sum =numerator/p
            return sum(ans_to_sum)

 
def bounded_ivp(y0,params, t_f = 1000, if_dict = False):
    '''
    NOTE: does not work for y0 = 0!!!!
    
    runs the ivp with a transformation of 
    (for y representing all the state variables)
    y --> u: y = a * exp (b * u). then  du/dt = dy/dt/(b*y)
    i choose a = b = 1 because it's easy
    ''' 
    y0 = np.array(y0)
    if np.any(y0) == 0:
        print("Bounded ivp does not work here\
        because the input contains a zero")
        return(0)
    
    a = 1; b = 1
    u0 = (1/b)*np.log(y0/a)
    out = solve_ivp(transformed_model, [0, t_f], y0 = u0,
                   method = "LSODA", args = (True, params)
                  )
    u_trajectory = out.y
    T = out.t
    y_trajectory = a*np.exp(b*u_trajectory)

    # extract results
    
    x_max = params['x_max']
    N1, N2 = y_trajectory[0:2]
    g_of_x_vec = y_trajectory[2:]
    xvec = np.arange(1,x_max+1,1)
    p = np.sum(xvec*g_of_x_vec.T,1)
    mean_x = mean_group_size_membership(g_of_x_vec.T, x_max, p)
    if if_dict:
        return dict(
            T=T, N1 = N1, N2 = N2, p = p, 
            g = g_of_x_vec, mean_x = mean_x
        )
    else:
        return T, N1, N2, p, g_of_x_vec, mean_x


    
def transformed_model(T, u0, arg, params):
    #For issue where need solve_ivp to keep some variable y s.t y > 0.
    # Soln from stack exchange: 
    # Alternatively, replace the density with y=a*exp(b*u) with sensibly chosen values for a and b, 
    # then du/dt = dy/dt/(b*y). 
    # Then g cannot become negative
    a = 1; b = 1
    y0 = a * np.exp(b*u0)
    y_ = full_model(T,y0, arg, params)
    y_ = np.array(y_)
    u_ = y_/(b*y0)
    return u_
        
