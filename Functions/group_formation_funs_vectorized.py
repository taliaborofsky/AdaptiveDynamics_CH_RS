import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from fun_response_funs import *
from fitness_funs import *

    
def group_formation_model_alt(t, f_of_x_vec,p,M1,M2, pop_process, params):
    '''
    the full system of balance equations for x = 1,2,3,...,x_max
    @inputs:
    t - time, necessary for running with solve_ivp
    f_of_x_vec - a vector of the number of groups of size 1, 2, 3, ..., x_max (maximum group size)
    p - population size of predators
    M1 - population size of big prey
    M2 - population size of small prey
    pop_process = True or False, whether or not to include death rate and birth rate
    params - is a dictionary of the parameters that must contain: 
            b1, b2,r, γ, a1, a2, h1, h2, α1_of_15, α2_of_1, s1, s2, limited_portions, 
            τx, δ, d
            (and b0 if limited_portions = False)
    @ returns
    df_dt for x = 1, 2, ..., xmax
    '''
    x_max = params['x_max']; τx = params['τx']; δ = params['δ']; d = params['d']
    
    xvec = np.arange(1,x_max+1,1)

    # it \tau_x > 0make population matrix = birth matrix + death matrix
    fitnessvec = fun_fitness(xvec, M1, M2, **params)
    if pop_process==True:
        Π = make_population_proc_matrix(xvec, fitnessvec, δ, τx, x_max)
    else: 
        Π = 0
    Ψ = make_group_form_mat(xvec, f_of_x_vec, fitnessvec, x_max,d)
    Φ = make_group_leave_mat(xvec, fitnessvec, x_max, d)
    return np.matmul( Π + Ψ + Φ, np.transpose(f_of_x_vec))/τx

def make_group_leave_mat(xvec, fitnessvec, x_max, d):
    
    ϕ_of_x = best_response_fun_given_fitness(1,xvec,fitnessvec,d)*xvec
    ϕ_of_x[0] = 0 # can't leave a group of 1
    
    first_row_and_upper_diag = ϕ_of_x[1:].copy()
    first_row_and_upper_diag[0] = 2*first_row_and_upper_diag[0] # if group of size 2 --> 1, 
                                                                # produces 2 solitaries
    diag_mat = np.diag(- ϕ_of_x) # fewer groups of size x if grps of size x --> x - 1
    upper_diag_mat = np.diag(first_row_and_upper_diag,k=1) # more grps of size x if grps of size x +1 --> x
    group_shrink_mat = diag_mat + upper_diag_mat
    group_shrink_mat[0,1:] = first_row_and_upper_diag # more solitaries when individuals leave groups

    return group_shrink_mat
def make_group_form_mat(xvec, f_of_x_vec, fitnessvec, x_max,d):
    '''
    example:
    >>params_reg = dict(b1=1,b2=0.1,r=0, γ=0, a1=1, a2=1, h1=0.5, h2=0.5, 
                                                    α1_of_1=0.05, α2_of_1=0.95, s1=2, s2=2, d = 20,
                                                    limited_portions = False)
    >>f_of_x_vec=[1,1,1]
    >> make_group_form_mat(x_vec=np.array([1,2,3]), f_of_x_vec, M1=10, M2=10, x_max=3,**params_reg)
    array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        , -0.99999985,  0.        ],
       [ 0.        ,  0.99999985,  0.        ]])
    '''
    # alterations
    f_of_1_vec = np.full(x_max,f_of_x_vec[0])
    f_of_1_vec[0] = f_of_1_vec[0] - 1
    xvec = xvec[:-1] # get rid of xmax
    
    join_grp_vec = np.zeros(x_max)
    
    best_response_vec = best_response_fun_given_fitness(xvec+1,1,fitnessvec,d) #S(x+1,1)
    join_grp_vec[:-1] = best_response_vec * f_of_1_vec[:-1]

    # make matrix
    diag_mat = np.diag(- join_grp_vec) # loss from class x as x --> x+1
    lower_diag_mat = np.diag(join_grp_vec[:-1],k=-1) # gain to class x+1 as x --> x+1
    group_form_mat = diag_mat + lower_diag_mat
    group_form_mat[-1, -1] = 0 # can't grow once at x_max
    group_form_mat[1,0] = 0.5*group_form_mat[1,0] # individuals forming pairs, have to multiply by 1/2
    return group_form_mat
                         
def make_population_proc_matrix(xvec, fitnessvec, δ, τx, x_max):
    '''
    make the population processes matrix Π = τx Π_W  + Π_D
    @inputs
    xvec = [1,2,..., x_max]
    fitnessvec = [\bar{w}(1), \bar{w}(2), ..., \bar{x_max}(1)] vector of per capita fitnesses
    δ = death rate
    τx = group time scale
    x_max = max grp size

    @returns
    np.ndarray that is x_max x x_max

    @example:
    >> params_reg = dict(b1=1,b2=0.1,r=0,γ=0, a1 = 1, a2 = 1, h1 = 0.5, h2 = 0.5, 
                  α1_of_1 = 0.05, α2_of_1 = 0.95, s1 = 2, s2 = 2, d = 100, limited_portions = False)
    >> xvec = np.array([1,2,3]); M1 = 10; M2 = 10; x_max = 3
    >> fitnessvec = fun_fitness(xvec, M1, M2, **params_reg) # this is array([0.24166667, 0.45833333, 0.53055556])
    >> make_population_proc_matrix(xvec, fitnessvec, δ, τx, x_max)
    array([[ 0.00141667,  0.01116467,  0.01591966],
       [ 0.        , -0.001999  ,  0.002994  ],
       [ 0.        ,  0.        , -0.002997  ]])
    '''
    Π_D = make_death_trans_matrix(xvec, δ, τx, x_max)

    # birth matrix
    Π_W = np.zeros((x_max,x_max))
    Π_W[0,:] = fitnessvec*xvec

    # population matrix
    Π = τx * Π_W + Π_D
    return Π
    
def make_death_trans_matrix(xvec, δ, τx, x_max):
    '''
    makes the death transition matrix
    the diagonal row is D(x)
    above the diagonal, entries are D(i,j), for i the row and j the column

    @inputs:
    xvec - vector 1, 2, ..., x_max
    δ = death rate
    τx = timescale of group dynamics
    x_max = max group size

    @output: an x-max x x_max numpy array

    @example 
    >> make_death_trans_matrix(xvec = np.array([1,2,3]), δ=0.1, τx=0.01, x_max=3)
    array([[-1.000000e-03,  1.998000e-03,  2.997000e-06],
       [ 0.000000e+00, -1.999000e-03,  2.994003e-03],
       [ 0.000000e+00,  0.000000e+00, -2.997001e-03]])
    '''
    # number of groups of size x decreases from deaths in group of size x
    Pi_D = np.zeros((x_max,x_max))
    D_of_x = fun_1_death(xvec, τx, δ)
    np.fill_diagonal(Pi_D, -D_of_x)

    # number of groups of size x increases if a larger group of size y has y-x deaths
    y_mat = np.full((x_max,x_max),xvec)
    x_mat = y_mat.copy().transpose()
    D_transition = fun_death_y_to_x(x=x_mat, y=y_mat, τx=τx, δ=δ, x_max=x_max)
    Pi_D = Pi_D + D_transition
    return Pi_D
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
