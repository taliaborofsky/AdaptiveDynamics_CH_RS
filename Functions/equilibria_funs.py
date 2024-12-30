import numpy as np
import matplotlib.pyplot as plt
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.integrate import solve_ivp
from scipy.optimize import root
from local_stability_funs import *

def find_mangel_clark(N1, N2, x_max, **params):
    # mangel and clark predicted that groups should grow until W(x^*) = W(1)
    # don't need A

    # simplest way... iterate and stop when reach x s.t. W(x) < W(1), then return x - 1
    W_of_1 = per_capita_fitness_from_prey_non_dim(1, N1, N2, **params)
    for x in range(2,x_max+1):
        W_of_x = per_capita_fitness_from_prey_non_dim(x, N1, N2, **params)
        if W_of_x < W_of_1:
            return x - 1
    return x # if reach x_max

    
    
def iterate_and_solve_equilibrium(params, t_f = 1000, tol = 1e-8):
    '''
    iterates from p = 3, N1 = 0.8, N2 = 0.7, 
    predators split evenly between groups of 1, 2, or 3
    then uses root to find equilibrium

    @returns
    P,N1,N2,g,mean_x at equilibrium, 
    and success (Boolean; true if the equilibria values are all nonnegative)
    '''
    x_max = params['x_max']
    x0 = [0.8, 0.7, *initiate_f_first_x(20, 2, x_max)]
    out2 = solve_ivp(full_model, [0, t_f], x0, method="LSODA",
                args=(True,params))
    T, N1, N2, p, g_of_x_vec, mean_x = get_results(out2, x_max)

    out = get_equilibrium(params, N1_0 = N1[-1], N2_0 = N2[-1], 
                          g_of_x_vec = g_of_x_vec[:,-1])
    P_eq, N1_eq, N2_eq, g_eq, mean_x_eq, success =get_results_eq(out,x_max)

    # to be successful, sum x*g = P
    # sum_x_g = np.sum(np.arange(1,x_max+1,1)*g_eq)
    # success = success and (np.abs(sum_x_g - P_eq )< tol)
    
    return P_eq, N1_eq, N2_eq, g_eq, mean_x_eq, success

def iterate_to_eq(initialstate, t_f, params):
    '''
    try to iterate to eq in t_f time steps
    '''
    out2 = solve_ivp(full_model, [0, t_f], initialstate, method="LSODA",
                args=(True,params))

    # extract results
    T,N1,N2,P,gxvec, mean_x = get_results(out2, params['x_max'])
    full_trajectory = [T, N1, N2, P, gxvec]
    # get values at potential equilibrium
    
    N1,N2,P,mean_x = [ item[-1] for item in [N1,N2,P,mean_x]]
    g = gxvec[:,-1]
    
    timederivatives = full_model(T[-1], [N1,N2,*g],True,params)
    
    success = np.all(np.abs(np.array(timederivatives)) < 1e-9)
    
    
    return np.array([P, N1, N2, *g]), success, mean_x, timederivatives, full_trajectory
    
def get_equilibria_vary_param(paramvec, paramkey, **params):
    '''
    Get a list of equilibrium values corresponding to the parameters
    '''


    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)

    # set up empty vectors
    meanxvec = np.zeros(len(paramvec))
    gxvecs  = np.zeros((len(paramvec), x_max))
    Pvec = meanxvec.copy()
    N1vec = meanxvec.copy()
    N2vec = meanxvec.copy()
    success_vec = meanxvec.copy()
    stability_vec = meanxvec.copy()
    
    for i, param in enumerate(paramvec):
        params = params.copy()
        params[paramkey] = param

        # try to iterate a little and then use root to solve for equilibrium
        out_eq = iterate_and_solve_equilibrium(params, t_f = 5)
        P, N1, N2, g, mean_x, success = out_eq
        
        if success==False:
            
            # try to get to equilibrium in just 200 steps #
            
            t_f = 500
            initialstate = [0.5,0.4, 20, *np.zeros(x_max-1)]
            finalpoint, success, mean_x, _, _ = iterate_to_eq(initialstate, t_f,
                                                                         params)
            [P,N1,N2,*g] = finalpoint

            # if that doesn't work, try solving from here
            if success == False:
                out = get_equilibrium(params, N1_0 = N1, N2_0 = N2, 
                          g_of_x_vec = g)
                P, N1, N2, g, mean_x, success =get_results_eq(out,x_max)
            # if that doesn't work, now do another 2000 steps
            if success == False:
                out = iterate_to_eq(finalpoint[1:], 5000,params)   
                finalpoint, success, mean_x, _, _ = out
            
                [P,N1,N2,*g] = finalpoint
            if success == False:
                out = get_equilibrium(params, N1_0 = N1, N2_0 = N2, 
                          g_of_x_vec = g)
                P, N1, N2, g, mean_x, success =get_results_eq(out,x_max)
            
        success_vec[i] = success
        
        gxvecs[i,:] = g
        Pvec[i] = P
        N1vec[i] = N1
        N2vec[i] = N2
        meanxvec[i] = mean_x


        # check stability
        try:
            if np.any(np.isnan(np.array([P,N1,N2,*g]))):
                stability_vec[i] = np.nan
        except TypeError:
            stability_vec[i] = np.nan
        else:
            J = fun_Jac(N1,N2,np.array(g),**params)
            stability = classify_stability(J)
            if stability == "Stable (attractive)":
                stability_vec[i] = 1
            elif stability == "Unstable":
                stability_vec[i] = -1
            else:
                stability_vec[i] = 0
        
    return Pvec, N1vec, N2vec, gxvecs,meanxvec,success_vec, stability_vec
    
def get_equilibrium(params, N1_0 = 0.5, N2_0 = 0.4, p_0 = 20, g_of_x_vec = None):
    '''
    finds the equilibrium using Fsolve for the population dynamics and group dynamics system
    if not given g_of_x_vec, then just has everyone initially solitary
    
    @returns:
    N1_eq, N2_eq, F_eq, P_eq, mean_x_eq
    '''
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    if not isinstance(g_of_x_vec, np.ndarray):
        #print('hi')
        x_f = 2 if x_max > 2 else x_max
        g_of_x_vec = initiate_f_first_x(p_0, x_f, x_max)
        
    x0 = [N1_0, N2_0, *g_of_x_vec]
    out = root(fun = nullclines_no_P, x0 = x0, 
                                  args = (params))
    return out

def abs_nullclines_no_P(initialstate, params):
    return np.sum(np.abs(nullclines_no_P(initialstate, params)))
