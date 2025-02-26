import numpy as np
import matplotlib.pyplot as plt
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.integrate import solve_ivp
from scipy.optimize import root
from local_stability_funs import *
from sim_graph_funs import *




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


def iterate_to_eq(initialstate, t_f, params, if_dict=False):
    '''
    Iterates the system to find an equilibrium within a specified time frame.

    Args:
        initialstate (array-like): Initial state of the system, including [N1, N2, g(1), ..., g(x_max)].
        t_f (float): Final time for the simulation.
        params (dict): Dictionary of system parameters.
        if_dict (bool, optional): If True, returns the equilibrium in dictionary format with additional information.
            Default is False.

    Returns:
        tuple:
            - If `if_dict=True`:
                eq_dict (dict): Dictionary containing:
                    - 'equilibrium' (np.ndarray): Final state after iterations, including [N1, N2, g(1), ..., g(x_max)].
                    - 'mean_x' (float): Mean experienced group size at equilibrium.
                    - 'var' (float): Variance of experienced group size at equilibrium.
                success (bool): Whether the system successfully reached equilibrium.
                timederivatives (np.ndarray): Time derivatives at the final state to verify equilibrium.
            - If `if_dict=False`:
                curr (list): Final state after iterations, including [N1, N2, g(1), ..., g(x_max), mean_x, var].
                success (bool): Whether the system successfully reached equilibrium.
                timederivatives (np.ndarray): Time derivatives at the final state to verify equilibrium.

    '''
    out2 = bounded_ivp(initialstate, params, t_f=t_f, if_dict=if_dict)
    
    if if_dict:
        T = out2['T']
        N1 = out2['N1']
        N2 = out2['N2']
        p = out2['p']
        g_of_x_vec = out2['g']
        mean_x = out2['mean_x']
        var = out2['var']
    else:
        T, N1, N2, p, g_of_x_vec, mean_x, var = out2
    
    # Extract results
    traj = [N1, N2, *g_of_x_vec, p]
    curr = [item[-1] for item in traj]
    [N1, N2, *g_eq, p] = curr

    # Check if at equilibrium
    success, timederivatives = check_at_equilibrium2(N1, N2, g_eq, params)

    # Handle invalid results
    if not np.isfinite(curr).all():
        success = False
    
    if if_dict:
        eq_dict = dict(equilibrium=np.array([N1,N2,*g_eq]), mean_x=mean_x[-1], var=var[-1], p = p)
        return eq_dict, success, timederivatives
    else:
        curr = curr[:-1] # take out p because haven't tested with other functions that use this
        curr.append(mean_x[-1])
        curr.append(var[-1])
        return curr, success, timederivatives

   
def get_equilibrium(params,N1_0,N2_0,g_of_x_vec):#, N1_0 = 0.5, N2_0 = 0.4, p_0 = 20, g_of_x_vec = None):
    '''
    finds the equilibrium s.t. N1, N2 > 0 using root for the population dynamics and group dynamics system
    RETIRED: if not given g_of_x_vec, then just has everyone initially solitary
    
    @returns:
    N1_eq, N2_eq, F_eq, P_eq, mean_x_eq
    '''
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    # if not isinstance(g_of_x_vec, np.ndarray):
    #     #print('hi')
    #     x_f = 2 if x_max > 2 else x_max
    #     g_of_x_vec = initiate_f_first_x(p_0, x_f, x_max)
        
    x0 = [N1_0, N2_0, *g_of_x_vec]
    out = root(fun = nullclines_no_P, x0 = x0, 
                                  args = (params))
    return out
def get_equilibria_from_init_pts(initial_points, tol_unique=1e-8, if_dict = False, **params):
    '''
    iterate through the initial points 
    and see if can use root to find equilibria

    This finds coexistence equilibria!!

    returns: 
    - np.array (if if_dict = False) consisting of [N1, N2, *g, mean_x, var]
    - dict (if if_dict = False) with keys equilibrium (np.ndarray of [N1, N2, *g]), mean_x, variance
    '''
    x_max = params['x_max']
    #curr_eq = np.zeros(2+x_max) #N1 = 0, N2 = 0, g(x) = 0
    results = []
    for i, point in enumerate(initial_points):
        out = get_equilibrium(params, N1_0 = point[0], N2_0 = point[1], g_of_x_vec = point[2:])

        # get the equilibrium values from the output
        sol = get_results_eq(out, x_max, if_dict = True)
        
        #P_eq, N1_eq, N2_eq, g_eq, mean_x_eq, success = [sol['P']
        
        if sol['success']: # the root finder found an equilibrium and it's "valid" (N1, N2, g(x) are in their ranges)
            if if_dict:
                new_eq = dict(equilibrium = np.array([sol['N1'], sol['N2'], *sol['g']]), 
                              mean_x = sol['mean_x'], var = sol['var'], p = sol['p'])
            else:
                new_eq = np.array([sol['N1'], sol['N2'], *sol['g'], sol['mean_x'], sol['var']])
            results.append(new_eq)
            #results = check_unique(results, new_eq, tol_unique)
    return results

    
def abs_nullclines_no_P(initialstate, params):
    return np.sum(np.abs(nullclines_no_P(initialstate, params)))



def nullclines_no_P(initialstate, params):
    '''
    returns the nullclines for N1, N2, g(1), g(2), ..., g(x_max)
    such that N1, N2 \neq 0
    @inputs
    initialstate = [N1, N2, g(1), ..., g(x_max)], type ndarray
    params = dictionary of params
    '''
    N1 = initialstate[0]
    N2 = initialstate[1]
    g_of_x_vec = initialstate[2:]

    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)


    
    

    N1_null, N2_null = N_nullclines(N1, N2, g_of_x_vec, xvec, **params)
    dgdT_vec = group_formation_model_non_dim(0, g_of_x_vec,N1,N2, params) # I put 0 for T
    
    return [N1_null, N2_null, *dgdT_vec]
    
def nullclines_big_prey_extinct(initialstate, params):
    '''
    returns the nullclines for N2,  g(1), g(2), ..., g(x_max)
    where N1 = 0 and N2 > 0
    @inputs
    initialstate = N2, g(1), ..., g(x_max)], type ndarray
    params = dictionary of params
    '''
    
    N2 = initialstate[0]
    g_of_x_vec = initialstate[1:]
    x_max = params['x_max']
    xvec = np.arange(1, x_max+1,1)
    _, N2_null = N_nullclines(0, N2, g_of_x_vec, xvec, **params)
    dgdT_vec = group_formation_model_non_dim(0, g_of_x_vec,0,N2, params) # put 0 for T, N1

    return [N2_null, *dgdT_vec]

def nullclines_small_prey_extinct(initialstate, params):
    '''
    returns the nullclines for N1,  g(1), g(2), ..., g(x_max)
    where N2 = 0 and N1 > 0
    @inputs
    initialstate = [N1, g(1), ..., g(x_max)],
    params = dictionary of params
    '''
    
    N1 = initialstate[0]
    g_of_x_vec = initialstate[1:]
    x_max = params['x_max']
    xvec = np.arange(1, x_max+1,1)
    N1_null, _ = N_nullclines(N1, 0, g_of_x_vec, xvec, **params)
    dgdT_vec = group_formation_model_non_dim(0, g_of_x_vec,N1,0, params) # put 0 for T, N1

    return [N1_null, *dgdT_vec]

def N_nullclines(N1, N2, g_of_x_vec, xvec, η1, η2, A1, A2, **params):
    '''
    dN1dT, dN2dT, the change in prey pop size versus time, non-dim'ed, divided by N_i
    @inputs:
    N1, N2 - non-dim'ed pred, big prey, and small prey pop sizes
    g_of_x_vec - array of g(1), g(2), ... , g(x_max)
    params - dic of params: must at least include H1, H2, α1_of_1, α2_of_1, s1, s2,
    '''

    
    α1 = fun_alpha1(xvec,**params) 
    α2 = fun_alpha2(xvec,**params) 

    # prey nonzero nullclines
    denominator = 1 + fun_H1(xvec,**params)*α1*N1 + fun_H2(xvec,**params)*α2*N2
    f1_no_N = A1*α1/denominator
    f2_no_N = A2*α2/denominator

    N1_null = η1 * (1-N1) - np.sum(g_of_x_vec * f1_no_N)
    N2_null = η2 * (1-N2) - np.sum(g_of_x_vec * f2_no_N)
    
    return N1_null, N2_null
    
def get_equilibrium_prey_i_extinct(params, i, Nj_0 = 0.4, 
                                p_0 = 20, g_of_x_vec = None):
    '''
    finds the equilibrium using root for the population dynamics and group dynamics system
    where N1 = 0
    if not given g_of_x_vec, then just has everyone initially solitary
    
    @returns:
    N1_eq, N2_eq, F_eq, P_eq, mean_x_eq
    '''
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    if not isinstance(g_of_x_vec, np.ndarray):
        #print('hi')
        x_f = 2 if x_max > 2 else x_max
        g_of_x_vec = initiate_g_first_x(x_f, x_max)
        
    x0 = [Nj_0, *g_of_x_vec]
    if i == 1:    
        out = root(fun = nullclines_big_prey_extinct, x0 = x0, 
                                  args = (params))
    elif i == 2:
        out = root(fun = nullclines_small_prey_extinct, x0 = x0, args = (params))
    return out  


def check_at_equilibrium2(N1,N2,g_of_x_vec, params):
    # check not negative
    curr = [N1, N2, *g_of_x_vec]
    condition_failed_1 = np.any(
        np.array(curr)<0
    )
    deriv_vec = full_model(
        0, curr, True, params
    )
    condition_failed_2 = np.any(np.abs(deriv_vec)>1e-8)
    if np.any([condition_failed_1, condition_failed_2]):#, condition_failed_3]):
        success = False
    else:
        success = True
    return success, deriv_vec
    # check derivative is zero
    # check sum x*g(x) = p
def get_results_eq(out, x_max, which_prey_extinct = -1, if_dict = False):
    '''
    Extracts the state variables at the equilibrium, calculates 
    mean experienced group size, and checks that the equilibrium 
    is valid (within the state variable domains)
    arguments:
        out (dict) - the output of root
        x_max (int) - maximum group size
        which_prey_extinct(int) - indicates if big prey (=1), small prey (=2) or neither (=-1) are extinct
        if_dict (bool) - whether to return outputs as dictionary
    @ returns: 
        - if if_dict = False: a tuple (P_eq, N1_eq, N2_eq, g_eq, mean_x_eq, success)
        - if_dict == True: a dictionary( with keys P, N1, N2, g, mean_x, success)
    '''
    xvec = np.arange(1,x_max+1,1)
    if which_prey_extinct == -1:
        g_eq = out.x[2:]
        N1_eq = out.x[0]
        N2_eq = out.x[1]
    else:
        g_eq = out.x[1:]
        Nj_eq = out.x[0]
        Ni_eq = 0
        N1_eq = Ni_eq if which_prey_extinct == 1 else Nj_eq
        N2_eq = Ni_eq if which_prey_extinct == 2 else Nj_eq
    P_eq = np.sum(xvec*g_eq); 
    
    mean_x_eq = mean_group_size_membership(g_eq,x_max,P_eq)

    # if predators are extinct, set mean experienced group size to 1
    if mean_x_eq < 1:
        mean_x_eq = 1
    # check not negative
    condition_failed_1 = np.any(np.array([P_eq, N1_eq, N2_eq, *g_eq, mean_x_eq])<0)
    # check root reached the end
    condition_failed_2 = out.success == False
    # check sum x*g(x) = p
    #condition_failed_3 = np.abs(np.sum(np.arange(1,x_max+1,1)*g_eq) - P_eq) > tol
    var = var_of_experienced_grp_size(g_eq)
    
    if np.any([condition_failed_1, condition_failed_2]):#, condition_failed_3]):
        success = False
        P_eq = np.nan; N1_eq = np.nan; N2_eq = np.nan; g_eq = np.nan; mean_x_eq = np.nan; var = np.nan
    else:
        success = True
    if if_dict:
        return dict(p=P_eq, N1 = N1_eq, N2 = N2_eq, g = g_eq, mean_x = mean_x_eq, var = var, success = success)
    else:
        return P_eq, N1_eq, N2_eq, g_eq, mean_x_eq, var, success

def initiate_g_first_x(x_f, x_max):
    
    g0 = np.zeros(x_max) + 1e-4
    g0[0:x_f] = 1
    return g0

def iterate_and_solve_equilibrium(params, t_f = 1000, tol = 1e-8, if_dict = False):
    '''
    iterates from a standard start point that tends to work
    then uses root to find equilibrium

    @returns
     - if if_dict == True: dictionary with keys equilibrium, mean_x, var, success
     - if if_dict == False; tuple of P, N1,N2,g, mean_x, var, success
     * note success is (BOOL for whether at equilibrium), 
    '''
    x_max = params['x_max']
    x_f = 3 if x_max >=3 else 2
    y0 = [0.71, 0.7, *initiate_g_first_x(3, x_max)]
    out2 = bounded_ivp(y0, params, t_f = t_f) 
    T, N1, N2, P, g_of_x_vec, mean_x = out2

    # extract new starting point
    traj = [N1,N2,*g_of_x_vec]
    curr = [item[-1] for item in traj]
    print(curr)

    out = get_equilibrium(params, N1_0 = curr[0], N2_0 = curr[1], 
                          g_of_x_vec = curr[2:])
    sol =get_results_eq(out,x_max, if_dict = if_dict) # P, N1, N2, g, mean_x, var, success


    if if_dict:
        return dict(equilibrium = np.array([sol['N1'], 
                                            sol['N2'], *sol['g']] ), 
                    mean_x = sol['mean_x'], var = sol['var'], 
                    success = sol['success'], p = sol['p'])
    else:
        return sol # tuple of P, N1, N2, g, mean_x, var, success
    
def get_equilibria_from_init_pts_i_extinct(initial_points, i, **params):
    '''
    iterate through the initial points and see if can use root to find equilibria
    prey i (1 or 2) extinct
    append to results if found an equilibrium

    '''
    x_max = params['x_max']
    curr_eq = np.zeros(2+x_max) #N1 = 0, N2 = 0, g(x) = 0
    results = []
    for point in initial_points:
        out = get_equilibrium_prey_i_extinct(params, i, Nj_0 = point[2-i], 
                                             g_of_x_vec = point[2:])
        sol = get_results_eq(out, x_max, which_prey_extinct = i, if_dict = True)
        
        if sol['success']: # the root finder found an equilibrium and it's "valid" (N1, N2, g(x) are in their ranges)
            new_result = dict(equilibrium = np.array([sol['N1'], sol['N2'], *sol['g']]),
                              p = sol['p'],
                              mean_x = sol['mean_x'], 
                              var = sol['var'])

            # append new_eq if it's unique from the last one
            #results = check_unique(results, new_eq, tol_unique)
            results.append(new_result)

    return results
######################################################################

# check and update functions below, if still needed

    
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
                sol =get_results_eq(out,x_max)
                P = sol['P']; N1 = sol['N1']; N2 = sol['N2']; g = sol['g'];  mean_x = sol['mean_x']; success = sol['success']
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



''' 
retired i think
def check_at_equilibrium(final_distribution, P, N1, N2,**params):
    
    # check dg(x)/dT \approx 0
    # @ returns: array dgdT_, and 1 if at equilibrium or 0 if not
    
    T = 1 # this doesn't matter
    dgdT_ = group_formation_model_non_dim(T, final_distribution,N1,N2, params)
    not_at_equilibrium = np.abs(dgdT_) > 1e-8
    if sum(not_at_equilibrium) > 0: # at least one dg(x)/dt is not zero
        return dgdT_, 0 # 0 means not at equilibrium
    else:
        return dgdT_, 1 # 1 means not at equilibrium

'''
