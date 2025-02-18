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
        return tildeδ * ( - x * g(x) + (x+1) * g(x+1) )
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
        deaths_vec = Tx * np.array([fun_deaths(x) for x in range(1,x_max+1)])
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
    g_of_x_vec = y_trajectory[2:] #dimensions x_max x T
    xvec = np.arange(1,x_max+1,1)[:,np.newaxis]
    p = np.sum(xvec*g_of_x_vec,0)
    mean_x = mean_group_size_membership(g_of_x_vec.T, x_max, p)
    var_exp_x = var_of_experienced_grp_size(g_of_x_vec)
    if if_dict:
        return dict(
            T=T, N1 = N1, N2 = N2, p = p, 
            g = g_of_x_vec, mean_x = mean_x, var = var_exp_x
        )
    else:
        return T, N1, N2, p, g_of_x_vec, mean_x, var_exp_x


    
def transformed_model(T, u0, arg, params):
    #For issue where need solve_ivp to keep some variable y s.t y > 0.
    # Soln from stack exchange: https://stackoverflow.com/questions/67487208/bounds-for-solve-ivp-integration
    # Alternatively, replace the density with y=a*exp(b*u) with sensibly chosen values for a and b, 
    # then du/dt = dy/dt/(b*y). 
    # Then g cannot become negative
    a = 1; b = 1
    y0 = a * np.exp(b*u0) # transform back into original coordinates
    y_ = full_model(T,y0, arg, params) # find derivative
    y_ = np.array(y_)
    u_ = y_/(b*y0) # find derivative of transformed coordinates
    return u_

def var_of_experienced_grp_size(group_densities, epsilon=1e-12):   
    '''
    Calculate the variance of experienced group size 
    from a vector of group densities.
    Args:
        group_densities (array-like): Matrix (or vector) of group densities.
            - If a vector: [g(1), g(2), ..., g(xmax)].
            - If a matrix: shape (xmax, time_steps), with each column representing densities at a time point. 
        epsilon (float): Small value added to the denominator to avoid division by zero (numerical regularization)
    Returns:
        float: Variance of the group size. Is 0 if predators extinct.
    '''
    group_densities = np.array(group_densities)

    # Group sizes (x = 1, 2, ..., xmax)
    x = np.arange(1, group_densities.shape[0] + 1)# shape: (xmax,) 
    if group_densities.ndim == 1:
        axis = None
    elif group_densities.ndim == 2:
        axis = 0 # for summing
        x = x[:,np.newaxis]
    
    # calculate predator population density p
    p = np.sum(x*group_densities, axis = axis)
    p = np.maximum(p, epsilon) # numerical regularization

    # calculate probability a predator is in a group of size x
    prob_experience_x = x*group_densities/p

    # Compute E[X] (mean experienced group size)
    mean_exp_x = np.sum(x * prob_experience_x, axis = axis)

    # Compute E[X^2] (mean of squared group sizes) for each column
    mean_x_squared = np.sum((x**2) * prob_experience_x, axis=axis) 
    
    # Variance = E[X^2] - (E[X])^2
    variance = mean_x_squared - (mean_exp_x**2)
    return variance

def get_initial_points(num_initial, x_max, p_upper = None, **params):
    ''' 
    get initial points to feed to the root finder 
    '''
    # α2_1 = params['α2_of_1']
    # α1_xm = fun_alpha1(x_max, **params)

    # Generate random values for N1, N2, and g(x) for each initial point
    np.random.seed(42)
    
    # N1 and N2 are between 0 and 1, not including 0
    N1_values = np.random.uniform(0.01, 1, num_initial)  # Shape: (num_initial,)
    N2_values = np.random.uniform(0.01, 1, num_initial)  # Shape: (num_initial,)

    if p_upper == None:
        gx_upper = 3# try this out
        # g(x) is between 0 and gx_upper for each x = 1, 2, ..., x_max
        g_values = np.random.uniform(0.01, gx_upper, (num_initial, x_max))  # Shape: (num_initial, x_max)
    else:
        g_values = get_random_g_bounded_p(p_upper, num_initial, x_max)
                                          
    # Combine N1, N2, and g(x) into a single array
    initial_points = np.hstack((N1_values[:, np.newaxis],  # Add N1 as the first column
                                N2_values[:, np.newaxis],  # Add N2 as the second column
                                g_values))  # Add g(x) as the remaining columns
    
    return initial_points
def update_params(param_key, param, params_base):
    '''
    given params_base, makes a copy dictionary of parameters
    and updates with the new param at param_key

    noe if param_key is scale, updates β1 and H1 entries

    @ returns: params
    '''
    params = params_base.copy()
        
    if param_key == "scale": # this means β1/β2 = H1/H2 and β2, H2 are set
        params['β1'] = params['β2']*param
        A_frac = params_base['A1']/params_base['A2']
        params['H1a'] = params['H2a'] * param * A_frac
        params['H1b'] = params['H2b'] * param * A_frac
        params['η1'] = params['η2']/param
    else:
        params[param_key] = param

        if "scale" in params:
            params = update_params("scale", params["scale"], params) # make sure everything still scaled correctly
                
    return params
def get_list_of_trajectories(params, t_f=1000, initial_points=None, num_init=4):
    '''
    Generates a list of trajectories by simulating the system from a set of initial conditions.

    Args:
        params (dict): Dictionary of system parameters used in the simulation.
        t_f (int, optional): Final time for the simulation (default is 1000).
        initial_points (array-like, optional): Initial points for the simulation.
            Each point is a list of the form [N1, N2, g(1), g(2), ..., g(xm)].
            If None, initial points will be generated using `get_initial_points`.
        num_init (int, optional): Number of initial points to generate if `initial_points` is None (default is 4).

    Returns:
        list of dict: A list of trajectories. Each trajectory is a dictionary returned by the `bounded_ivp` function
                      and includes keys like 'T', 'N1', 'N2', 'g', 'p', and 'mean_x'.

    Behavior:
        - If `initial_points` is not provided or invalid, generates `num_init` initial points using `get_initial_points`.
        - Simulates the system for each initial point using the `bounded_ivp` function.
        - Appends the resulting trajectory dictionary to the output list.

    Dependencies:
        - `get_initial_points`: Function to generate initial points if not provided.
        - `bounded_ivp`: Function that simulates the system and returns a trajectory as a dictionary.
    '''
    if type(initial_points) != np.ndarray: # so it's None or some invalid entry
        print("generating initial points")
        initial_points = get_initial_points(num_init,**params)
    trajectories = []
    for i, init_state in enumerate(initial_points):
        results = bounded_ivp(init_state, params, if_dict=True)
        trajectories.append(results)
    return trajectories # each is a dictionary

def get_random_g_bounded_p(p_upper, num_initial, x_max):
    """
    Generates random g values such that sum(x * g(x)) <= p_upper.
    Repeats the process until num_initial valid g vectors are obtained.
    
    Args:
        p_upper (float): Upper bound for the sum(x * g(x)).
        num_initial (int): Desired number of valid g vectors.
        x_max (int): Maximum group size.

    Returns:
        np.ndarray: An array of shape (num_initial, x_max) containing valid g vectors.
    """
    g_list = []
    while len(g_list) < num_initial:
        g_mat = np.zeros((num_initial, x_max))
        preds_left = p_upper * np.ones(num_initial)  # Track remaining predator allocation for each vector
        
        for x in range(1, x_max + 1):
            gi = np.random.uniform(0.01, preds_left / x, num_initial)
            g_mat[:, x - 1] = gi
            
            # Update current predator population
            preds_left -= x * gi
            preds_left[preds_left < 0] = 0  # Ensure no negative remaining capacity
    
        # Calculate total population p for each g vector
        p_vals = np.sum(g_mat * np.arange(1, x_max + 1), axis=1)
        
        # Filter valid g vectors where total population <= p_upper
        valid_indices = np.where(p_vals <= p_upper)[0]
        valid_g = g_mat[valid_indices]
        
        # Add valid g vectors to the list
        g_list.extend(valid_g.tolist())

    # Limit the result to exactly num_initial vectors
    g_good = np.array(g_list[:num_initial])

    return g_good
