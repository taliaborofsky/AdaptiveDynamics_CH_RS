import numpy as np
from multiprocessing import Pool, cpu_count
from group_w_pop_funs import bounded_ivp, get_initial_points  # Ensure this function works with multiprocessing

def classify_if_extinction(sol, extinction_threshold):
    '''

    Check if the trajectory (in sol) ended in an extinction equilibrium,
    and classify the type of extinction equilibrium
    
    Args:
    sol - dictionary of T, N1, N2, g, mean_x, p from bounded_ivp
    extinction_threshold: (float) if a state variable is less than this,
        it's extinct
    Returns: string that is one of
        "predator_extinct_both_prey_extinct", 
        "predator_extinct_big_prey_extinct",
        "predator_extinct_small_prey_extinct", "predator_extinct",
        "big_prey_extinct", "small_prey_extinct", "no_extinction"
    '''
    # get final point of trajectory
    final_g = sol['g'][:,-1]  # Extract g values from the solution
    total_predators = sol['p'][-1]  
    final_N1 = sol['N1'][-1]
    final_N2 = sol['N2'][-1]
    
    if total_predators < extinction_threshold:
        if final_N1 < extinction_threshold and final_N2 < extinction_threshold:
            return "predator_extinct_both_prey_extinct"
        elif final_N1 < extinction_threshold and final_N2 >= extinction_threshold:
            return "predator_extinct_big_prey_extinct"
        elif final_N2 < extinction_threshold and final_N1 >= extinction_threshold:
            return "predator_extinct_small_prey_extinct"
        else:
            return "predator_extinct"
    elif final_N1 < extinction_threshold and final_N2 >= extinction_threshold:
        return "big_prey_extinct"
    elif final_N2 < extinction_threshold and final_N1 >= extinction_threshold:
        return "small_prey_extinct"
    return "no_extinction"
    
def simulate_single_point(args):
    """
    Simulate the system for a single set of initial conditions and check for extinction.
    
    Args:
        args (tuple): Contains (y0, t_f, params, extinction_threshold).
    
    Returns:
        bool: True if predators go extinct, False otherwise.
    """
    y0, t_f, params, extinction_threshold = args
    # Simulate the system using the bounded IVP solver
    sol = bounded_ivp(y0, params, t_f, if_dict = True)
    
    # Check for extinction, classify type
    return classify_if_extinction(sol, extinction_threshold)


def extinction_analysis_multiprocessing(
    num_points, t_f, params, p_upper = 3, 
    extinction_threshold=1e-6, n_jobs=None
):
    """
    Perform extinction analysis using multiprocessing for parallel execution.
    
    Args:
        num_points (int): Number of random initial points.
        iterations (int): Number of iterations (e.g., 2000).
        params (dict): Parameters for the system.
        extinction_threshold (float): Threshold to consider extinction.
        n_jobs (int or None): Number of parallel processes. If None, uses all available cores.
    
    Returns:
        float: Proportion of initial points leading to extinction.
    """
    # Step 1: Generate initial points
    init_pts = get_initial_points(num_points, p_upper = 3, **params)
    N1 = init_pts[:,0]
    N2 = init_pts[:,1]
    g_vectors = init_pts[:,2:]
    
    # Prepare the initial conditions and arguments for each simulation
    simulation_args = [
        (np.concatenate(([N1[i], N2[i]], g_vectors[i])), 
         t_f, params, extinction_threshold)
        for i in range(num_points)
    ]
    
    # Determine the number of processes to use
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Step 2: Use a Pool to parallelize simulations
    with Pool(processes=n_jobs) as pool:
        extinction_results = pool.map(simulate_single_point, simulation_args)
    
    # Step 3: Calculate proportions for each extinction type
    extinction_types = [
        "predator_extinct_both_prey_extinct",
        "predator_extinct_big_prey_extinct",
        "predator_extinct_small_prey_extinct",
        "predator_extinct",
        "big_prey_extinct",
        "small_prey_extinct",
        "no_extinction"
    ]
    extinction_counts = {
        etype: extinction_results.count(etype) for etype in extinction_types
    }

    # Normalize counts to proportions
    total = sum(extinction_counts.values())
    extinction_proportions = {etype: count / total for etype, count in extinction_counts.items()}

    return extinction_proportions
