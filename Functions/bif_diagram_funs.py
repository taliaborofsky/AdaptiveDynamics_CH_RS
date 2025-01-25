import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(1, 'Functions')
#from scipy.optimize import root
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from local_stability_funs import *
from equilibria_funs import *
from sim_graph_funs import *
from matplotlib.lines import Line2D

param_lab_dic = dict(η1 = "Growth of big prey, " + r'$\eta_1$', 
            η2 = "Growth of small prey, " + r'$\eta_1$', 
            A = "Relative attack rates, " + r'$A$', 
            β1 = "Benefit of big prey, " + r'$\beta_1$',
            β2 = "Benefit of small prey, " + r'$\beta_1$', 
            H1= "Handling time of big prey, " + r'$H_1$', 
            H2= "Handling time of small prey," + r'$H_2$', 
            α1_of_1= "Capture probability of big prey\nby solitary predator, " + r'$\alpha_1(1)$',
            α2_of_1="Capture probability of small prey\nby solitary predator, " + r'$\alpha_2(1)$', 
            s1="Critical group size for big prey, " + r'$s_1$', 
            s2="Critical group size for small prey, " + r'$s_2$', 
            α2_fun_type = 'Shape of capture probability for small prey',
            x_max = 'Max group size, ' + r'$x_{max}$',
            d = "Decision accuracy, " + r'$d$',
            Tx = "Timescale of group dynamics, " + r'$T_x$',
            scale = "Prey size ratio, " + r'$\beta_1/\beta_2$')


    

def get_perturbations(equilibrium, num, strength):
    '''
    Generate perturbed versions of a given equilibrium.

    Args:
        equilibrium (array): The equilibrium point to perturb. [N1, N2, g(1), g(2), ..., g(xmax)]
        num (int): Number of perturbed points to generate.
        strength (float): Perturbation strength.

    Returns:
        np.ndarray: Array of perturbed points.
    '''
    #  initiate empty array
    perturbations = np.zeros((num, equilibrium.shape[0]))
    # fill array
    for i in range(0,num):
        #perturb
        curr_perturbed = equilibrium + strength * np.random.standard_normal(len(equilibrium))
        curr_perturbed[0:2] = np.clip(curr_perturbed[0:2], 1e-4, 1-1e-4) # make sure 0<N1,N2 <1
        curr_perturbed[2:] = np.clip(curr_perturbed[2:], a_min = 1e-8, a_max = None)
        perturbations[i,:] = curr_perturbed
    return perturbations

    # returns "Stable (attractive)", "Unstable", "Marginally stable (needs further analysis)"
    # and "Indeterminate stability (needs further analysis)"


    

def make_equilibria_dataframe(rows, param_key, x_max):
    # Define the column names
    columns = [param_key, 'N1', 'N2'] + \
              [f'g{i}' for i in range(1, x_max + 1)] + \
              ['mean_x', 'var', 'equilibrium_type', 'stability']
    
    # Initialize the DataFrame with specific dtypes
    dtype_dict = {col: 'float64' for col in columns[:-2]}  # All numeric columns
    dtype_dict.update({'equilibrium_type': 'object', 'stability': 'object'})  # Categorical columns
    
    df = pd.DataFrame(rows, columns=columns).astype(dtype_dict)  # Set explicit dtypes
    return df




def store_and_perturb(rows, equilibria, num_perturbations, perturb_strength, params, param, equilibrium_type):
    '''
    stores equilibrium in a row to eventually add to dataframe and gets perturbations
    
    arguments
        rows (list): list of rows to add to dataframe, 
                        with columns param, N1, N2, g(1),...,g(xmax), mean_x, var, equilibrium_type, stability
        equilibria (list): list of dictionaries with keys equilibrium, mean_x, var (and maybe success?)
        num_perturbations (int): number of perturbationsto make from each equilibrium
        perturb_strength (float): how much to perturb by (typically 0.01)
        param (float): the param value (that's being varied)
    returns
        tuple: rows (list) and perturbed_pts(list)
    '''
    
    # Initiate empty perturbed_pts array
    perturbed_pts = np.array([]).reshape(0, 2 + params['x_max'])
    if np.size(equilibria) > 0:
        for eq_dic in equilibria:
            equilibrium = eq_dic['equilibrium']
            perturbed_pts = get_perturbations(equilibrium, 
                                              num_perturbations, perturb_strength)
            stability = classify_equilibrium(equilibrium, params) 
            
            # Prepare row for DataFrame
            row = [param, *equilibrium, eq_dic['mean_x'], eq_dic['var'],
                   equilibrium_type, stability]
            rows.append(row)
    return rows, perturbed_pts
    
def get_prey_extinct_equilibria(param_key, param_vec, i, num_init = 10,
                               num_perturbations = 2,
                          perturb_strength = 0.01, 
                          **params_base):

    equilibrium_type = ['Big Prey Extinct','Small Prey Extinct'][i-1]
    # Initiate empty perturbed_pts array
    perturbed_pts = np.array([]).reshape(0, 2 + params_base['x_max'])

    # initiate rows to append later
    rows = []

    # get random initial points
    init_pts0= get_initial_points(num_initial=num_init,**params_base.copy())

    for param in param_vec:
        # Update param dictionary
        params = update_params(param_key, param, params_base)
        # Add perturbed_pts to init_pts
        init_pts = np.vstack((init_pts0, perturbed_pts))

        equilibria = get_equilibria_from_init_pts_i_extinct(
            init_pts, i, **params
        )
        rows, perturbed_pts = store_and_perturb(rows, equilibria, num_perturbations, 
                                                perturb_strength, params, param, equilibrium_type)
        # if np.size(equilibria) > 0:
        #     for eq_dic in equilibria:
        #         equilibrium = eq_dic['equilibrium']
        #         perturbed_pts = get_perturbations(equilibrium, 
        #                                           num_perturbations, perturb_strength)
        #         stability = classify_equilibrium(equilibrium, params) 
                
        #         # Prepare row for DataFrame
        #         row = [param, *equilibrium, eq_dic['mean_x'], eq_dic['var'],
        #                'Coexistence', stability]
        #         rows.append(row)
        
    return rows

def get_coexistence_equilibria(param_key, param_vec, num_init=10,
                               num_perturbations=2, perturb_strength=0.01, t_f=1100,
                               **params_base):
    '''
    Get equilibria using N1 >0, N2 > 0 nullclines.
    '''
    # Initiate empty perturbed_pts array
    perturbed_pts = np.array([]).reshape(0, 2 + params_base['x_max'])


    # initiate rows to append later
    rows = []

    # Get random initial points
    init_pts0 = get_initial_points(num_initial=num_init, **params_base.copy())
    
    for param in param_vec:
        # Update param dictionary
        params = update_params(param_key, param, params_base)
        
        # Add perturbed_pts to init_pts
        init_pts = np.vstack((init_pts0, perturbed_pts))

        # Use init_pts to get equilibria in which N1, N2 > 0
        coexist_equilibria = get_equilibria_from_init_pts(
            init_pts, tol_unique=1e-8, **params.copy(), if_dict=True
        )
    
        # Iterate if no points found using solver
        if np.size(coexist_equilibria) == 0:
            initialstate = np.array([.7, .7, *initiate_g_first_x(3, params["x_max"])])
            coexist_eq, success, _ = iterate_to_eq(initialstate, t_f, params, if_dict = True)
            coexist_equilibria = [coexist_eq] if success else []

        # Classify stability and store equilibria
        rows, perturbed_pts = store_and_perturb(rows, coexist_equilibria, num_perturbations, 
                                                perturb_strength, params, param, 'Coexistence')
        # if np.size(coexist_equilibria) > 0:
        #     for eq_dic in coexist_equilibria:
        #         equilibrium = eq_dic['equilibrium']
        #         perturbed_pts = get_perturbations(equilibrium, 
        #                                           num_perturbations, perturb_strength)
        #         stability = classify_equilibrium(equilibrium, params) 
                
        #         # Prepare row for DataFrame
        #         row = [param, *equilibrium, eq_dic['mean_x'], eq_dic['var'],
        #                'Coexistence', stability]
        #         rows.append(row)


    
    return rows

def get_predator_extinct_equilibria(param_key, param_vec, params_base):
    '''
    Find equilibria where predators are extinct.

    Args:
        param_key (str): Parameter to vary.
        param_vec (array): Values of the parameter to iterate over.
        params_base (dict): Base system parameters.

    Returns:
        array: Arrays of rows with param being varied in first entry
    '''
    rows = []
    for param in param_vec:
        params = update_params(param_key, param, params_base)
        # get extinction
        g_of_x_extinct = np.zeros(params['x_max'])
        pred_extinct_equilibria = [[1,1,*g_of_x_extinct],[1,0,*g_of_x_extinct],
                               [0,1,*g_of_x_extinct]]
        for equilibrium in pred_extinct_equilibria:
            stability = classify_equilibrium(equilibrium, params) 
            # Prepare row for DataFrame
            mean_x = 1
            var = 0
            row = [param, *equilibrium, mean_x, var,
                       'Predator Extinct', stability]
            rows.append(row)
    return(rows)


def get_bif_input(param_key, param_vec, params_base, num_init = 10, 
                      num_perturbations = 2,
                          perturb_strength = 0.01,t_f = 1100):
    '''
    Generate a dataframe with inputs to use for plot a bifurcation diagram for the system.
    Dependencies:
        get_coexistence_equilibria
        get_predator_extinct_equilibria
        get_prey_extinct_equilibria
    Args:
        param_key (str): Parameter to vary.
        param_vec (array): Values of the parameter to iterate over.
        params_base (dict): Base system parameters.
        num_init (int): Number of random initial points to generate.
        num_perturbations (int): Number of perturbations to generate.
        perturb_strength (float): Perturbation strength.
        t_f (float): Simulation end time.
    returns:
        dataframe: a dataframe with columns `param_key`, N1, N2, g(1), g(2), ..., g(xmax), mean_x, var, equilibrium_type, stability
        
    '''
    # get coexistence equilibria
    rows_co = get_coexistence_equilibria(
        param_key, param_vec,
        num_init, num_perturbations, perturb_strength, t_f = t_f,
        **params_base)
    rows_pred_extinct = get_predator_extinct_equilibria(param_key, param_vec, params_base)
    rows_prey1_extinct = get_prey_extinct_equilibria(param_key, param_vec, i=1, num_init = num_init,
                               num_perturbations = num_perturbations,
                          perturb_strength = perturb_strength, 
                          **params_base) 
    rows_prey2_extinct = get_prey_extinct_equilibria(param_key, param_vec, i=2, num_init = num_init,
                               num_perturbations = num_perturbations,
                          perturb_strength = perturb_strength, 
                          **params_base) 

    # Combine rows, skipping any empty arrays
    rows_list = [rows_co, rows_pred_extinct, rows_prey1_extinct, rows_prey2_extinct]
    non_empty_rows = [r for r in rows_list if len(r) > 0] 

    # Perform stacking only on non-empty arrays
    if non_empty_rows:  # Ensure there's something to stack
        rows = np.vstack(non_empty_rows)
    else:
        rows = np.empty((0, params_base['x_max'] + 7))  # Create an empty array with the correct number of columns
    # add all rows to a DataFrame
    df = make_equilibria_dataframe(rows, param_key, params_base['x_max'])
    return df
    
def plot_bif_diagrams(param_key, df):
    '''
    Generate and plot a bifurcation diagram for the system.
    Dependencies:
        plot_equilibria
        get_bif_input
        format_bif_diagrams
    Args:
        param_key (str): Parameter to vary. on x axis
        df: dataframe with info on equilibria, generated using get_bif_input
        

    Returns:
        tuple: 4 Figures for various bifurcation diagrams, 1 figure with just the legend
    '''
    figx, axx = plt.subplots(1,1)
    figN1, axN1 = plt.subplots(1,1)
    figN2, axN2 = plt.subplots(1,1)
    figNsum, axNsum = plt.subplots(1,1)
    figvar, axvar = plt.subplots(1,1)
    
    # plot
    color_map = {
        'Coexistence': 'black',
        'Predator Extinct': 'blue',
        'Big Prey Extinct': 'red',
        'Small Prey Extinct': 'cyan'
    }
    stable_markers = dict(label = "Stable ", marker = "o", s = 5)
    unstable_markers = dict(label = "Unstable ", marker = 'D', 
                            s = 30, facecolors='none')
    marker_map = {
        "Stable (attractive)": stable_markers,    # Filled circle
        "Unstable": unstable_markers   # Open diamond
    }

    y_col = 'N1'
    for (equilibrium_type, stability), group in df.groupby(['equilibrium_type', 'stability']):
        color = color_map.get(equilibrium_type, 'gray')  # Default to gray if unknown type
        marker_dict = marker_map.get(stability, 'x')         # Default to 'x' if unknown stability
        # edgecolor = 'black' if stability == 'Unstable' else None

        # Scatter plot for each group
        for y_col, ax in zip(['mean_x','N1','N2','var'],[axx, axN1, axN2, axvar]):
            ax.scatter(
                group[param_key], group[y_col],
                color=color, **marker_dict
            )
        axNsum.scatter(group[param_key], group['N1']+group['N2'],
                color=color, **marker_dict)

    format_bif_diagrams(axx, axN1, axN2, axNsum, axvar, param_key)
        
    return dict(mean_x = figx, N1 = figN1, N2 = figN2, N1_plus_N2 =figNsum, var=figvar)

def plot_bif_legend():
    # add legend on separate figure
    
    figlegend, axlegend = plt.subplots(1,1)
    
    stable_markers = dict(marker = "o", markersize = 3, linestyle = '')
    unstable_markers = dict(marker = 'D', color='none',markersize=6, linestyle='')
    
    legend_handles = [
        # Stable (dots)
        Line2D([0], [0], **stable_markers, color='black', label='Stable, Coexist'),
        Line2D([0], [0], **stable_markers, color='blue', label='Stable, Predator Extinct'),
        Line2D([0], [0], **stable_markers, color='red', label='Stable, Prey 1 Extinct'),
        Line2D([0], [0], **stable_markers, color='cyan', label='Stable, Prey 2 Extinct'),
    
        # Unstable (unfilled diamonds)
        Line2D([0], [0],  **unstable_markers, markeredgecolor='black', label='Unstable, Coexist'),
        Line2D([0], [0], **unstable_markers, markeredgecolor='blue', label='Unstable, Predator Extinct'),
        Line2D([0], [0], **unstable_markers, markeredgecolor='red', label='Unstable, Prey 1 Extinct'),
        Line2D([0], [0], **unstable_markers, markeredgecolor='cyan', label='Unstable, Prey 2 Extinct'),
    ]
    axlegend.axis('off') 
    axlegend.legend(handles = legend_handles, loc='center', ncol = 1,
                   fontsize = 20)

    return figlegend, axlegend
    
def format_bif_diagrams(axx, axN1, axN2, axNsum, axvar, param_key):
    '''
    Format bifurcation diagram axes and add a legend.

    Args:
        axx, axN1, axN2, axNsum (Axes): Axes for plotting.
        param_key (str): Parameter used for the x-axis.

    Returns:
        None
    '''
    param_lab_dic = dict(η1 = "Growth of big prey, " + r'$\eta_1$', 
                η2 = "Growth of small prey, " + r'$\eta_1$', 
                A = "Relative attack rates, " + r'$A$', 
                β1 = "Benefit of big prey, " + r'$\beta_1$',
                β2 = "Benefit of small prey, " + r'$\beta_1$', 
                H1= "Handling time of big prey, " + r'$H_1$', 
                H2= "Handling time of small prey," + r'$H_2$', 
                α1_of_1= "Capture probability of big prey\nby solitary predator, " + r'$\alpha_1(1)$',
                α2_of_1="Capture probability of small prey\nby solitary predator, " + r'$\alpha_2(1)$', 
                s1="Critical group size for big prey, " + r'$s_1$', 
                s2="Critical group size for small prey, " + r'$s_2$', 
                α2_fun_type = 'Shape of capture probability for small prey',
                x_max = 'Max group size, ' + r'$x_{max}$',
                d = "Decision accuracy, " + r'$d$',
                Tx = "Timescale of group dynamics, " + r'$T_x$',
                scale = "Prey size ratio, " + r'$\beta_1/\beta_2$')
    
    Nsumlab = "Sum of Prey Densities,\n "+r'$N_1 + N_2$'
    for ax, ylab in zip([axx, axN1, axN2, axNsum, axvar],
                        [mean_x_lab, N1lab, N2lab,
                        Nsumlab, standard_labs['var']]):
        xlab = param_lab_dic[param_key]
        format_ax(ax,xlab,ylab, xlim = None, ylim=None,
              fs_labs = 20, if_legend = False)



#### retired or not used anymore
def check_unique(results, new_eq_dic, tol_unique = 1e-8):
    '''
    Check if a new equilibrium is unique and append it to the results if it is.

    Args:
        results (list): List of dictionaries of existing equilibria.
        new_eq_dic (dict): New equilibrium to check.
        tol_unique (float): Tolerance for determining uniqueness.

    Returns:
        list: Updated results with the new equilibrium appended if unique.
    '''
    if len(results)>0:
        for result in results:
            if np.any(np.abs(new_eq_dic['equilibrium'] - result['equilibrium']) > tol_unique):
                results.append(new_eq_dic)
    else:
        results.append(new_eq_dic)
    return results
