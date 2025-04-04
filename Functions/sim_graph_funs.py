import numpy as np
import matplotlib.pyplot as plt
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.integrate import solve_ivp
from scipy.optimize import root
from local_stability_funs import *
#from equilibria_funs import *


#colors = ['k','r','b','cyan', 'magenta','orange',
#         'gray', 'green']
colors = ['r', 'orange', 'magenta', 'purple', 'blue', 
          'cornflowerblue', 'turquoise','k', 'gray']
colors_x = ['k', 'b', 'r', 'm']
#markers = ["o","","v", ""]
Plab = r'$p$,  Pred. Pop Density'
Pscaledlab = r'$P$,  Pred. Scaled Density'
N1lab = r'$N_1$, Scaled Big Prey'+ '\nDensity'
N2lab = r'$N_2$, Scaled Small Prey' + '\nDensity'
Tlab = r'$T$, Scaled time'
mean_x_lab = "Mean Experienced\nGroup Size"
freq_x_lab = r'Freq$(x)$'
β1lab = r'$\beta_1$'
gxlab = r'g$(x)$'
def gxlabel(i):
    first_three = ["Density of singletons,\n" + r'$g(1)$', 
                   "Density of pairs,\n" + r'$g(2)$', 
                   "Density of trios,\n" + r'$g(3)$'] 
    if i <4:
        return first_three[i-1]
    else:
        return "Density of groups\n of size %d, "%i + r'$g($' + str(i) + r'$)'
standard_labs = dict(
    p = r'$p$,  Pred. Density', N1 = r'$N_1$, Scaled Big Prey'+ '\nDensity', 
    N2 = r'$N_2$, Scaled Small Prey' + '\nDensity', T = r'$T$, Scaled time',
    mean_x = "Mean Experienced\nGroup Size",
    freq_x = r'Freq$(x)$', β1 = r'$\beta_1$', gx = r'g$(x)$', 
    var = "Variance of\nExperienced Group Size"
)

figure_ops = dict(bbox_inches = 'tight', 
                  format = 'eps', dpi=300, transparent=False,
                 pad_inches=0)

param_lab_dic = dict(η1 = "Growth of big prey, " + r'$\eta_1$', 
            η2 = "Growth of small prey, " + r'$\eta_2$', 
            A = "Relative Attack Rate of Both prey," + r'$A_1=A_2=A$',
            A1 = "Relative Attack rate of Big Prey, " + r'$A_1$', 
            A2 = "Relative Attack rate of Small Prey, " + r'$A_2$', 
            β1 = "Benefit of big prey, " + r'$\beta_1$',
            β2 = "Benefit of small prey, " + r'$\beta_1$', 
            H1a= "Group-independent Handling\ntime of big prey, " + r'$H_{1a}$', 
            H2a= "Group-independent Handling\ntime of small prey," + r'$H_{2a}$', 
            H1b= "Group-dependent Handling\ntime of big prey, " + r'$H_{1b}$', 
            H2b= "Group-dependent Handling\ntime of small prey," + r'$H_{2b}$', 
            α1_of_1= "Capture probability of big prey\nby solitary predator, " + r'$\alpha_1(1)$',
            α2_of_1="Capture probability of small prey\nby solitary predator, " + r'$\alpha_2(1)$', 
            s1="Critical group size for big prey, " + r'$s_1$', 
            s2="Critical group size for small prey, " + r'$s_2$', 
            α2_fun_type = 'Shape of capture probability for small prey',
            x_max = 'Max group size, ' + r'$x_{max}$',
            d = "Decision accuracy, " + r'$d$',
            Tx = "Timescale of group dynamics, " + r'$T_x$',
            scale = "Prey size ratio, " + r'$\beta_1/\beta_2$')




def format_ax(ax,xlab,ylab, xlim = None, ylim=None,
              fs_labs = 20, fs_legend = 16, if_legend = False,
             ncol_legend = 1):
    '''
    applies my formatting to provided ax:
    sets xlim, ylim, and fontsizes. gets rid of right and top borders, 
    adds legend
    @ no return
    '''
    ax.set_xlabel(xlab, fontsize = fs_labs)
    ax.set_ylabel(ylab, fontsize = fs_labs)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    if if_legend:
        ax.legend(fontsize = fs_legend, ncol = ncol_legend)
        
def get_results(out2,x_max):
    '''
    gets results from output of simulation
    out: dictionary that's output of solve_ivp
    x_max: max group size

    @returns:
    T, N1, N2, P, g_of_x_vec, mean_x
    '''
    N1, N2 = out2.y[0:2]
    g_of_x_vec = out2.y[2:]
    xvec = np.arange(1,x_max+1,1)
    p = np.sum(xvec*g_of_x_vec.T,1)
    mean_x = mean_group_size_membership(g_of_x_vec.T, x_max, p)
    T = out2.t
    return T, N1, N2, p, g_of_x_vec, mean_x
def add_arrow(line, start_ind = None,  direction='right', size=15, color=None):
    """
    add an arrow to a line.
    Edited from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot

    line:       Line2D object
    start_ind:   index of start of the arrow
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    #position=None,
    
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    #if position is None:
    #    position = xdata.mean()
    # find closest index
    if start_ind == None:
        position = xdata.mean()
        start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, 
                        mutation_scale=15)
    )
    


def plot_all(T,N1,N2,p,mean_x, xlim = [-10, 600]):
    '''
    plots N1, N2, p, and mean experienced group size
    versus scaled time

    @return: fig, ax
    '''
    fig, ax = plt.subplots(1,1)
    ax.plot(T,N2,'k', label = r'$N_2$')
    ax.plot(T,N1,'r', label = r'$N_1$')
    ax.plot(T,p,'b', label = r'$p$')
    if isinstance(mean_x, np.ndarray):
        ax.plot(T, mean_x, 'magenta', label = r'$\bar{x}$')
    format_ax(ax, xlab = Tlab,ylab = '',if_legend = True,
         xlim = xlim,fs_labs = 18)
    return fig, ax
def plot_x_vs_y(x,y,xlab,ylab,arrow_inds):
    '''
    plots x on horizontal axis, y on vertical axis
    adds arrows at x[arrow_inds]

    @returns: fig, ax
    '''
    fig, ax = plt.subplots(1,1)
    l = ax.plot(x,y,'k')
    for ind in arrow_inds:
        add_arrow(l[0], start_ind = ind)
        format_ax(ax, xlab = xlab, ylab = ylab, fs_labs = 18)
    return fig, ax
    
def plot_portion_x(fig, ax, out, x_max, xlim = [-1,500], ncol_legend = 1):
    '''
    plots time vs x*f(x)
    out is output from solve_ivp
    @inputs
    ax is the axis that is already made
    out is the output of solve_ivp

    @ returns: fig, ax
    '''
    T = out.t
    print(T[-1])
    g_of_x_vec = out.y[2:]

    xvec = np.arange(1,x_max+1,1)
    xg = xvec*g_of_x_vec.T

    p = np.sum(xg,1)
    
    # find F_of_x that are big enough
    portion_x = (xg.T/p).T

    portion_x[p<1e-10] = np.nan
    
    xlist = []
    xglist = []
    for x in range(1,x_max+1):
        portion_x_curr = portion_x[:,x-1]
        if max(portion_x_curr)>.1:
            xlist.append(x)
            xglist.append(portion_x_curr)
            
    labels = [f'x={x}' for x in xlist]
    for i, portion_x_curr in enumerate(xglist):
        ax.plot(T, portion_x_curr, label = labels[i], c = colors[i])
        
    format_ax(ax,Tlab,r'$xg(x)/p$', xlim = xlim, ylim=None,
              fs_labs = 20, fs_legend = 16, if_legend = True, ncol_legend = ncol_legend)
    return fig, ax

def print_param_caption(Tx, η1, η2, A1, A2, β1, β2, H1a, H1b, H2a, H2b,
                        α1_of_1, α2_of_1, 
                        s1, s2, α2_fun_type,**params):
    '''
    Prints parameter caption in latex format
    no return
    '''
    caption = 'The parameters are '
    caption += f'$\\eta_1 = {η1}, \\eta_2 = {η2}, '
    caption += f'A_ 1= {A1}, A_2= {A2}, \\beta_1 = {β1}, \\beta_2 = {β2}, '
    caption += 'H_{1a} =' +f'{H1a}, ' 'H_{1b} =' + f'{H1b}, '
    caption += 'H_{2a}' +f' = {H2a}, ' + 'H_{2b}' + f'{H2b}, T_x = {Tx}, ' 
    if α2_fun_type == 'constant':
        caption += f'\\alpha_1(1) = {α1_of_1}, s_1 ={s1}$, '
        caption += f'and $\\alpha_2(x) = {α2_of_1}$ is constant.'
    else:
        caption += f'\\alpha_1(1) = {α1_of_1}, \\alpha_2(1) = {α2_of_1}, '
        caption += f's_1 = {s1}$, and $s_2 = {s2}$' 
    
    print(caption)





def get_traj_plot_input(params, t_f = 1000, initial_points = None, 
                        num_init=4):
    '''
    initial_points is either none (so generates initial points) 
    or a list of up to 4 points, each of form [N1,N2, g(1), g(2), ..., g(xm)]
    '''
    if type(initial_points) != np.ndarray: # so it's None or some invalid entry
        print("generating initial points")
        initial_points = get_initial_points(num_init,**params)
    trajectories = []
    for i, init_state in enumerate(initial_points):
        #out2 = solve_ivp(grp.full_model, [0, t_f], init_state, 
        #                 method = "LSODA", args = (True, params))
        # results  = get_results(out2, x_max) # T, N1, N2, P, g_of_x_vec, mean_x
        results = bounded_ivp(init_state, params, t_f = t_f, if_dict=True)
        trajectories.append(results)
    return trajectories # each is a dictionary
def plot_with_arrow(ax, x,y,i, label, start_ind):
    """
    Plots a line on the given axis with optional arrows and a marker for the endpoint.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the line.
    x : array-like
        X-coordinates of the data points.
    y : array-like
        Y-coordinates of the data points.
    i : int
        Index for selecting the color from `colors_x`.
    label : str
        Label for the plotted line.
    start_ind : int or list of int
        Index (or indices) indicating where to place arrows along the line.

    Behavior:
    ---------
    - Plots the line with the specified color and label.
    - If the last two points are very close (within 1e-6 tolerance), marks the endpoint with an orange star.
    - Adds arrows at the specified indices to indicate directionality.

    Notes:
    ------
    - The function assumes `colors_x` is a predefined list of colors.
    - Calls `add_arrow()` to add arrows to the plot.

    """
    l = ax.plot(x,y,colors_x[i], label = label)
    line_zorder = l[0].get_zorder()
    if np.all(np.abs([x[-1] - x[-2], y[-1]-y[-2]])<1e-6):
        ax.scatter(x[-1],y[-1],c='orange', edgecolors = 'k',linewidths=.4,
                   marker = "*", s = 100, zorder=line_zorder+1)

    #plot arrows
    if type(start_ind) == int:
        start_ind = [start_ind]
    for elt in start_ind:
        add_arrow(l[0], start_ind = elt)   
        

def plot_trajectory(key_x, key_y, trajectories, xlab=None,
                    ylab=None, start_inds=[50, 50, 50, 50],
                    if_legend=False, g_x = None, g_y = None):
    '''
    Plots two state variables from a set of trajectories.

    Args:
        key_x (str): The key corresponding to the state variable to plot on the x-axis.
        key_y (str): The key corresponding to the state variable to plot on the y-axis.
        trajectories (list of dict): A list of dictionaries, where each dictionary represents a trajectory.
            Each dictionary should have the state variables as keys, and their values as lists or arrays.
        xlab (str, optional): Label for the x-axis. If not provided, uses `standard_labs[key_x]`.
        ylab (str, optional): Label for the y-axis. If not provided, uses `standard_labs[key_y]`.
        start_inds (list of int, optional): Indices indicating where to start plotting arrows for each trajectory.
            Default is [50, 50, 50, 50].
        if_legend (bool, optional): Whether to include a legend on the plot. Default is False.
        g_x (integer, optional): The group size density to plot on the x axis
        g_y (integer, optional): the group size density to plot on the y axis

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        ax (matplotlib.axes.Axes): The axes object for the plot.

    Behavior:
        - Iterates through the list of trajectories and plots the selected state variables.
        - Labels the axes using either the provided `xlab` and `ylab` or default labels from `standard_labs`.
        - Formats the axis using the `format_ax` function.
        - Optionally includes a legend if `if_legend` is set to True.

    Dependencies:
        - `standard_labs`: A dictionary mapping state variable keys to their corresponding axis labels.
        - `format_ax`: A custom function to format the axis.
        - `plot_with_arrow`: A custom function to plot trajectories with arrows starting at specified indices.
    '''
    fig, ax = plt.subplots(1,1)

    # set the label for the horizontal axis
    if key_x == 'g':
        if g_x == None:
            print('Need a group size for horizontal axis')
        xlab = gxlabel(g_x)
    else:
        xlab = standard_labs[key_x] if xlab == None else xlab
        print(xlab)
    
    # set the label for the vertical axis
    if key_y == 'g':
        if g_y == None:
            print('Need a group size for vertical axis')
        ylab = gxlabel(g_y)
    else:
        ylab = standard_labs[key_y] if ylab == None else ylab

            
    for i, traj in enumerate(trajectories):
        label = "Initial State %d"%i
        # set the variables for the horizontal (x_var) and vertical axis (y_var)
        x_var = traj['g'][g_x-1] if key_x == 'g' else traj[key_x]
        y_var = traj['g'][g_y-1] if key_y == 'g' else traj[key_y]
        # plot with arrows starting at start_inds
        plot_with_arrow(
            ax, x_var, y_var, i,
                            label, start_inds[i])
    format_ax(ax, xlab, ylab, if_legend = if_legend)
    return fig, ax
    
def plot_trajectory_vs_T(key_y, trajectories, 
                     ylab = None, if_legend = False, g_y = None):
    '''
    Plots a state variable versus scaled time from a set of trajectories.

    Args:
        key_y (str): The key corresponding to the state variable to plot on the y-axis. 
                    If a group density, uses g_y to determine which one
        trajectories (list of dict): A list of dictionaries, where each dictionary represents a trajectory.
            Each dictionary should have the state variables as keys, and their values as lists or arrays.
        ylab (str, optional): Label for the y-axis. If not provided, uses `standard_labs[key_y]`.
        start_inds (list of int, optional): Indices indicating where to start arrows for each trajectory.
            Default is [50, 50, 50, 50].
        if_legend (bool, optional): Whether to include a legend on the plot. Default is False.
        g_y (int, optional): group size of group density to plot
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        ax (matplotlib.axes.Axes): The axes object for the plot.

    Behavior:
        - Iterates through the list of trajectories and plots the selected state variable versus T.
        - Labels the axes using either the provided ``ylab` or default labels from `standard_labs`.
        - Formats the axis using the `format_ax` function.
        - Optionally includes a legend if `if_legend` is set to True.

    Dependencies:
        - 'colors_x': a list of colors ['k', 'b', 'r', 'm']
        - `standard_labs`: A dictionary mapping state variable keys to their corresponding axis labels.
        - `format_ax`: A custom function to format the axis.
    '''
    fig, ax = plt.subplots(1,1)
    # set the label for the vertical axis
    if key_y == 'g':
        if g_y == None:
            print('Need a group size for vertical axis')
        ylab = gxlabel(g_y)
    else:
        ylab = standard_labs[key_y] if ylab == None else ylab
        
    for i, traj in enumerate(trajectories):
        y_var = traj['g'][g_y-1] if key_y == 'g' else traj[key_y]
        label = "Initial State %d"%i
        plt.plot(
            traj['T'], y_var, colors_x[i], label = label
        )
    format_ax(ax, standard_labs['T'], ylab, if_legend = if_legend)
    return fig, ax

def make_traj_plots(params, t_f =1000, 
               grp_size1 = 2, grp_size2 = 3, start_inds = None,
              initial_points = None, num_init = 4,
                   if_legend = False):
    '''
    Plots trajectories from four initial plots, and plots on the trajectories four different projections: 
        1. N1 vs mean experienced group size
        2. N1 vs N2
        3. g(1) vs g(grp_size1)
        4. g(1) vs g(grp_size2)
        5. variance vs time
    initial points: list of initial points of form [N1, N2, g(1), g(2), ..., g(x_max)]
    Arguments
    - params: params dictionary
    - t_f: final time point for solve_ivp simulation
    - grp_size1: group size on y axis of ax_g2
    - grp_size2: group size for y axis of ax_g3
    - start_inds: start index for arrow on plots in ax1, axN, ax_g2, ax_g3

    Returns
    (tuple) fig1, figN, fig_g2, fig_g3, fig_var
    '''

    if start_inds == None:
        print('No start indices for arrows. Using 50.')
        row = [50,50,50,50]
        start_inds = [row,row,row,row]
    fig1, ax1 = plt.subplots(1,1) # N1 vs mean_x
    figN, axN = plt.subplots(1,1) # N1 vs N2
    fig_g2, ax_g2 = plt.subplots(1,1) #g(1) vs g(2)
    fig_g3, ax_g3 = plt.subplots(1,1) #g(1) vs g(3)
    fig_var, ax_var = plt.subplots(1,1) #variance vs T
    
    trajectories = get_traj_plot_input(params, t_f = t_f, 
                                       initial_points = initial_points,
                                       num_init = num_init)
    
    for i, traj in enumerate(trajectories):
        #T, N1, N2, P, g_of_x_vec, mean_x = traj
        if np.any(np.isnan(traj['mean_x'])):
            print("oh no! mean x is nan")
        elif  np.any(traj['mean_x']<0):
            print("oh no! mean x is negative")
            print("i=%d"%i)
        # check 
        label = "Initial State %d"%i
        plot_with_arrow(ax1, traj['N1'], traj['mean_x'],i,
                        label, start_inds[0][i])
        plot_with_arrow(axN, traj['N1'], traj['N2'], i, 
                        label, start_inds[1][i])
        #g(1) vs g(grp_size1)
        plot_with_arrow(ax_g2, traj['g'][0], traj['g'][grp_size1-1], i, 
                        label, start_inds[2][i])
        #g(1) vs g(grp_size2)
        plot_with_arrow(ax_g3, traj['g'][0], traj['g'][grp_size2 - 1], i, 
                        label, start_inds[3][i])
        # variance
        ax_var.plot(traj['T'],traj['var'], colors_x[i], label = label)
        #plot_with_arrow(ax_var, traj['T'],traj['var'], i, label

        #axN.plot(N1, N2, colors_x[i], label = label)
        #ax_g2.plot(g_of_x_vec[0], g_of_x_vec[1], colors_x[i], label = label)
        #ax_g3.plot(g_of_x_vec[0], g_of_x_vec[2], colors_x[i], label = label)

    format_ax(ax1, N1lab,mean_x_lab, if_legend = if_legend)
    format_ax(axN, N1lab,N2lab, if_legend = if_legend)
    format_ax(ax_g2, 'g(1)', 'g(%d)'%grp_size1, if_legend = if_legend)
    format_ax(ax_g3, 'g(1)', 'g(%d)'%grp_size2, if_legend = if_legend)

    return fig1, figN, fig_g2, fig_g3, fig_var
