import numpy as np
import matplotlib.pyplot as plt
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.optimize import fsolve

colors = ['k','r','b','cyan', 'magenta','orange',
         'gray', 'green']
markers = ["o","","v", ""]

def format_ax(ax,xlab,ylab, xlim = None, ylim=None,
              fs_labs = 20, fs_legend = 16, if_legend = False):
    ax.set_xlabel(xlab, fontsize = fs_labs)
    ax.set_ylabel(ylab, fontsize = fs_labs)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    if if_legend:
        ax.legend(fontsize = fs_legend)
        
def get_results(out,x_max):
    '''
    gets results from output of simulation
    out: dictionary that's output of solve_ivp
    x_max: max group size

    @returns:
    T, N1, N2, P, mean_x
    '''
    P, N1, N2 = out2.y[0:3]
    F_of_x_vec = out2.y[3:]
    mean_x = mean_group_size_membership(F_of_x_vec.T, x_max, P)
    T = out2.t
    return T, N1, N2, P, mean_x
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
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
    
Plab = r'$P$, Scaled Pred. Pop Size'
N1lab = r'$N_1$, Scaled Big Prey Pop Size'
N2lab = r'$N_2$, Scaled Small Prey Pop Size'
Tlab = r'$T$, Scaled time'
mean_x_lab = "Mean Group Size\n Membership"

def plot_all(T,N1,N2,P,mean_x, xlim = [-10, 600]):
    fig, ax = plt.subplots(1,1)
    ax.plot(T,N2,'k', label = r'$N_2$')
    ax.plot(T,N1,'r', label = r'$N_1$')
    ax.plot(T,P,'b', label = r'$P$')
    if isinstance(mean_x, np.ndarray):
        ax.plot(T, mean_x, 'magenta', label = r'$\bar{x}$')
    format_ax(ax, xlab = Tlab,ylab = '',if_legend = True,
         xlim = xlim,fs_labs = 18)
    return fig, ax
def plot_x_vs_y(x,y,xlab,ylab,arrow_inds):
    fig, ax = plt.subplots(1,1)
    l = ax.plot(x,y,'k')
    for ind in arrow_inds:
        add_arrow(l[0], start_ind = ind)
        format_ax(ax, xlab = xlab, ylab = ylab, fs_labs = 18)
    return fig, ax
    
def plot_portion_x(ax, out, x_max, xlim = [-1,500]):
    '''
    plots time vs x*F(x)
    @inputs
    ax is the axis that is already made
    out is the output of solve_ivp

    @ returns: fig, ax
    '''
    T = out.t
    print(T[-1])
    F_of_x_vec = out.y[3:]
    P = out.y[0]
    # find F_of_x that are big enough
    xvec = np.arange(1,x_max+1,1)
    xF = xvec*F_of_x_vec.T
    portion_x = (xF.T/P).T
    xlist = []
    xflist = []
    for x in range(1,11):
        portion_x_curr = portion_x[:,x-1]
        if max(portion_x_curr)>.1:
            xlist.append(x)
            xflist.append(portion_x_curr)
            
    labels = [f'x={x}' for x in xlist]
    for i, portion_x_curr in enumerate(xflist):
        ax.plot(T, portion_x_curr, label = labels[i], c = colors[i])
        
    format_ax(ax,Tlab,r'$xF(x)/P$', xlim = xlim, ylim=None,
              fs_labs = 20, fs_legend = 16, if_legend = True)
    return fig, ax

def print_param_caption(ξ, Tx, η1, η2, A1, β1, β2, H1, H2, α1_of_1, α2_of_1, 
                        s1, s2, α2_fun_type,**params):
    caption = 'The parameters are '
    caption += f'$\\xi = {ξ}, \\eta_1 = {η1}, \\eta_2 = {η2}, '
    caption += f'A_1 = {A1}, \\beta_1 = {β1}, \\beta_2 = {β2}, '
    caption += f'H_1 = {H1}, H_2 = {H2}, T_x = {Tx}, ' 
    if α2_fun_type == 'constant':
        caption += f'\\alpha_1(1) = {α1_of_1}, s_1 ={s1}$, '
        caption += f'and $\\alpha_2(x) = {α2_of_1}$ is constant.'
    else:
        caption += f'\\alpha_1(1) = {α1_of_1}, \\alpha_2(1) = {α2_of_1}, '
        caption += f's_1 = {s1}$, and $s_2 = {s2}$' 
    
    print(caption)

def get_equilibrium(params, N1_0 = 0.5, N2_0 = 0.5, P_0 = 5, F_of_x_vec = None):
    '''
    finds the equilibrium using Fsolve
    if not given F_of_x_vec, then just has everyone initially solitary
    
    @returns:
    N1_eq, N2_eq, F_eq, P_eq, mean_x_eq
    '''
    x_max = params['x_max']
    if ~isinstance(F_of_x_vec, np.ndarray):
        F_of_x_vec = np.zeros(x_max)
        F_of_x_vec[0] = P_0
    x0 = [N1_0, N2_0, *F_of_x_vec]
    out, infodict, ier, mesg = fsolve(func = full_model_no_P, x0 = x0, 
                                  args = (params), full_output = 1)
    F_eq = out[2:]
    P_eq = np.sum(np.arange(1,x_max+1,1)*F_eq); 
    N1_eq = out[0]
    N2_eq = out[1]
    mean_x_eq = mean_group_size_membership(F_eq,10,P_eq)

    return N1_eq, N2_eq, F_eq, P_eq, mean_x_eq
    