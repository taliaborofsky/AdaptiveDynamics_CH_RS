import numpy as np
import matplotlib.pyplot as plt
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.integrate import solve_ivp
from scipy.optimize import root
from local_stability_funs import *
from equilibria_funs import *

#colors = ['k','r','b','cyan', 'magenta','orange',
#         'gray', 'green']
colors = ['r', 'orange', 'magenta', 'purple', 'blue', 
          'cornflowerblue', 'turquoise','k', 'gray']
markers = ["o","","v", ""]
Plab = r'$p$,  Pred. Pop Density'
Pscaledlab = r'$P$,  Pred. Scaled Density'
N1lab = r'$N_1$, Scaled Big Prey'+ '\nDensity'
N2lab = r'$N_2$, Scaled Small Prey' + '\nDensity'
Tlab = r'$T$, Scaled time'
mean_x_lab = "Mean Experienced\nGroup Size"
freq_x_lab = r'Freq$(x)$'
β1lab = r'$\beta_1$'
gxlab = r'g$(x)$'
figure_ops = dict(bbox_inches = 'tight', dpi = 600)

def format_ax(ax,xlab,ylab, xlim = None, ylim=None,
              fs_labs = 20, fs_legend = 16, if_legend = False,
             ncol_legend = 1):
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
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
    


def plot_all(T,N1,N2,p,mean_x, xlim = [-10, 600]):
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

def print_param_caption(Tx, η1, η2, A, β1, β2, H1, H2, α1_of_1, α2_of_1, 
                        s1, s2, α2_fun_type,**params):
    caption = 'The parameters are '
    caption += f'$\\eta_1 = {η1}, \\eta_2 = {η2}, '
    caption += f'A = {A}, \\beta_1 = {β1}, \\beta_2 = {β2}, '
    caption += f'H_1 = {H1}, H_2 = {H2}, T_x = {Tx}, ' 
    if α2_fun_type == 'constant':
        caption += f'\\alpha_1(1) = {α1_of_1}, s_1 ={s1}$, '
        caption += f'and $\\alpha_2(x) = {α2_of_1}$ is constant.'
    else:
        caption += f'\\alpha_1(1) = {α1_of_1}, \\alpha_2(1) = {α2_of_1}, '
        caption += f's_1 = {s1}$, and $s_2 = {s2}$' 
    
    print(caption)




def plot_F_equilibrium(paramvec, gxvecs, xvec, xlab, ylab, 
                       ncol_legend = 1, xlim = None, ylim = None,
                       fig = None, ax = None):
    '''
    Plots distribution g(x)
    can take for gxvecs either g(x) or \bar{g}(x), the frequency of predators in groups of size x
    '''
    if ax == None:
        fig, ax = plt.subplots(1,1)
        
    
    
    colors_x = ['r', 'orange', 'magenta', 'purple', 'blue', 'cornflowerblue', 'k']

    xmax = len(xvec)
    if gxvecs.shape[1] == xmax:
        gxvecs = gxvecs.T
        
    for x in xvec:
        if np.any(gxvecs[x-1]>1e-2):
            ax.plot(paramvec, gxvecs[x-1], colors_x[x-1], label = r'$x=$%d'%x)
    format_ax(ax,xlab,ylab, fs_labs = 18, fs_legend = 16, if_legend = True,
             ncol_legend = ncol_legend)
    return fig, ax
    





'''
def generate_params_using_weitz(A1, β2, H2, η2, weight_fraction_prey, 
                                α1_of_1 = 0.05, α2_of_1 = 0.95, s1 = 2, 
                                s2 = 2, α2_fun_type = 'sigmoid', x_max = 10, 
                                d = 10, Tx = .01):
    attack_fraction = 1/(1/A1 - 1)
    β1 = β2 * attack_fraction * weight_fraction_prey**(0.25)
    H1 = H2 * attack_fraction * weight_fraction_prey**(0.25)
    η1 = η2 * weight_fraction_prey**(-0.25)
    params = dict(η1 = η1, η2 = η2, A1 = A1, β1 = β1, β2 = β2, 
                   H1=H1, H2=H2, 
                  α1_of_1=α1_of_1, α2_of_1=α2_of_1, 
                  s1=s1, s2=s2, α2_fun_type = α2_fun_type,
                  x_max = x_max, d = d,
                 Tx = Tx, r = 0, γ = 0, pop_process = True)
    return params
'''
    
def plot_freq_x_eq(paramvec, gxvecs, xvec, Pvec, xlab, ylab = r'Freq$(x)$', 
                       ncol_legend = 1, xlim = None, ylim = None,
                       fig = None, ax = None):
    prob_x = (xvec*gxvecs).T/Pvec
    fig, ax = plot_F_equilibrium(paramvec, prob_x, xvec, xlab, ylab, 
                       ncol_legend = ncol_legend, xlim = None, ylim = None,
                       fig = None, ax = None)
    return fig, ax


def plot_W_mode_comparison(xvec,N1vec,N2vec,gxvecs, params, fig = None, ax = None):
    '''
    Plots W(x) for the mode of x, and for x=1, and the mode of x + 1
    '''
    if fig == None:
        fig,ax = plt.subplots(1,1)
    x_mode = np.argmax(xvec*gxvecs,1)+1
    W_of_mode_x_plus_1 = per_capita_fitness_from_prey_non_dim(x_mode + 1, N1vec, N2vec, **params)
    W_of_mode_x = per_capita_fitness_from_prey_non_dim(x_mode, N1vec, N2vec, **params)
    W_of_1 = per_capita_fitness_from_prey_non_dim(1, N1vec, N2vec, **params)
    ax.plot(β1vec, W_of_1, 'crimson', label = r'solitary')
    ax.plot(β1vec, W_of_mode_x, 'magenta', label = r'mode$(x)$')
    ax.plot(β1vec, W_of_mode_x_plus_1 - W_of_1, 'purple', label = r'mode$(x+1)$')

    format_ax(ax, β1lab, 'Per Capita Fitness', if_legend = True)
    return fig, ax
