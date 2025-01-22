import numpy as np
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.linalg import eigvals

def fun_Jac(N1,N2,gvec,**params):
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    size = len(gvec)+2


    # stuff used for multiple rows that speeds it up
    # the gradient of f(x) vs N1, N2 for x = 1, 2, ..., x_max. 
    # rows correspond to N1, N2. columns to x
    
    grad_f_1 = fun_grad_func_response(1,xvec,N1,N2,**params)
    grad_f_2 = fun_grad_func_response(2,xvec,N1,N2,**params)
    
    Jac = np.zeros((size,size))
    Jac[0,:] = fun_grad_big_prey(N1, N2, gvec, grad_f_1, **params)
    Jac[1,:] = fun_grad_small_prey(N1, N2, gvec, grad_f_2, **params)
    Jac[2:,:] = fun_Jac_groups(N1, N2, gvec, grad_f_1, grad_f_2, xvec, **params)

    return Jac
    
def fun_grad_func_response(i,x, N1,N2,H1,H2,A, **params):
    '''
    The gradient of the (scaled) functional response on prey i wrt N1, N2
    returns an array with 2 rows (N1, N2) and x_max columns
    '''
    alpha1 = fun_alpha1(x,**params)
    alpha2 = fun_alpha2(x,**params)
    denom = (1 + alpha1 * H1 * N1/x + alpha2 * H2 * N2/x)**2
    if i == 1:
        return A*np.array([ alpha1*(1 + alpha2 * H2 * N2/x), 
                         - alpha1 * alpha2 * H2 * N1/x])/denom
    elif i == 2:
        return A*np.array([ - alpha1 * alpha2 * H1 * N2/x,
                         alpha2 * (1 + alpha1 * H1 * N1/x)])/denom

def fun_grad_big_prey(N1,N2,gvec, grad_f_1, η1, x_max, **params):
    '''
    return gradient of big prey vs n1, n2, g(1), .. g(x)
    '''
    
    # the sum of  g(x) * [ del g_1/ del N_1, del g_1 / del N_2]
    grad_sum_g_y = np.sum(grad_f_1 * gvec,1)
    
    delU1_N1 = η1 * (1 - 2 * N1) - grad_sum_g_y[0] 
    delU1_N2 = - grad_sum_g_y[1]

    xvec = np.arange(1,x_max+1,1)
    f1_vec = fun_response_non_dim(xvec, N1, N2, 1,**params)
    delU1_g = - f1_vec

    to_return = np.array([delU1_N1, delU1_N2, *delU1_g])

    return to_return

def fun_grad_small_prey(N1,N2,gvec, grad_f_2, η2,  x_max, **params):
    '''
    return gradient of small prey vs n1, n2, g(1), .. g(x)
    '''

    # the sum of g(x) * [ del f_2/ del N_1, del f_2 / del N_2]
    grad_sum_g_y = np.sum(grad_f_2 * gvec,1)

    delU2_N1 = - grad_sum_g_y[0]
    delU2_N2 = η2 * (1 - 2 * N2) - grad_sum_g_y[1]

    xvec = np.arange(1,x_max+1,1)
    f2_vec = fun_response_non_dim(xvec,N1,N2,2,**params)
    delU2_g = - f2_vec

    to_return = np.array([delU2_N1, delU2_N2, *delU2_g])

    return to_return
def fun_Jac_groups(N1, N2, gvec, grad_f_1, grad_f_2, xvec, x_max, Tx, d,
                   η1, η2, **params):
    
    Jac = np.zeros((len(gvec),len(gvec)+2))

    # some of the calculations i'll need ready for the actual partial derivatives 
    # of the master equation of dg(x)/dt for x = 1,2, ... , x_max

    
    partial_π = params['β1'] * grad_f_1 + params['β2'] * grad_f_2 # p.d.v. of yield (pi) vs N1, N2 (rows)
                            # for x = 1, 2, ... ,x_max (columns)
    π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params) # yield from
    fitnessvec = π_vec/xvec

    # partial derivative of S(1,x) wrt N1, N2 for x = 1, 2, ...,x_max
    # form [[dS(1,2)/dN1, dS(1,2)/dN2], ..., [dS(1,x_max)/dN1, dS(1,x_max/dN2]]... so rows correspond to x
    partial_S_vec = [fun_partial_S_wrt_prey(N1, N2, x, fitnessvec, π_vec, partial_π, d,**params) \
                                 for x in range(2,x_max+1)]

    S_x_1_vec = [best_response_fun_given_fitness(x,1,fitnessvec,d) for x in range(2,x_max+1)]
    td = 1 - η1 - η2
    def g(x):
        return gvec[x-1]
    def partial_S(x):
        # grad S(1,x) wrt N1, N2
        return partial_S_vec[x-2]
    def S(x,y=1):
        # NEED TO CHECK
        if y == 1:
            return S_x_1_vec[x-2]
        if x == 1:
            return 1 - S_x_1_vec[y-2]
        #return best_response_fun_given_fitness(x,y,fitnessvec,d) # old way that recalculates each time
    def π(x):
        return π_vec[x-1]

    # first row. splitting into portion from group fission/fusion and pop processes
    Q1_Ni_group = (1/Tx) * (2*g(2) * partial_S(2) + \
                       np.sum(np.array([partial_S(x) * ( x * g(x) + g(1) * g(x-1)) \
                               for x in range(2,x_max+1)]),0))
    Q1_Ni_pop = g(x_max) * partial_π[:,-1] - g(1) * partial_π[:,0]
    Q1_Ni = Q1_Ni_group + Q1_Ni_pop
    

    Q1_g1 = (-2*g(1)*S(2) - sum([g(x-1)*S(x,1) \
                                       for x in range(3,x_max+1)]))/Tx - π(1) - td
    Q1_g2 = (4*S(1,2) - g(1)*S(3,1))/Tx + 2*td
    Q1_gx = [(x*S(1,x) - g(1)*S(x+1,1))/Tx for x in range(3,x_max)] #FILL IN
    Q1_gxmax = x_max*S(1,x_max)/Tx + π(x_max)
    Jac[0,:] = np.array([*Q1_Ni, Q1_g1, Q1_g2, *Q1_gx, Q1_gxmax])
    

    # second row
    Q2_Ni = (1/Tx) * ( - partial_S(2) * (2*g(2) + 0.5*(g(1))**2) \
                            + partial_S(3)*(3*g(3) + g(1)*g(2))) \
                        + g(1)*partial_π[:,0] - g(2)*partial_π[:,1]
    Q2_g = np.zeros(len(gvec))
    Q2_g[0] = (1/Tx)* (g(1) * S(2,1) - g(2) * S(3,1)) + π(1)# partial wrt g(1)
    Q2_g[1] = -(1/Tx) * (2* S(1,2) + g(1)*S(3,1)) - π(2) - 2*td
    Q2_g[2] = (3/Tx)*S(1,3) + 3 * td
    
    Jac[1,:] = np.array([*Q2_Ni, *Q2_g])
    

    # 3rd through 2nd to last row (for 2 < x < x_max)
    
    for x in range(3,x_max):
        Qx_Ni = (1/Tx) * (partial_S(x+1) *( (x+1)*g(x+1) + g(1)*g(x) ) \
                          - partial_S(x)*(x*g(x) + g(1)*g(x-1)))\
                    + g(x-1)*partial_π[:,x-2] - g(x) * partial_π[:,x-1]
        
        Qx_g = np.zeros(len(gvec))
        
        Qx_g[0] = (1/Tx)* (g(x-1)*S(x,1) - g(x)*S(x+1,1)) # wrt g(1)
        Qx_g[x-2] = (1/Tx) * g(1)*S(x,1) + π(x-1) # wrt g(x-1)
        Qx_g[x-1] = -(1/Tx) * (x*S(1,x) + g(1)*S(x+1,1)) - π(x) - x*td # wrt g(x)
        Qx_g[x] = (1/Tx)*(x+1)*S(1,x+1) + (x+1)*td # wrt g(x+1)

        Jac[x-1,:] = np.array([*Qx_Ni,*Qx_g])


    # last row, g(x_max)
    Qxmax_Ni = -(1/Tx)*partial_S(x_max) * (x_max*g(x_max) + g(1)*g(x_max-1)) \
                + g(x_max-1)*partial_π[:,x_max-2] 
    Qxmax_g = np.zeros(len(gvec))
    Qxmax_g[0] = (1/Tx)*g(x_max-1)*S(x_max,1)
    Qxmax_g[x_max-2] = (1/Tx)*g(1)*S(x_max,1) + π(x_max-1) # wrt g(x-1)
    Qxmax_g[x_max-1] = -(1/Tx)*x_max*S(1,x_max) - x_max*td
    Jac[-1,:] = np.array([*Qxmax_Ni, *Qxmax_g])
                                           
    
    return Jac
    
def fun_partial_S_wrt_prey(N1, N2, x, π_vec, fitnessvec, partial_π,
                           d, **params):
    '''
    partial derivative of S(1,x) wrt N1, N2 for a specified x
    array with 2 entries corresponding to N1, N2
    '''
    
    S_1_x = best_response_fun_given_fitness(1,x,fitnessvec,d)
    S_x_1 = 1 - S_1_x

    partial_π_of_1 = partial_π[:,0] # [ \partial π(1) / \partial N_1, \partial π(1) / \partial N_2]
    partial_π_of_x = partial_π[:,x-1] # [ \partial π(x) / \partial N_1, \partial π(x) / \partial N_2]
    π_of_1 = π_vec[0] # π(1)
    π_of_x = π_vec[x-1] # π(x)

    partial_S_1_x = d*S_1_x * S_x_1 * ( partial_π_of_1 * 1 / π_vec[0] \
                                   - partial_π_of_x * 1 / π_of_x)
    
    return partial_S_1_x
    
def fun_Jac_groups_nopop(N1, N2, gvec, x_max, Tx, d, **params):
    '''
    Finds the Jacobian for group dynamics with no populatin dynamics
    '''
    
    Jac = np.zeros((len(gvec),len(gvec)+2))
    
    def g(x):
        return gvec[x-1]
    def S(x,y=1):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)

        
    # first row    
    Q1_g1 = (-2*g(1)*S(2,1) - sum([g(x)*S(x+1,1) \
                                       for x in range(2,x_max)]))/Tx 
    Q1_g2 = (4*(1-S(2)) - g(1)*S(3))/Tx 
    Q1_gx = [(x*(1-S(x)) - g(1)*S(x+1))/Tx for x in range(3,x_max)] #FILL IN
    Q1_gxmax = x_max*(1 - S(x_max))/Tx 
    Jac[0,:] = np.array([Q1_g1, Q1_g2, *Q1_gx, Q1_gxmax])

    # second row
    Q2_g = np.zeros(len(gvec))
    Q2_g[0] = (1/Tx)* (g(1) * S(2) - g(2) * S(3)) # partial wrt g(1)
    Q2_g[1] = -(1/Tx) * (2* (1 - S(2)) + g(1)*S(3)) 
    Q2_g[2] = (3/Tx)*(1 - S(3)) 
    
    Jac[1,:] = Q2_g

    # 3rd through 2nd to last row (for 2 < x < x_max)
    
    for x in range(3,x_max):
        
        Qx_g = np.zeros(len(gvec))
        
        Qx_g[0] = (1/Tx)* (g(x-1)*S(x) - g(x)*S(x+1)) # wrt g(1)
        Qx_g[x-2] = (1/Tx) * g(1)*S(x) # wrt g(x-1)
        Qx_g[x-1] = -(1/Tx) * (x*(1 - S(x)) + g(1)*S(x+1)) # wrt g(x)
        Qx_g[x] = (1/Tx)*(x+1)*(1-S(x+1)) # wrt g(x+1)

        Jac[x-1,:] = Qx_g



    Qxmax_g = np.zeros(len(gvec))
    Qxmax_g[0] = (1/Tx)*g(x_max-1)*S(x_max,1)
    Qxmax_g[x_max-2] = (1/Tx)*g(1)*S(x_max,1)  # wrt g(x-1)
    Qxmax_g[x_max-1] = -(1/Tx)*x_max*S(1,x_max) 
    Jac[-1,:] = Qxmax_g
                                           
    
    return Jac
        



def classify_stability(J):
    '''
    Compute the eigenvalues of the Jacobian matrix
    returns "Stable (attractive)", "Unstable", "Marginally stable (needs further analysis)",
    or "Indeterminate stability (needs further analysis)"
    '''
    # Compute the eigenvalues of the Jacobian matrix
    eigenvalues = np.linalg.eigvals(J)
    
    # Check the real parts of the eigenvalues
    real_parts = np.real(eigenvalues)
    
    # Classify the stability based on the real parts of the eigenvalues
    if np.all(real_parts < 0):
        return "Stable (attractive)"
    elif np.any(real_parts > 0):
        return "Unstable"
    elif np.all(real_parts <= 0):
        return "Marginally stable (needs further analysis)"
    else:
        return "Indeterminate stability (needs further analysis)"

def classify_equilibrium(equilibrium, params):
    '''
    equilibrium = [N1, N2, *gvec]
    @ returns: stability (string)

    Note: stability is one of:
    "Stable (attractive)", "Unstable", 
    "Marginally stable (needs further analysis)",
    or "Indeterminate stability (needs further analysis)"
    '''
    [N1,N2,*gvec] = equilibrium
    J = fun_Jac(N1,N2,gvec,**params) 
    if not np.isfinite(J).all():
        print(J)
        print(equilibrium)
        print(params)
    stability = classify_stability(J)
    
    return stability
# def fun_jacobian_one_grp(P, N1, N2, x, η1, η2, A1, β1, β2, **params):
#     H1 = params['H1']
#     H2=params['H2']
#     Y1 = fun_Y1(N1,N2,x,**params)
#     Y2 = fun_Y2(N1,N2,x,**params)
#     α2 = fun_alpha2(x, **params)
#     α1 = fun_alpha1(x, **params)
#     td = 1 - η1 - η2
#     A2 = 1 - A1
    
#     J = np.zeros((3, 3))

#     J[0, 0] = -td + (1/x) * (β1 * Y1 + β2 * Y2)
#     J[0, 1] = P / x * β1 * (α1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
#     J[0, 2] = P / x * β2 * (α2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
    
#     J[1, 0] = -A1 * Y1 / x
#     J[1, 1] = η1 * (1 - 2 * N1) - P / x * A1 * (α1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
#     J[1, 2] = -P / x * A1 * (α1 * α2 * H2 * N2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
    
#     J[2, 0] = -A2 * Y2 / x
#     J[2, 1] = -P / x * A2 * (α2 * α1 * H1 * N1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
#     J[2, 2] = η2 * (1 - 2 * N2) - P / x * A2 * (α2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
#     return J

