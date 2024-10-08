import numpy as np
from fitness_funs_non_dim import *
from group_w_pop_funs import *
from scipy.linalg import eigvals

def fun_Jac(N1,N2,fvec,**params):
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    size = len(fvec)+2


    # stuff used for multiple rows that speeds it up
    grad_Y_1 = fun_grad_func_response(1,xvec,N1,N2,**params)
    grad_Y_2 = fun_grad_func_response(2,xvec,N1,N2,**params)
    
    Jac = np.zeros((size,size))
    Jac[0,:] = fun_grad_big_prey(N1, N2, fvec, grad_Y_1, **params)
    Jac[1,:] = fun_grad_small_prey(N1, N2, fvec, grad_Y_2, **params)
    Jac[2:,:] = fun_Jac_groups(N1, N2, fvec, grad_Y_1, grad_Y_2, xvec, **params)

    return Jac

def fun_Jac_groups(N1, N2, fvec, grad_Y_1, grad_Y_2, xvec, x_max, Tx, d,
                   η1, η2, **params):
    
    Jac = np.zeros((len(fvec),len(fvec)+2))
    
    partial_π = params['β1'] * grad_Y_1 + params['β2'] * grad_Y_2
    π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params)
    fitnessvec = π_vec/xvec
    partial_S_vec = [fun_partial_S_wrt_prey(N1, N2, x, partial_π, x_max,d,**params) \
                                 for x in range(2,x_max+1)]
    td = 1 - η1 - η2
    def f(x):
        return fvec[x-1]
    def partial_S(x):
        return partial_S_vec[x-2]
    def S(x,y=1):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def π(x):
        return π_vec[x-1]
        
    # first row
    Q1_Ni_group = (1/Tx) * (2*f(2) * partial_S(2) + \
                       np.sum(np.array([partial_S(x) * ( x * f(x) + f(1) * f(x-1)) \
                               for x in range(2,x_max+1)]),0))
    Q1_Ni_pop = f(x_max) * partial_π[:,-1] - f(1) * partial_π[:,0]
    Q1_Ni = Q1_Ni_group + Q1_Ni_pop
    
    Q1_f1 = (-2*f(1)*S(2,1) - sum([f(x)*S(x+1,1) \
                                       for x in range(2,x_max)]))/Tx - π(1) - td
    Q1_f2 = (4*(1-S(2)) - f(1)*S(3))/Tx + 2*td
    Q1_fx = [(x*(1-S(x)) - f(1)*S(x+1))/Tx for x in range(3,x_max)] #FILL IN
    Q1_fxmax = x_max*(1 - S(x_max))/Tx + π(x_max)
    Jac[0,:] = np.array([*Q1_Ni, Q1_f1, Q1_f2, *Q1_fx, Q1_fxmax])

    # second row
    Q2_Ni = (1/Tx) * ( - partial_S(2) * (2*f(2) + 0.5*(f(1))**2) \
                            + partial_S(3)*(3*f(3) + f(1)*f(2))) \
                        + f(1)*partial_π[:,0] - f(2)*partial_π[:,1]
    Q2_f = np.zeros(len(fvec))
    Q2_f[0] = (1/Tx)* (f(1) * S(2) - f(2) * S(3)) + π(1)# partial wrt f(1)
    Q2_f[1] = -(1/Tx) * (2* (1 - S(2)) + f(1)*S(3)) - π(2) - 2*td
    Q2_f[2] = (3/Tx)*(1 - S(3)) + 3 * td
    
    Jac[1,:] = np.array([*Q2_Ni, *Q2_f])

    # 3rd through 2nd to last row (for 2 < x < x_max)
    
    for x in range(3,x_max):
        Qx_Ni = (1/Tx) * (partial_S(x+1) *( (x+1)*f(x+1) + f(1)*f(x) ) \
                          - partial_S(x)*(x*f(x) + f(1)*f(x-1)))\
                    + f(x-1)*partial_π[:,x-2] - f(x) * partial_π[:,x-1]
        
        Qx_f = np.zeros(len(fvec))
        
        Qx_f[0] = (1/Tx)* (f(x-1)*S(x) - f(x)*S(x+1)) # wrt f(1)
        Qx_f[x-2] = (1/Tx) * f(1)*S(x) + π(x-1) # wrt f(x-1)
        Qx_f[x-1] = -(1/Tx) * (x*(1 - S(x)) + f(1)*S(x+1)) - π(x) - x*td # wrt f(x)
        Qx_f[x] = (1/Tx)*(x+1)*(1-S(x+1)) + (x+1)*td # wrt f(x+1)

        Jac[x-1,:] = np.array([*Qx_Ni,*Qx_f])


    # last row, f(x_max)
    Qxmax_Ni = -(1/Tx)*partial_S(x_max) * (x_max*f(x_max) + f(1)*f(x_max-1)) \
                + f(x_max-1)*partial_π[:,x_max-2] - f(x_max) * partial_π[:,x_max-1]
    Qxmax_f = np.zeros(len(fvec))
    Qxmax_f[0] = (1/Tx)*f(x_max-1)*S(x_max,1)
    Qxmax_f[x_max-2] = (1/Tx)*f(1)*S(x_max,1) + π(x_max-1) # wrt f(x-1)
    Qxmax_f[x_max-1] = -(1/Tx)*x_max*S(1,x_max) - x_max*td
    Jac[-1,:] = np.array([*Qxmax_Ni, *Qxmax_f])
                                           
    
    return Jac

def fun_Jac_groups_nopop(N1, N2, fvec, x_max, Tx, d, **params):
    '''
    Finds the Jacobian for group dynamics with no populatin dynamics
    '''
    
    Jac = np.zeros((len(fvec),len(fvec)+2))
    
    def f(x):
        return fvec[x-1]
    def S(x,y=1):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)

        
    # first row    
    Q1_f1 = (-2*f(1)*S(2,1) - sum([f(x)*S(x+1,1) \
                                       for x in range(2,x_max)]))/Tx 
    Q1_f2 = (4*(1-S(2)) - f(1)*S(3))/Tx 
    Q1_fx = [(x*(1-S(x)) - f(1)*S(x+1))/Tx for x in range(3,x_max)] #FILL IN
    Q1_fxmax = x_max*(1 - S(x_max))/Tx 
    Jac[0,:] = np.array([Q1_f1, Q1_f2, *Q1_fx, Q1_fxmax])

    # second row
    Q2_f = np.zeros(len(fvec))
    Q2_f[0] = (1/Tx)* (f(1) * S(2) - f(2) * S(3)) # partial wrt f(1)
    Q2_f[1] = -(1/Tx) * (2* (1 - S(2)) + f(1)*S(3)) 
    Q2_f[2] = (3/Tx)*(1 - S(3)) 
    
    Jac[1,:] = Q2_f

    # 3rd through 2nd to last row (for 2 < x < x_max)
    
    for x in range(3,x_max):
        
        Qx_f = np.zeros(len(fvec))
        
        Qx_f[0] = (1/Tx)* (f(x-1)*S(x) - f(x)*S(x+1)) # wrt f(1)
        Qx_f[x-2] = (1/Tx) * f(1)*S(x) # wrt f(x-1)
        Qx_f[x-1] = -(1/Tx) * (x*(1 - S(x)) + f(1)*S(x+1)) # wrt f(x)
        Qx_f[x] = (1/Tx)*(x+1)*(1-S(x+1)) # wrt f(x+1)

        Jac[x-1,:] = Qx_f



    Qxmax_f = np.zeros(len(fvec))
    Qxmax_f[0] = (1/Tx)*f(x_max-1)*S(x_max,1)
    Qxmax_f[x_max-2] = (1/Tx)*f(1)*S(x_max,1)  # wrt f(x-1)
    Qxmax_f[x_max-1] = -(1/Tx)*x_max*S(1,x_max) 
    Jac[-1,:] = Qxmax_f
                                           
    
    return Jac
        
def fun_grad_func_response(i,x, N1,N2,H1,H2,**params):
    '''
    The gradient of the (scaled) functional response on prey i wrt N1, N2
    returns an array with 2 rows (N1, N2) and x_max columns
    '''
    alpha1 = fun_alpha1(x,**params)
    alpha2 = fun_alpha2(x,**params)
    denom = (1 + alpha1 * H1 * N1 + alpha2 * H2 * N2)**2
    if i == 1:
        return np.array([ alpha1*(1 + alpha2 * H2 * N2), 
                         - alpha1 * alpha2 * H2 * N1])/denom
    elif i == 2:
        return np.array([ - alpha1 * alpha2 * H1 * N2,
                         alpha2 * (1 + alpha1 * H1 * N1)])/denom

def fun_grad_big_prey(N1,N2,fvec, grad_Y_1, η1, A1, x_max, **params):
    '''
    return gradient of big prey vs n1, n2, f(1), .. f(x)
    '''
    
    # the sum of A_1 * f(x) * [ del Y_1/ del N_1, del Y_1 / del N_2]
    grad_sum_f_y = A1 * np.sum(grad_Y_1 * fvec,1)
    
    delU1_N1 = η1 * (1 - 2 *N1) - grad_sum_f_y[0] 
    delU1_N2 = - grad_sum_f_y[1]

    xvec = np.arange(1,x_max+1,1)
    Y1_vec = fun_response_non_dim(xvec, N1, N2, 1,**params)
    delU1_f = -A1 * Y1_vec

    to_return = np.array([delU1_N1, delU1_N2, *delU1_f])

    return to_return

def fun_grad_small_prey(N1,N2,fvec, grad_Y_2, η2, A1, x_max, **params):
    '''
    return gradient of small prey vs n1, n2, f(1), .. f(x)
    '''
    A2 = 1 - A1

    # the sum of A_2 * f(x) * [ del Y_2/ del N_1, del Y_2 / del N_2]
    grad_sum_f_y = A2 * np.sum(grad_Y_2 * fvec,1)

    delU2_N1 = - grad_sum_f_y[0]
    delU2_N2 = η2 * (1 - 2 * N2) - grad_sum_f_y[1]

    xvec = np.arange(1,x_max+1,1)
    Y2_vec = fun_response_non_dim(xvec,N1,N2,2,**params)
    delU2_f = -A2 * Y2_vec

    to_return = np.array([delU2_N1, delU2_N2, *delU2_f])

    return to_return
def fun_partial_S_wrt_prey(N1, N2, x, partial_π,x_max,
                           d, **params):
    '''

    array with 2 entries corresponding to N1, N2
    '''

    xvec = np.arange(1, x_max+1,1)
    π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params)
    fitnessvec = π_vec/xvec
    
    
    S_1_x = best_response_fun_given_fitness(1,x,fitnessvec,d)
    S_x_1 = 1 - S_1_x

    partial_S_1_x = d*S_1_x * S_x_1 * ( partial_π[:,0] * 1 / π_vec[0] \
                                   - partial_π[:,x-1] * 1 / π_vec[x-1])
    
    return partial_S_1_x

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

def fun_jacobian_one_grp(P, N1, N2, x, η1, η2, A1, β1, β2, **params):
    H1 = params['H1']
    H2=params['H2']
    Y1 = fun_Y1(N1,N2,x,**params)
    Y2 = fun_Y2(N1,N2,x,**params)
    α2 = fun_alpha2(x, **params)
    α1 = fun_alpha1(x, **params)
    td = 1 - η1 - η2
    A2 = 1 - A1
    
    J = np.zeros((3, 3))

    J[0, 0] = -td + (1/x) * (β1 * Y1 + β2 * Y2)
    J[0, 1] = P / x * β1 * (α1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    J[0, 2] = P / x * β2 * (α2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
    
    J[1, 0] = -A1 * Y1 / x
    J[1, 1] = η1 * (1 - 2 * N1) - P / x * A1 * (α1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    J[1, 2] = -P / x * A1 * (α1 * α2 * H2 * N2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
    
    J[2, 0] = -A2 * Y2 / x
    J[2, 1] = -P / x * A2 * (α2 * α1 * H1 * N1 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    J[2, 2] = η2 * (1 - 2 * N2) - P / x * A2 * (α2 / (1 + H1 * α1 * N1 + H2 * α2 * N2)**2)
    
    return J

