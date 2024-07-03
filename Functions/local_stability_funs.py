import numpy as np
from fitness_funs_non_dim import *
from group_w_pop_funs import *

def fun_Jac(N1,N2,Fvec,**params):
    x_max = params['x_max']
    xvec = np.arange(1,x_max+1,1)
    size = len(Fvec)+2


    # stuff used for multiple rows that speeds it up
    grad_Y_1 = fun_grad_func_response(1,xvec,N1,N2,**params)
    grad_Y_2 = fun_grad_func_response(2,xvec,N1,N2,**params)
    
    Jac = np.zeros((size,size))
    Jac[0,:] = fun_grad_big_prey(N1, N2, Fvec, grad_Y_1, **params)
    Jac[1,:] = fun_grad_small_prey(N1, N2, Fvec, grad_Y_2, **params)
    Jac[2:,:] = fun_Jac_groups(N1, N2, Fvec, grad_Y_1, grad_Y_2, xvec, **params)

    return Jac

def fun_Jac_groups(N1, N2, Fvec, grad_Y_1, grad_Y_2, xvec, x_max, Tx, ξ,d,
                   η1, η2, **params):
    
    Jac = np.zeros((len(Fvec),len(Fvec)+2))

    partial_π = params['β1'] * grad_Y_1 + params['β2'] * grad_Y_2
    π_vec = yield_from_prey_non_dim(xvec, N1, N2, **params)
    fitnessvec = π_vec/xvec
    partial_S_vec = [fun_partial_S_wrt_prey(N1, N2, x, partial_π, x_max,d,**params) \
                                 for x in range(2,x_max+1)]
    td = 1 - η1 - η2
    def F(x):
        return Fvec[x-1]
    def partial_S(x):
        return partial_S_vec[x-2]
    def S(x,y=1):
        return best_response_fun_given_fitness(x,y,fitnessvec,d)
    def π(x):
        return π_vec[x-1]
        
    # first row
    Q1_Ni_group = (1/Tx) * (2*F(2) * partial_S(2) + \
                       np.sum(np.array([partial_S(x) * ( x * F(x) + ξ * F(1) * F(x-1)) \
                               for x in range(2,x_max+1)]),0))
    Q1_Ni_pop = F(x_max) * partial_π[:,-1] - F(1) * partial_π[:,0]
    Q1_Ni = Q1_Ni_group + Q1_Ni_pop
    
    Q1_F1 = (-2*ξ*F(1)*S(2,1) - sum([F(x)*ξ*S(x+1,1) \
                                       for x in range(2,x_max)]))/Tx - π(1) - td
    Q1_F2 = (4*(1-S(2)) - ξ*F(1)*S(3))/Tx + 2*td
    Q1_Fx = [(x*(1-S(x)) - ξ*F(1)*S(x+1))/Tx for x in range(3,x_max)] #FILL IN
    Q1_Fxmax = x_max*(1 - S(x_max))/Tx + π(x_max)
    Jac[0,:] = np.array([*Q1_Ni, Q1_F1, Q1_F2, *Q1_Fx, Q1_Fxmax])

    # second row
    Q2_Ni = (1/Tx) * ( - partial_S(2) * (2*F(2) + 0.5*ξ*(F(1))**2) \
                            + partial_S(3)*(3*F(3) + ξ*F(1)*F(2))) \
                        + F(1)*partial_π[:,0] - F(2)*partial_π[:,1]
    Q2_F = np.zeros(len(Fvec))
    Q2_F[0] = (ξ/Tx)* (F(1) * S(2) - F(2) * S(3)) + π(1)# partial wrt F(1)
    Q2_F[1] = -(1/Tx) * (2* (1 - S(2)) + ξ*F(1)*S(3)) - π(2) - 2*td
    Q2_F[2] = (3/Tx)*(1 - S(3)) + 3 * td
    
    Jac[1,:] = np.array([*Q2_Ni, *Q2_F])

    # 3rd through 2nd to last row (for 2 < x < x_max)
    
    for x in range(3,x_max):
        Qx_Ni = (1/Tx) * (partial_S(x+1) *( (x+1)*F(x+1) + ξ*F(1)*F(x) ) \
                          - partial_S(x)*(x*F(x) + ξ*F(1)*F(x-1)))\
                    + F(x-1)*partial_π[:,x-2] - F(x) * partial_π[:,x-1]
        
        Qx_F = np.zeros(len(Fvec))
        
        Qx_F[0] = (ξ/Tx)* (F(x-1)*S(x) - F(x)*S(x+1)) # wrt F(1)
        Qx_F[x-2] = (ξ/Tx) * F(1)*S(x) + π(x-1) # wrt F(x-1)
        Qx_F[x-1] = -(1/Tx) * (x*(1 - S(x)) + ξ*F(1)*S(x+1)) - π(x) - x*td # wrt F(x)
        Qx_F[x] = (1/Tx)*(x+1)*(1-S(x+1)) + (x+1)*td # wrt F(x+1)

        Jac[x-1,:] = np.array([*Qx_Ni,*Qx_F])


    # last row, F(x_max)
    Qxmax_Ni = -(1/Tx)*partial_S(x_max) * (x_max*F(x_max) + ξ*F(1)*F(x_max-1)) \
                + F(x_max-1)*partial_π[:,x_max-2] - F(x_max) * partial_π[:,x_max-1]
    Qxmax_F = np.zeros(len(Fvec))
    Qxmax_F[0] = (ξ/Tx)*F(x_max-1)*S(x_max,1)
    Qxmax_F[x_max-2] = (ξ/Tx)*F(1)*S(x_max,1) + π(x_max-1) # wrt F(x-1)
    Qxmax_F[x_max-1] = -(1/Tx)*x_max*S(1,x_max) - x_max*td
    Jac[-1,:] = np.array([*Qxmax_Ni, *Qxmax_F])
                                           
    
    return Jac
        
def fun_grad_func_response(i,x, N1,N2,H1,H2,**params):
    '''
    The gradient of the (scaled) functional response on prey i wrt N1, N2
    returns an array with 2 rows (N1, N2) and x_max columns
    '''
    alpha1 = fun_attack_rate(x, 1,**params)
    alpha2 = fun_attack_rate(x,2,**params)
    denom = (1 + alpha1 * H1 * N1 + alpha2 * H2 * N2)**2
    if i == 1:
        return np.array([ alpha1*(1 + alpha2 * H2 * N2), 
                         - alpha1 * alpha2 * H2 * N1])/denom
    elif i == 2:
        return np.array([ - alpha1 * alpha2 * H1 * N2,
                         alpha2 * (1 + alpha1 * H1 * N1)])/denom

def fun_grad_big_prey(N1,N2,Fvec, grad_Y_1, η1, A1, x_max, **params):
    '''
    return gradient of big prey vs n1, n2, F(1), .. F(x)
    '''
    
    # the sum of A_1 * F(x) * [ del Y_1/ del N_1, del Y_1 / del N_2]
    grad_sum_F_y = A1 * np.sum(grad_Y_1 * Fvec,1)
    
    delU1_N1 = η1 * (1 - 2 *N1) - grad_sum_F_y[0] 
    delU1_N2 = - grad_sum_F_y[1]

    xvec = np.arange(1,x_max+1,1)
    Y1_vec = fun_response_non_dim(xvec, N1, N2, 1,**params)
    delU1_F = -A1 * Y1_vec

    to_return = np.array([delU1_N1, delU1_N2, *delU1_F])

    return to_return

def fun_grad_small_prey(N1,N2,Fvec, grad_Y_2, η2, A1, x_max, **params):
    '''
    return gradient of small prey vs n1, n2, F(1), .. F(x)
    '''
    A2 = 1 - A1

    # the sum of A_2 * F(x) * [ del Y_2/ del N_1, del Y_2 / del N_2]
    grad_sum_F_y = A2 * np.sum(grad_Y_2 * Fvec,1)

    delU2_N1 = - grad_sum_F_y[0]
    delU2_N2 = η2 * (1 - 2 * N2) - grad_sum_F_y[1]

    xvec = np.arange(1,x_max+1,1)
    Y2_vec = fun_response_non_dim(xvec,N1,N2,2,**params)
    delU2_F = -A2 * Y2_vec

    to_return = np.array([delU2_N1, delU2_N2, *delU2_F])

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
