import numpy as np
import scipy as sp
from fitness_funs_non_dim import *

#from fun_response_funs import *

def pop_model_one_grp_size_x_constant(t, initialstate, x, params):
    P, N1, N2= initialstate
    dPdT = dPdT_one_grp_size(P, N1, N2, x, **params)
    dN1dT = dN1dT_one_grp_size(P, N1, N2, x, **params)
    dN2dT = dN2dT_one_grp_size(P, N1, N2, x, **params)
    return [dPdT, dN1dT, dN2dT]
    
def full_system_one_grp(initialstate, x, params):
    ''' need this for finding roots'''
    t=0
    return pop_model_one_grp_size_x_constant(t, initialstate, x, params)


    
def dPdT_one_grp_size(P, N1, N2, x, η1, η2, β1, β2, **params):
    '''
    the equation for dPdT, the change in predator population size versus time, 
    non-dimensionalized. All preditors in groups of the same size

    @inputs
    P, N1, N2 - nondimensionalized predator, big prey, and small prey pop sizes
    x - the group size of predators
    params - dic of params: η1, η2, A1, β1, β2, H1, H2, α1_of_1, α2_of_1, s1, s2,
    '''
    F_of_x = P/x
    tildeY1_of_x = fun_response_non_dim(x,N1,N2,1,**params)
    tildeY2_of_x = fun_response_non_dim(x,N1,N2,2,**params)
    td = 1 - η1 - η2

    return F_of_x * (β1 * tildeY1_of_x + β2 * tildeY2_of_x) - td*P
    
def dN1dT_one_grp_size(P, N1, N2, x, η1, A1, **params):
    tildeY1_of_x = fun_response_non_dim(x,N1,N2,1,**params)
    F_of_x = P/x
    return η1*N1*(1-N1) - A1 * F_of_x * tildeY1_of_x


def dN2dT_one_grp_size(P, N1, N2, x, η2, A1, **params):
    tildeY2_of_x = fun_response_non_dim(x,N1,N2,2,**params)
    F_of_x = P/x
    A2 = 1 - A1
    return η2*N2*(1-N2) - A2 * F_of_x * tildeY2_of_x   


