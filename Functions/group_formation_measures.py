import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from fun_response_funs import *
from fitness_funs import *
from group_formation_funs import *


def mean_group_size_membership(f_of_x_vec,p,x_max):
    '''
    average group size any individual is in
    this is not the same as the average group size
    '''
    xsquared = np.arange(1,x_max+1,1)**2
    total = sum(xsquared*f_of_x_vec)
    return total/p

def mean_group_size(f_of_x_vec,x_max):
    '''
    the average grp sizes
    '''
    total = 0
    num_grps = 0
    total = sum(np.arange(1,x_max+1,1)*np.array(f_of_x_vec))
    num_grps = sum(f_of_x_vec)
    return total/num_grps


def fun_num_groups(f_of_x_vec,p,x_max):
    total = 0
    for x in range(1,x_max+1):
        total += fun_f_of_x(x, f_of_x_vec, p, x_max,**dict())
    return total


def check_at_equilibrium(f_of_x_vec, p, M1, M2,  pop_process, **params):
    '''
    checks that dfdt = 0
    @input:
    f_of_x_vec = (f(1), f(2), ..., f(x_max))
    p, M1, M2 = pred, big prey, and small prey pop sizes, respectively
    pop_process = True or False, whether to include death rates and birth rates
    params = dic of params
    @returns
    1 if at equilibrium, 0 otherwise
    '''
    t = 1 # this doesn't matter
    dfdt_ = group_formation_model_separate(t, f_of_x_vec,p,M1,M2, params)
    at_equilibrium = np.abs(dfdt_) > 1e-10
    if sum(at_equilibrium) > 0: # at least one df(x)/dt is not zero
        return dfdt_, 0
    else:
        return dfdt_, 1
