import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

    
def fun_response(x,M1,M2,index,a1,a2,h1,h2,**params):
    '''
    functional response to prey as a function of predator group size (x) and 
    prey population sizes (M1, M2)
    a_iα_i(x)M_i/(1 + h_1 a_1 α_1(x) M_1 + h_2 a_2 α_2(x) M_2)
    
    @inputs:
    x - pred group size
    M1 - big prey pop size
    M2 - small prey pop size
    index - 1 (big prey) or 2 (small prey)
    a1 - attack rate of big prey
    a2 - attack rate of small prey
    h1 - handling time per big prey caught
    h2 - handling time per small prey caught
    params: a dictionary of other parameters, that at least must include α1_of_1, α2_of_1, s1, s2

    @returns
    functional response for prey type <index> (a float)

    @examples
    >>fun_response(x=1,M1=10,M2=10,index=1,a1=1,a2=1,h1=0.5,h2=0.5, 
                    **dict(α1_of_1 = 0.05, α2_of_1 = 0.95, s1 = 2, s2 = 2) )
    0.08333333333333336
    (answer should be 10*a1*α1_of_1/(1+h1*a1*α1_of_1*10 + h2*a2*α2_of_1*10) = 0.08333333333333333
    
    '''
    
    α1 = fun_attack_rate(x,1,**params)
    α2 = fun_attack_rate(x,2,**params)
    if index == 1:
        numerator = a1*α1*M1
    elif index == 2:
        numerator = a2*α2*M2
    denominator = 1 + a1*α1*h1*M1 + a2*α2*h2*M2
    return numerator/denominator


def fun_attack_rate(x, index, α1_of_1, α2_of_1, s1, s2, **params):
    '''
    The attack rate as a function of x
    
    @inputs:
    x: group size, 1,2,3,...
    index: 1 or 2, indicates prey type 1 (big prey) or 2 (small prey)
    α1_of_1: the attack rate of big prey for group size 1
    α2_of_1: the attack rate of small prey for group size 1
    s1: critical group size for big prey, must be >= 2
    s2: critical group size for small prey, must be >= 2
    
    @returns:
    attackrate (a float)

    @example:
    >> fun_attack_rate(1,2,0.05,0.95,2,2,**dict())
    0.9500000000000001
    >> fun_attack_rate(1,1,0.05,0.95,2,2,**dict())
    0.05000000000000001
    
    '''
    if index == 1:
        θ_1 = - np.log(1/α1_of_1 - 1)/(1-s1)
        return 1/(1 + np.exp(- θ_1 * (x - s1)))
    elif index == 2:
        θ_2 = - np.log(1/α2_of_1 - 1)/(1-s2)
        return 1/(1 + np.exp(- θ_2 * (x - s2)))
    
