import numpy as np
import scipy as sp

def fun_alpha1(x, α1_of_1, s1, **params):
    ''' capture prob of big prey'''
    θ_1 = - np.log(1/α1_of_1 - 1)/(1-s1)
    return 1/(1 + np.exp(- θ_1 * (x - s1)))
    
def fun_alpha2(x, α2_fun_type, α2_of_1, s2, **params):
    ''' capture prob of small prey'''
    if α2_fun_type == 'constant':
        return α2_of_1
    else:
        θ_2 = - np.log(1/α2_of_1 - 1)/(1-s2)
        return 1/(1 + np.exp(- θ_2 * (x - s2)))

def fun_attack_rate(x, index, α1_of_1, α2_of_1, s1, s2, α2_fun_type,
                    **params):
    '''
    RETIRING THIS BECAUSE IT'S CUMBERSOME...
    
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
        if α2_fun_type == 'constant':
            return α2_of_1
        else:
            θ_2 = - np.log(1/α2_of_1 - 1)/(1-s2)
            return 1/(1 + np.exp(- θ_2 * (x - s2)))

def fun_Y1(x,N1,N2,**params):
    return fun_response_non_dim(x,N1,N2,1,**params)
def fun_Y2(x,N1,N2,**params):
    return fun_response_non_dim(x,N1,N2,2,**params)
def fun_response_non_dim(x, N1, N2, index, H1, H2, **params):
    '''
    non-dimensionalized functional response to prey as a function of predator group size (x) and 
    (non-dimensionalized prey population sizes (N1, N2)
    
    @inputs:
    x - pred group size
    N1, N2 - non-dim big prey and small prey pop sizes, respec
    index - 1 (big prey) or 2 (small prey)
    H1, H2 - non-dimensionalized handling times of big prey, small prey, respec
    params: a dictionary of other parameters, that at least must include 
                α1_of_1, α2_of_1, s1, s2

    @returns
    functional response for prey type <index> (a float)

    @examples
    >>fun_response_non_dim(x=1,N1=0.8,N2=0.8,index=1,a=1,H1=5,H2=5, 
                    **dict(α1_of_1 = 0.05, α2_of_1 = 0.95, s1 = 2, s2 = 2) )
    0.008000000000000002
    '''
    
    α2 = fun_attack_rate(x,2,**params) 
    α1 = fun_attack_rate(x,1,**params)
    if index == 1:
        numerator = α1*N1
    elif index == 2:
        numerator = α2*N2
    denominator = 1 + α1*H1*N1 + α2*H2*N2
    return numerator/denominator
def yield_from_prey_non_dim(x,N1,N2,β1, β2, **params):
    '''
    this is \tilde{pi} in the model, which is pi/(g1 + g2 + delta)
    @inputs:
    x - pred group size
    N1, N2 - big prey, small prey non-dim'ed pop size
    β1, β2 - prey profitability for big prey, small prey, respec
    '''
    tilde_π = β1 * fun_response_non_dim(x, N1, N2, 1,**params) \
                          + β2 * fun_response_non_dim(x, N1, N2, 2, **params)
    return tilde_π
def per_capita_fitness_from_prey_non_dim(x, N1, N2, β1, β2, **params):
    '''
    This is \tilde{w} in the model
    @inputs:
    x - pred group size
    N1, N2 - big prey, small prey non-dim'ed pop size
    β1, β2 - prey profitability for big prey, small prey, respec
    '''
    w_per_capita = (1/x)*(β1 * fun_response_non_dim(x, N1, N2, 1,**params) \
                          + β2 * fun_response_non_dim(x, N1, N2, 2, **params))
    return w_per_capita
    
def fitness_from_prey_non_dim(x, N1, N2, r, γ,**params):
    '''
    portion of inclusive fitness from each prey type, stored in an array, after potentially unequal sharing
    @inputs:
    x - pred group size
    N1, N2 - big prey, small prey non-dim pop size
    r - relatedness between group members
    γ - extent of reproductive skew (portion of subordinate's food donated to dominant)
    params - dictionary of other parameters

    @returns:
    np.array([<inclusive fitness from big prey>, <inclusive fitness from small prey>])
    (so the rows correspond to prey types
    '''
    # set portion size, need to account for x being an array

    w_per_capita = per_capita_fitness_from_prey_non_dim(x, N1, N2, **params)
    try:
        if x > 1:
            repro_exchange = (1-γ)*(1-r) + r*x
            return w_per_capita * repro_exchange
        else:
            return w_per_capita
    except ValueError:
        repro_exchange = np.ones(np.shape(x))
        repro_exchange[x>1] = (1-γ)*(1-r) + r*x[x>1]
        return  w_per_capita * repro_exchange
