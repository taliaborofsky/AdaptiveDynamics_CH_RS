#using UnPack

module ModelFuns

export fullsystem

function fullsystem(u,p)
    #=
    dN1/dT, dN2/dT, dg1/dT, dg2/dT, ... , dg(xmax)/dT
    u = N1,N2,g(1), g(2), ..., g(xmax)
    p is a named tuple or dictionary of parameters
    =#
    return u .+ 1  # Just an example
end

function fun_dN1(u,p)
    dN1dT = 0
    return dN1dT
end
end