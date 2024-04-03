# This is based off of the network code from https://github.com/erolakcay/CooperationDynamicNetworks used in "Collapse and rescue of cooperation in evolving dynamic networks" 2018

using StatsBase
using LinearAlgebra

mutable struct NetworkParam
    pn::Float64            # probability connects to parent's neighbors
    pr::Float64            # probability connects to other random neighbors
    r0::Float64            # initial density of small prey
    R0::Float64            # initial density of big prey
    netsize::Int64         # num individuals
    generations::Int64     # num generations, so num time steps is netsize * generations
    br::Float64             # benefit of catching small prey
    bR::Float64             # total benefit of catching big prey
    betar::Float64           # depletion constant of small prey
    betaR::Float64           # depletion constant of big prey
    H::Float64            # horizontal transmission
    retint::Int64         # Interval at which the simulation saves output, 
                          # in terms of number of death/birth events. 
                          #Default = 0 reverts to saving things at intervals equal to network size
    replicates::Int64     # "number of replicates to run for the same parameter value"
    networksaveint::Int64
    
    # inner constructor allows me to have optional arguments
    # semicolons separate necessary from optional arguments. here all arguments have default constructors
    NetworkParam(;pn::Float64=0.5, pr::Float64=0.01, r0::Float64=1.0, R0::Float64=1.0, 
        netsize::Int64=100, generations::Int64=100, br::Float64=0.2, bR::Float64=1.0,
        betar::Float64=0.5, betaR::Float64=0.5, H::Float64=0.01, retint::Int64=0, replicates::Int64=1,
        networksaveint::Int64=1) = 
    new(pn, pr, r0, R0, netsize, generations, br, bR, betar, betaR, H, retint, replicates, networksaveint)
end

function runSim(params::NetworkParam)
    return 1,1,1,1,1
end

function runSimNetSave(params::NetworkParam)
    return 1,1,1,1,1
end