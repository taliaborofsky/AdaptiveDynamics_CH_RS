module AnalyzeResults

using UnPack
include("ModelHelperFuns.jl")
include("ModelFuns.jl")
using .ModelHelperFuns
using .ModelFuns

export get_p, get_meanx

function get_p(g, x_max)
    x = 1:x_max
    p = sum(x .* g, dims = 1)
end

function get_meanx(g::AbstractMatrix, x_max::Int, p::AbstractMatrix)
    """
    Average group size any individual is in when `p` is a vector.
    g is a 2d matrix
    """
    x_vec = 1:x_max
    numerator = (x_vec.^2) .* g
    mask = (p .> 1e-10) .& all(g .> 0, dims=1)
    numerator_sum = sum(numerator, dims=1)
    ans = copy(p)
    ans[mask] .= numerator_sum[mask] ./ p[mask]
    ans[.!mask] .= 1
    return ans
end

function get_meanx(g::AbstractVector, x_max::Int, p::Number)
    #=
    Average group size any individual is in when `p` is a scalar.
    will add method for p a vector and g a matrix later
    =#
    x_vec = 1:x_max
    numerator = x_vec.^2 .* g
    if p < 1e-10 && all(g .> 0)
        return 1
    else
        return sum(numerator) / p
    end
end

end