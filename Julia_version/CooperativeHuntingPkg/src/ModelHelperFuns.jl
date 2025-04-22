module ModelHelperFuns
using UnPack

export fun_H1, fun_H2, fun_alpha1, fun_alpha2, fun_f1, fun_f2, fun_W

function fun_H1(x, parameters)

    @unpack H1a, H1b = parameters 

    H1a + H1b / x
end

function fun_H2(x, parameters)
    @unpack H2a, H2b = parameters 
    return H2a + H2b / x
end

function fun_alpha1(x, parameters)
    @unpack α1_of_1, s1 = parameters
    
    θ_1 = -log(1/α1_of_1 - 1) / (1 - s1)
    return 1 / (1 + exp(-θ_1 * (x - s1)))
end

function fun_alpha2(x, parameters)
    @unpack α2_fun_type, α2_of_1, s2 = parameters

    if α2_fun_type == "constant"
        return α2_of_1
    else
        θ_2 = -log(1 / α2_of_1 - 1) / (1 - s2)
        return 1 / (1 + exp(-θ_2 * (x - s2)))
    end
end

fun_f1(x, N1, N2, parameters) = fun_response_non_dim(x, N1, N2, 1, parameters)
fun_f2(x, N1, N2, parameters) = fun_response_non_dim(x, N1, N2, 2, parameters)

function fun_response_non_dim(x, N1, N2, index, parameters)
    @unpack A1, A2 = parameters

    H1 = fun_H1(x, parameters)
    H2 = fun_H2(x, parameters)
    α1 = fun_alpha1(x, parameters)
    α2 = fun_alpha2(x, parameters)

    numerator = index == 1 ? A1 * α1 * N1 :
                index == 2 ? A2 * α2 * N2 :
                error("Invalid index: must be 1 or 2")

    denominator = 1 + α1 * H1 * N1 + α2 * H2 * N2

    return numerator / denominator
end

function fun_W(x, N1, N2, parameters)
    #= 
    per capita fitness from prey
    =#
    @unpack β1, β2 = parameters
    f1 = fun_f1(x,N1,N2,parameters)
    f2 = fun_f2(x,N1,N2,parameters)

    # this is what's being returned
    W = (β1*f1 + β2*f2)/x
end

end