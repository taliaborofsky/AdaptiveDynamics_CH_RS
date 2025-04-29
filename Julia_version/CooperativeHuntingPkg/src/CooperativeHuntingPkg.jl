module CooperativeHuntingPkg

using UnPack#: @unpack

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")

using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults

export update_parameters, scale_parameters, fun_H1, fun_H2, fun_alpha1, fun_alpha2
export fun_f1, fun_f2, fun_W, fun_S_given_W
export fullsystem_scaled!, fullsystem!, fun_dg!, fun_dN1dT!, fun_dN2dT!
export get_p, get_meanx

end
