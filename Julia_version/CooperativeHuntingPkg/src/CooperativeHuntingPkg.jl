module CooperativeHuntingPkg

using UnPack#: @unpack

include("ModelHelperFuns.jl")
include("ModelFuns.jl")

using .ModelHelperFuns
using .ModelFuns

export fun_H1, fun_H2, fun_alpha1, fun_alpha2, fun_f1, fun_f2, fun_W

export fullsystem

end
