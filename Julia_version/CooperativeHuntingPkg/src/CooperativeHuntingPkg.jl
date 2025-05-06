module CooperativeHuntingPkg

using UnPack#: @unpack

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
include("MyBifTools.jl")
include("ApparentCompTools.jl")

using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults
using .MyBifTools
using .ApparentCompTools

export update_parameters, scale_parameters, fun_H1, fun_H2, fun_alpha1, fun_alpha2
export fun_f1, fun_f2, fun_W, fun_S_given_W
export fullsystem_scaled!, fullsystem!, fun_dg!, fun_dN1dT!, fun_dN2dT!
export get_p, get_meanx
export extract_branch_matrix, fullsystem_scaled, recordFromSolution, simulate_and_plot
export iterate_to_last_pt_scaled, do_continuation
export plot_branches, plot_segments!, plot_nice_bif, do_base_continuations
export Jacobian, fun_grad_func_response

end
