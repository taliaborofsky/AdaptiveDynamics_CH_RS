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

# ModelHelperFuns
export update_parameters, scale_parameters, scale_parameters2
export fun_H1, fun_H2, fun_alpha1, fun_alpha2
export fun_f1, fun_f2, fun_W, fun_S_given_W
# ModelFuns
export fullsystem_scaled!, fullsystem_scaled2!, fullsystem!, fun_dg!, fun_dN1dT!, fun_dN2dT!
export fun_dg_nopop!, fun_dg_nopop
export system_scaled_nogroups, system_scaled_nogroups!, system_nogroups!
export system_nogroups

# AnalyzeResults
export get_p, get_meanx, get_prob_in_x
export ylabel_dic, param_label_dic
# MyBifTools
export extract_branch_matrix, extract_branch_matrix_nog, fullsystem_scaled, fullsystem_scaled_logTg
export recordFromSolution, simulate_and_plot
export iterate_to_last_pt_scaled, do_continuation
export plot_branches, plot_segments!, plot_nice_bif, plot_nice_bif_Tg
export do_base_continuations, do_base_continuations_nog
export recordFromSolution_nog, diagram_2_recursion, diagram_2_recursion_nog 
export make_and_save_nice_plots, make_and_save_nice_plots_nog
export plot_comparison_branches, plot_comparison_branches_filtered
export continue_sp
export equilibrium_nogroups
# ApparentCompTools
export Jacobian, Jacobian_g, get_∂N2_∂N1, get_∂N1_∂N2

end
