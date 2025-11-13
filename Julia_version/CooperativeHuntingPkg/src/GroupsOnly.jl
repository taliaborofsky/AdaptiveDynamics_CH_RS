module GroupsOnly

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults

using UnPack
using LinearAlgebra
using ForwardDiff #this should be able to numerically find a jacobian
using Polynomials
using LaTeXStrings
using Plots

export find_mangel_clark, get_g_equilibria, classify_equilibrium_g
export update_params, bifurcation_g_input, get_x_maximizes_pc_fitness
export heatmap_bif_g
export get_g_equilibria_givenW

# these were supposed to load with my package but i guess it didn't work
ylabel_dic = Dict(
    :N1 => L"N_1"*", Scaled Big Prey\nDensity",
    :N2 => L"N_2"*", Scaled Small Prey\nDensity",
    :mean_x => "Mean Experienced\nGroup Size, "*L"\bar{x}",
    :p => L"Predator density, $p$",
    :Nsum => "Sum of Prey Densities,\n"*L"N_1 + N_2",
    :g1 => L"g(1)",
    :g2 => L"g(2)",
    :g3 => L"g(3)"
)

param_label_dic = Dict(
    :α1_of_1 => L"\alpha_1(1)",
    :α2_of_1 => L"\alpha_2(1)",
    :s1 => L"s_1",
    :s2 => L"s_2",
    :H1a => L"H_{1a}",
    :H2a => L"H_{2a}",
    :H2b => L"H_{2b}",
    :A1 => L"Relative Attack Rate of Big Prey, $A_1$",
    :A2 => L"A_2",
    :η2 => L"Growth of Small Prey, $\eta_2$",
    :β2 => L"\beta_2",
    :x_max => L"x_{max}",
    :Tg => "Relative Group Dynamics \nTimescale, "*L"T_g",
    :d => L"d",
    :scale => L"Scale, $\beta_1/\beta_2$"
)

function find_mangel_clark(N1, N2, params)
    # Mangel and Clark predicted that groups should grow until W(x^*) = W(1)
    # Simplest way: iterate and stop when W(x) < W(1), then return x - 1
    @unpack x_max = params
    W_of_1 = fun_W(1, N1, N2, params)
    for x in 2:x_max
        W_of_x = fun_W(x, N1, N2, params)
        if W_of_x < W_of_1
            return x - 1
        end
    end
    return x_max  # If reach x_max
end


function get_g_equilibria(P, N1, N2, params)
    """
    Finds all the g(x) equilibria for a certain p, N1, N2 combination.
    Assumes population sizes are constant.

    Returns gvec
    """
    x_max = params[:x_max]
    xvec = 1:x_max

    # Get the root for g(1)

    W = fun_W(xvec,N1,N2,params)
    S_of_1_x = fun_S_given_W(W[1], W, params)
    S_of_x_1 = 1 .- S_of_1_x
    c_vec = S_of_x_1./(xvec.*S_of_1_x)
    c_vec[2] = c_vec[2]/2    
    # Compute coefficients for g(1)
    coefficients = [x * prod(c_vec[1:x]) for x in xvec]  # Reverse order
    coeff_full = vcat(-P, coefficients)  # Append -P to the coefficients
    # Find roots
    roots_all = roots(Polynomials.Polynomial(coeff_full))
    
    # Filter real positive roots. there should only be one.
    g1 = real(filter(x -> isreal(x) && real(x) > 0, roots_all)[1])

    # Compute g(x) for each g1 root
    gvec = [prod(c_vec[1:x]) * g1^x for x in xvec]

    return gvec
end


function classify_equilibrium_g(g, N1, N2, params)
    """
    Compute the eigenvalues of the Jacobian matrix for just the dynamics of dg(x)/dt.
    Returns:
    - "Stable (attractive)"
    - "Unstable"
    - "Marginally stable (needs further analysis)"
    - "Indeterminate stability (needs further analysis)"
    """
    # Compute the Jacobian matrix for group dynamics
    J = ForwardDiff.jacobian(u -> fun_dg_nopop(u, params, 1),g[1:end-1])
    #J = Jacobian_g(N1, N2, g, params)

    # Compute the eigenvalues of the Jacobian matrix
    eigenvalues = eigen(J).values
    # Check the real parts of the eigenvalues
    real_parts = real.(eigenvalues)

    # Classify the stability based on the real parts of the eigenvalues
    if all(real_parts .< 0)
        return true
    else
        return false
    end
end
function update_params(paramkey::Symbol, param, params_base::Dict{Symbol, Any})
    # Create a copy of the base parameters
    params = deepcopy(params_base)
    
    # Update the parameter specified by paramkey with the new value
    params[paramkey] = param
    
    # Scale the parameters
    params = scale_parameters(params)
    
    return params
end

function bifurcation_g_input(p, N1, N2, paramkey::Symbol, 
    paramvec, params_base::Dict{Symbol, Any})
    #=
    Loop over elements of paramvec, 
    finding the g equilibrium and stability for each
    paramater value
    =#
    x_max = params_base[:x_max]
    # Initialize arrays to store equilibrium g values and their corresponding stability
    results_g = zeros(length(paramvec), x_max)  # 2D array
    stability_results = Vector{Bool}(undef, length(paramvec))
    
    
    # Iterate over paramvec
    for (i, param) in enumerate(paramvec)
        # Update parameters
        params = update_params(paramkey, param, params_base)

        # Find the single equilibrium g vector
        gvec = get_g_equilibria(p, N1, N2, params)

        # Get stability
        stability = classify_equilibrium_g(gvec, N1, N2, params)

        # Store results
        results_g[i, :] = gvec  # Store the vector as a row
        stability_results[i] = stability
    end

    return Dict(
        :results_g => results_g, 
        :stability_results => stability_results
        )
end

function get_x_maximizes_pc_fitness(N1, N2, params)
    xvec = 1:params[:x_max]
    W_of_x = fun_W(xvec, N1, N2, params)  # Use fun_W instead of per_capita_fitness_from_prey_non_dim
    max_index = argmax(W_of_x)
    return max_index
end


# stuff for a simple Gaussian W

function get_g_equilibria_givenW(P, W, params)
    """
    Finds all the g(x) equilibria for a certain p, N1, N2 combination.
    Assumes population sizes are constant.

    Returns gvec
    """
    x_max = params[:x_max]
    xvec = 1:x_max

    # if no l, phi in params, then set equal to 1
    for k in (:leave_param, :fuse_param)
        get!(params, k, 1)
    end

    @unpack leave_param,fuse_param = params
    l, ϕ = leave_param, fuse_param
    # Get the root for g(1)

    S_of_1_x = fun_S_given_W(W[1], W, params)
    S_of_x_1 = 1 .- S_of_1_x
    c_vec = S_of_x_1./(l .* xvec.* S_of_1_x)
    c_vec[1] = 1.0

    # Compute coefficients for g(1)
    coefficients = [x * ϕ/2 * prod(c_vec[1:x]) for x in xvec]  # Reverse order
    coefficients[1] = 1.0 # the coefficient of g_1 

    coeff_full = vcat(-P, coefficients)  # Append -P to the coefficients
    # Find roots
    roots_all = roots(Polynomials.Polynomial(coeff_full))

    # Filter real positive roots. there should only be one.
    g1 = real(filter(x -> isreal(x) && real(x) > 0, roots_all)[1])

    # Compute g(x) for each g1 root
    gvec = [0.5 * ϕ * prod(c_vec[1:x]) * g1^x for x in xvec]
    gvec[1] = g1

    return gvec
end

function fun_W_gauss(x, p)
    @unpack a, x0, σ = p # height, x that maximizes fecundity, standard deviation
    W = a .* exp.( - (x .- x0).^2 ./(2*σ^2))
end

function bifurcation_g_input_simpleW(p, paramkey::Symbol, 
    paramvec, params_base::Dict{Symbol, Any})
    #=
    Loop over elements of paramvec, 
    finding the g equilibrium and stability for each
    paramater value
    =#
    x_max = params_base[:x_max]
    # Initialize arrays to store equilibrium g values and their corresponding stability
    results_g = zeros(length(paramvec), x_max)  # 2D array
    stability_results = Vector{Bool}(undef, length(paramvec))
    xvec = 1:x_max
    params = deepcopy(params_base)

    # Iterate over paramvec
    for (i, param) in enumerate(paramvec)

        # Update parameters
        params[paramkey] = param

        # Find the single equilibrium g vector
        W = fun_W_gauss(xvec,params)
        gvec = get_g_equilibria_givenW(p, W, params)

        # Get stability
        stability = classify_equilibrium_g(gvec, N1, N2, params)

        # Store results
        results_g[i, :] = gvec  # Store the vector as a row
        stability_results[i] = stability
    end

    return Dict(
        :results_g => results_g, 
        :stability_results => stability_results
        )
end

#= add l, phi to p if not there, if p is dict or named tupe =#
ensure_l_phi(p) = p isa AbstractDict ?
    (get!(p, :l, 1); get!(p, :phi, 1.0); p) :
    (; p..., l = get(p, :l, 1.0), phi = get(p, :phi, 1.0))


function fun_dg_simpleW!(dg, g, p, T)
#=
Group dynamics with leaving and singletons "modulated" by a leave_param and fuse_param
Uses a gaussian W
=#
    # p - parameters - has a, x0, σ, Tg, d, and x_max
    
    # if no l, phi in paramater dictionary (called p), then set equal to 1
    p = ensure_l_phi(p)
    # for k in (:leave_paaram, :fuse_param)
    #     get!(p, k, 1)
    # end
    # unpack basic ingredients
    @unpack x_max, Tg, fuse_param, leave_param = p
    xvec = 1:x_max

    # i'll need fitnesses and best response functions
    Wvec = fun_W_gauss(xvec, p) 
    W1 = Wvec[1]
    S_1_x = fun_S_given_W(Wvec[1],Wvec, p)
    for x in xvec
        if x==1
            if x_max == 1
                groups_2_split = 0
                join_groups = 0
                leave_larger_grps = 0
            else
                groups_2_split = 4*g[2]*S_1_x[2]/Tg
                join_groups = -(g[1]/Tg)*sum(g[2:end-1].*(1 .- S_1_x[3:end]))
                join_groups_singletonsfuse = - (g[1]./ Tg) .*g[1] .* (1 .- S_1_x[2]) 
                if x_max > 2
                    leave_larger_grps = sum(xvec[3:end].*g[3:end].*S_1_x[3:end])/Tg
                else
                    leave_larger_grps = 0
                end
            end
            dg[1] = (leave_param * (groups_2_split + leave_larger_grps) + join_groups 
                        + fuse_param * join_groups_singletonsfuse)

        elseif x == 2
            individual_leaves = - 2*g[2]*S_1_x[2]/Tg
            if x_max == 2
                threes_to_pairs = 0
                pairs_to_threes = 0
            else
                pairs_to_threes = - g[2]*g[1]*(1-S_1_x[3])/Tg
                threes_to_pairs = 3*g[3]*S_1_x[3]/Tg
                # deaths = td * (3*g[3] - 2*g[2])
                # births = g[1]*Wvec[1] - 2*g[2]*Wvec[2]
            end
            form_dyads = (g[1])^2*(1-S_1_x[2])/(2*Tg)
            dg[2] = (leave_param * (individual_leaves + threes_to_pairs) 
                    + pairs_to_threes + fuse_param * form_dyads 
                )

        elseif x == x_max
            individual_leaves = - x*g[x]*S_1_x[x]/Tg
            smaller_grp_grows_to_xm  = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            dg[x] = (leave_param * individual_leaves + smaller_grp_grows_to_xm )

        else
            individual_leaves = -(x/Tg)*g[x]*S_1_x[x]
            grows_to_larger_grp = - g[x]*g[1]*(1 - S_1_x[x+1])/Tg
            smaller_grp_grows_to_x = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            larger_grps_shrink = (x+1)*g[x+1]*S_1_x[x+1]/Tg
            dg[x] = (leave_param * (individual_leaves + larger_grps_shrink) 
                    + grows_to_larger_grp + smaller_grp_grows_to_x
                 )
            
        end
        # do something with x 
    end
end

function fun_dg_simpleW(g, p, T)
    dg = deepcopy(g)
    fun_dg_simpleW!(dg, g, p, T)
    return dg
end

function classify_equilibrium_g_simpleW(g, N1, N2, params)
    """
    Compute the eigenvalues of the Jacobian matrix for just the dynamics of dg(x)/dt.
    Returns:
    - "Stable (attractive)"
    - "Unstable"
    - "Marginally stable (needs further analysis)"
    - "Indeterminate stability (needs further analysis)"
    """
    # Compute the Jacobian matrix for group dynamics
    J = ForwardDiff.jacobian(u -> fun_dg_simpleW(u, params, 1),g[1:end-1])
    #J = Jacobian_g(N1, N2, g, params)

    # Compute the eigenvalues of the Jacobian matrix
    eigenvalues = eigen(J).values
    # Check the real parts of the eigenvalues
    real_parts = real.(eigenvalues)

    # Classify the stability based on the real parts of the eigenvalues
    if all(real_parts .< 0)
        return true
    else
        return false
    end
end

function heatmap_bif_g(gmat, P::Number, N1::Number, N2::Number, paramkey, paramvec, params_base)
    #=
    uses a heatmap to plot Prob(x), if P, N1, N2 constant
    uses the function fun_W to get W and thus find the highest x for which W(x) >= W(1) (called x^*)
        and the x that maximizes W (called x_0)
    =#

    prob = get_prob_in_x(gmat, P, params_base[:x_max])
    hm = heatmap(
        paramvec,     # x = rows
        1:size(prob,2),     # y = columns
        prob',              # transpose so rows map to x
        c = cgrad(:grays, rev=true),    # high values → dark
        #colorrev=false,      # darker = higher
        xlabel=param_label_dic[paramkey],
        ylabel=L"Group size, $x$",
        title="Probability heatmap",
        ylims = [1, params_base[:x_max]]
    )
    # find x_mc
    x_mc_vec = similar(paramvec)
    x_opt_vec = similar(paramvec)
    for (i,param) in enumerate(paramvec)
            params = update_params(paramkey, param, params_base)
            W = fun_W(1:params[:x_max],N1,N2,params)

            # give the index of the first group size x, where x > 1, such that W(x) >=W(1) and W(x+1)<W(1).
            # otherwise (if the fitness is > W(1) for all group sizes) give the maximum group size
            x_mc_vec[i] = any(W[2:end] .< W[1]) ? findfirst( W[2:end] .< W[1]) : params[:x_max]
            x_opt_vec[i] = argmax(W)
    end
    plot!(paramvec, x_mc_vec, label = L"x^*")
    plot!(paramvec, x_opt_vec, label = L"x_0")
    return hm
end


end