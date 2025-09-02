module MyBifTools

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults
using Plots 
using DifferentialEquations
using BifurcationKit
using LaTeXStrings
using Measures

# plotting defaults
default(
    guidefontsize=22,   # controls xlabel, ylabel, title font size
    tickfontsize=12,    # controls axis number labels
    legendfontsize=14,  # controls legend font
    linewidth=2,        # controls default line thickness
    grid = false,        # turns off grid in background
    fontfamily="Computer Modern" # font family that matches latex
)

export extract_branch_matrix, extract_branch_matrix_nog, fullsystem_scaled, fullsystem_scaled_logTg
export recordFromSolution, simulate_and_plot
export iterate_to_last_pt_scaled, do_continuation
export plot_branches, plot_segments!, plot_nice_bif, plot_nice_bif_Tg
export do_base_continuations, do_base_continuations_nog
export recordFromSolution_nog, diagram_2_recursion, diagram_2_recursion_nog 
export make_and_save_nice_plots, make_and_save_nice_plots_nog
export plot_comparison_branches, plot_comparison_branches_filtered

ylabel_dic = Dict(
    :N1 => L"N_1"*", Scaled Big Prey\nDensity",
    :N2 => L"N_2"*", Scaled Small Prey\nDensity",
    :mean_x => "Mean Experienced\nGroup Size, "*L"\bar{x}",
    :P => L"Predator density, $P$",
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
    :scale => L"Scale, $\beta_1/\beta_2$",
    :x => "Group Size"
)
# Group and Pop Dynamics
    function extract_branch_matrix(branch, x_max)
        # Start with N1 and N2
        fields = [:N1, :N2]
        
        # Add g1, g2, ..., gx_max dynamically
        append!(fields, Symbol("g$i") for i in 1:x_max)

        push!(fields, :mean_x)
        push!(fields,:P)
        
        # Extract the fields dynamically using getproperty
        branch_out = [getproperty(branch, field) for field in fields]
        branch_out_mat = hcat(branch_out...)

        # # filter out negative values
        # filtered_mat = filter(row -> all(x -> x >= 0, row), eachrow(branch_out_mat))
        # filtered_mat = reduce(hcat, filtered_mat)
        
        # filter out negative values and keep track of the indices
        valid_indices = findall(row -> all(x -> x>= -1e-5, row)
                        && all(x-> x<=1.02, row[1:2]), 
                        eachrow(branch_out_mat))
        # valid_indices = [i for (i, row) in 
        #                 enumerate(eachrow(branch_out_mat)) 
        #                 if all(x -> x >= -1e-10, row)]
        filtered_mat = branch_out_mat[valid_indices, :]

        # Filter branch.stability using the valid indices
        filtered_stability = Bool.(branch.stable[valid_indices])
        filtered_param = branch.param[valid_indices]
        
        out_dict = Dict(field => filtered_mat[:, i] 
                    for (i, field) in enumerate(fields))
        out_dict[:param] = filtered_param
        out_dict[:stable] = filtered_stability
        out_dict[:Nsum] = out_dict[:N1] .+ out_dict[:N2]

        out_nt = NamedTuple(out_dict)


        return out_nt
    end

    function fullsystem_scaled(u,p,T=0)
        du = copy(u).*0
        fullsystem_scaled!(du,u,p,T)
        return du
    end

    function fullsystem_scaled_logTg(u,p,T=0)
        du = copy(u).*0.0
        # Create a new NamedTuple with Tg updated to 10^Tg_exp
        new_p = (;p..., (Tg = 10.0^p.Tg_exp))
        fullsystem_scaled!(du,u,new_p,T)
        return du
    end

    function recordFromSolution(u,p_nt; k...)
        N1, N2 = u[1], u[2]
        x_max = length(u[3:end])
        g = NamedTuple(Symbol("g$i") => u[2 + i] for i in 1:x_max)  # Dynamically assign g1, g2, ..., gx_max
        P = get_p(u[3:end], x_max)
        mean_x = get_meanx(u[3:end], x_max, P)
        return (N1 = N1, N2 = N2, g..., P = P, mean_x = mean_x)
    end


    function manually_get_branches(p_nt)
        ### coexist
        u0 = [0.1,0.1, fill(0.1,params_base[:x_max])...]

        #set last point as initialcondition
        u0 = iterate_to_last_pt_scaled(u0, p_nt; tf = 10000)
        br_co = do_continuation(u0, p_nt)

        #### predators extinct
        x_max = params[:x_max]
        u1 = [1.0,1.0,zeros(x_max)...]
        br_P_extinct = do_continuation(u1, p_nt)

        #### big prey extinct
        u0 = [0,0.9,fill(0.1,x_max)...]
        u3 = iterate_to_last_pt_scaled(u0, p_nt; tf = 10000)
        br_N1_extinct = do_continuation(u3, p_nt)

        ##### small prey extinct
        u0 = [1.0,0,fill(0.1,x_max)...]
        u2 = iterate_to_last_pt_scaled(u0, p_nt; tf = 10000)
        br_N2_extinct = do_continuation(u2, p_nt)

        return (coexist = br_co, predator_extinct = br_P_extinct, 
                big_prey_extinct = br_N1_extinct, 
                small_prey_extinct = br_N2_extinct)
    end

    function plot_branches(br_list; 
        vars_list = [(:param, :N1), (:param, :N2)], 
        layout = (1, 2), legend = false, 
        ylims_list = fill([-0.01, 1.05], length(vars_list)),
        plot_special_points = true)
        ylabel_dic = Dict(
            :N1 => L"N_1",
            :N2 => L"N_2",
            :mean_x => L"\bar{x}",
            :P => L"p"
        )
        
        opts = (linewidthunstable = 1, linewidthstable = 3, 
            plotspecialpoints = plot_special_points)
        plt_list = []
        for i in 1:length(vars_list)  # Iterate over indices of vars_list
            vars = vars_list[i]
            ylab = ylabel_dic[vars[2]]
            plt = plot(br_list..., vars = vars; 
                    ylims = ylims_list[i], 
                    ylabel = ylab, opts...)
            push!(plt_list, plt)
        end
        display(plot(plt_list..., layout = layout; legend = legend))
        return plt_list
    end

    function plot_nice_bif_Tg(br_list, y_axis_symbol;
        has_g = true, 
        x_max = 5,
        ylims = nothing,
        colorblind_palette = [:blue, :green, 
        :purple, :pink, :brown, :gray, :orange, ])
        
        """
        Plots branches with a specified y-axis variable for the x axis being Tg

        # Arguments
        - `br_list`: A list of branches to plot.
        - `y_axis_symbol`: The symbol representing the y-axis variable (e.g., `:N1`, `:N2`).
        - `colorblind_palette`: A palette of colors for the branches (default is colorblind-friendly).

        # Returns
        - A plot of the branches with the specified y-axis variable.

        x_axis_symbol
        """

        plt = plot()
        colorblind_palette = [colorblind_palette..., # doing this allows us to handle more branches
                            colorblind_palette..., colorblind_palette...]
        for (j, branch) in enumerate(br_list)
            out_nt = extract_branch_matrix(branch, x_max)
            if !isempty(out_nt.param)  # Check if out_nt.param is not empty
                x = 10.0 .^ out_nt.param
                plot_segments!(x, getfield(out_nt, y_axis_symbol), Bool.(out_nt.stable); 
                            plot_legend = false, color = colorblind_palette[j])
            end
        end
        # Add legend for stability
        #plot!([], [], linestyle=:solid, color=:black, label="stable", legend=true)
        #plot!([], [], linestyle=:dash, color=:black, label="unstable", legend=true)
        plot!(xlabel = "\n"*param_label_dic[:Tg],
                ylabel = ylabel_dic[y_axis_symbol], bottommargin=6mm,
                xscale = :log10)
        if ylims !== nothing
            plot!(ylims = ylims)
        end     
        plot!(plt,size = (600,480))
        return plt

        end



    function do_base_continuations(p_nt, x_max; lens = (@optic _.scale), 
        p_min = 1.01, p_max = 8.0, systemfunction =fullsystem_scaled, dsmax = 0.01)
        """
        p_nt is the named tuple of parameters
        x_max is max group size
        returns br_list, a named tuple

        k can include lens = (@optic _.scale)
        """
        ### find coexistence
        kargs = (
            systemfunction = systemfunction, 
            lens = lens, p_min = p_min, 
            p_max = p_max)

        u0 = fill(0.1, x_max+2)
        #set last point as initialcondition
        u0 = iterate_to_last_pt_scaled(u0, p_nt; tf = 500)
        br_co = do_continuation(u0, p_nt; kargs...)

        #### predators extinct
        u1 = [1.0,1.0,zeros(x_max)...]
        br_P_extinct = do_continuation(u1, p_nt; kargs...)


        ##### small prey extinct
        u0 = [1.0,0,fill(0.1,x_max)...]
        u2 = iterate_to_last_pt_scaled(u0, p_nt; tf = 10000)
        br_N2_extinct = do_continuation(u2, p_nt; kargs...)

        #### big prey extinct
        u0 = [0,0.99,fill(0.1,x_max)...]
        u3 = iterate_to_last_pt_scaled(u0, p_nt; tf = 1000)
        br_N1_extinct = do_continuation(u3, p_nt; kargs...)


        ### big prey and predator extinct
        u0 = [.99,0,fill(0,x_max)...]
        u3 = iterate_to_last_pt_scaled(u0, p_nt; tf = 1000)
        br_N1P_extinct = do_continuation(u3, p_nt; kargs...)

        ### small prey and predator extinct
        u0 = [0,0.9,fill(0,x_max)...]
        u3 = iterate_to_last_pt_scaled(u0, p_nt; tf = 1000)
        br_N2P_extinct = do_continuation(u3, p_nt; kargs...)

        br_list = (coexist = br_co, predator_extinct = br_P_extinct, 
                N2_extinct = br_N2_extinct, N1_extinct = br_N1_extinct,
                    N1P_extinct = br_N1P_extinct, N2P_extinct = br_N2P_extinct)

        return br_list # actually a named tuple


    end

# No Group Dynamics
    function recordFromSolution_nog(u,p_nt; k...)
        N1, N2, P = u[1], u[2], u[3]
        return (N1 = N1, N2 = N2, P = P)
    end

    function extract_branch_matrix_nog(branch, x)
        # x is unused place holder
        
        fields = [:N1, :N2, :P]
        
        # Extract the fields dynamically using getproperty
        branch_out = [getproperty(branch, field) for field in fields]
        branch_out_mat = hcat(branch_out...)

        # # filter out negative values
        # filtered_mat = filter(row -> all(x -> x >= 0, row), eachrow(branch_out_mat))
        # filtered_mat = reduce(hcat, filtered_mat)
        
        # filter out negative values and keep track of the indices
        valid_indices = findall(row -> all(x -> x>= -1e-5, row)
                        && all(x-> x<=1.02, row[1:2]), 
                        eachrow(branch_out_mat))
        # valid_indices = [i for (i, row) in 
        #                 enumerate(eachrow(branch_out_mat)) 
        #                 if all(x -> x >= -1e-10, row)]
        filtered_mat = branch_out_mat[valid_indices, :]

        # Filter branch.stability using the valid indices
        filtered_stability = Bool.(branch.stable[valid_indices])
        filtered_param = branch.param[valid_indices]
        
        out_dict = Dict(field => filtered_mat[:, i] 
                    for (i, field) in enumerate(fields))
        out_dict[:param] = filtered_param
        out_dict[:stable] = filtered_stability
        out_dict[:Nsum] = out_dict[:N1] .+ out_dict[:N2]

        out_nt = NamedTuple(out_dict)


        return out_nt
    end
    function do_base_continuations_nog(
        p_nt; lens = (@optic _.scale), 
        p_min = 0.1, p_max = 8.0, 
        systemfunction =system_scaled_nogroups, 
        dsmax = 0.01)
        """
        p_nt is the named tuple of parameters
        returns br_list, a named tuple

        k can include lens = (@optic _.scale)
        """
        ### find coexistence
        kargs = (
            systemfunction = systemfunction, 
            lens = lens, p_min = p_min, 
            p_max = p_max)

        u0 = fill(0.1, 3)
        #set last point as initialcondition
        u0 = iterate_to_last_pt_scaled(
            u0, p_nt; 
            tf = 500, 
            systemfunction = systemfunction)
        br_co = do_continuation(u0, p_nt; kargs...)

        #### predators extinct
        u1 = [1.0,1.0,0]
        br_P_extinct = do_continuation(u1, p_nt; kargs...)


        ##### small prey extinct
        u0 = [0.99,0,0.1]
        u2 = iterate_to_last_pt_scaled(
            u0, p_nt; 
            tf = 10000,
            systemfunction = systemfunction)
        br_N2_extinct = do_continuation(u2, p_nt; kargs...)

        #### big prey extinct
        u0 = [0,0.99,0.1]
        u3 = iterate_to_last_pt_scaled(
            u0, p_nt; 
            tf = 10000,
            systemfunction = systemfunction)
        br_N1_extinct = do_continuation(u3, p_nt; kargs...)


        ### big prey and predator extinct
        u0 = [.99,0,0]
        u3 = iterate_to_last_pt_scaled(
            u0, p_nt; 
            tf = 10000,
            systemfunction = systemfunction)
        br_N1P_extinct = do_continuation(u3, p_nt; kargs...)

        ### small prey and predator extinct
        u0 = [0,0.9,0]
        u3 = iterate_to_last_pt_scaled(
            u0, p_nt; 
            tf = 10000,
            systemfunction = systemfunction)
        br_N2P_extinct = do_continuation(u3, p_nt; kargs...)

        br_list = (coexist = br_co, predator_extinct = br_P_extinct, 
                N2_extinct = br_N2_extinct, N1_extinct = br_N1_extinct,
                    N1P_extinct = br_N1P_extinct, N2P_extinct = br_N2P_extinct)

        return br_list # actually a named tuple


    end

    

    function diagram_2_recursion_nog(p_nt; lens = (@optic _.scale), 
        p_min = 0.1, p_max = 8.0, 
        systemfunction =system_scaled_nogroups, 
        dsmax = 0.01)
        
        br_list = do_base_continuations_nog(
            p_nt; lens = lens, 
            p_min = p_min, p_max = p_max, 
            systemfunction =systemfunction, 
            dsmax = dsmax)
        extra_branches = []
        for (name, br) in pairs(br_list)
            for (i, sp) in enumerate(br.specialpoint)
                if sp.type == :bp
                    push!(
                        extra_branches, 
                        continuation(
                            br, i; 
                            bothside = true)
                        )
                end
            end
        end
        extra_branches = filter(!isnothing, extra_branches);

        return (br_list, extra_branches)
    end

    function plot_comparison_branches_filtered(
        list_left, list_right,x_axis_symbol; 
        ymax = 2.0, has_g = true, x_max = 5,
        plot_fun = plot_nice_bif)

        if plot_fun == plot_nice_bif_Tg
            args = [:N1] # y_axis symbol
        else
            args = [:N1, x_axis_symbol] # [y_axis_symbol, x_axis_symbol]
            print("hi")
        end

        kargs = (has_g = has_g, x_max = x_max)
        args[1] = :N1
        N1kargs = (kargs..., ylims = [-0.05, 1.0])
        pltN1 = plot_fun(list_left, args...; N1kargs...)
        pltN1_extra = plot_fun(
            list_right, args...; N1kargs...)
        plot!(pltN1, pltN1_extra, ylabel = "N1", xlabel = "")

        args[1] = :N2
        pltN2 = plot_fun(list_left, args...; N1kargs...)
        pltN2_extra = plot_fun(list_right, args...; N1kargs...)
        plot!(pltN2, pltN2_extra, ylabel = "N2", xlabel = "")

        args[1] = :P
        pkargs = (kargs..., ylims = [-0.05, ymax])
        pltp = plot_fun(
            list_left, 
            args...; 
            pkargs...)
        pltp_extra = plot_fun(
            list_right,
            args...; 
            pkargs...)
        plot!(pltp, pltp_extra, ylabel = "P")
        plot(
            pltN1, pltN1_extra, pltN2, pltN2_extra, 
            pltp, pltp_extra, layout = (3,2)
            )
    end

    """
    plot_comparison_branches(br_list, extra_branches, x_axis_symbol; ymax = 2.0)
    - left: branches from base continuations 
    (and any extra branches already decided as unique) 
    - right: branches to compare to
    """
    function plot_comparison_branches(br_list, extra_branches; x_axis_symbol=:param, ymax = 2.0)

        
        pltN1 = plot(br_list..., ylims = [-0.05, 1.01])
        pltN1_extra = plot(
            br_list..., 
            extra_branches..., 
            ylims = [-0.05, 1.01])

        pltN2 = plot(br_list..., vars = (:param, :N2), ylims = [-0.05, 1.01])
        pltN2_extra = plot(
            br_list...,
            extra_branches...,
            vars = (:param, :N2), 
            ylims = [-0.05, 1.01])

        pltp = plot(
            br_list..., 
            vars = (:param, :P), 
            ylims = [-0.01, ymax])
        pltp_extra = plot(
            br_list..., 
            extra_branches..., 
            vars = (:param, :P), 
            ylims = [-0.01, ymax])

        plot(
            pltN1, pltN1_extra, pltN2, pltN2_extra, 
            pltp, pltp_extra, layout = (3,2)
            )
    end

    

# Functions that work for both
    function simulate_and_plot(
        u0, p_nt; tf = 500, system = fullsystem_scaled!,
        start_ind = nothing)
        tspan = (0.0, tf)
        prob = ODEProblem(system, u0, tspan, p_nt)
        sol = solve(prob)    

        # Extract N1 and N2 from the solution
        N1 = sol[1, :]  # First component of the solution
        N2 = sol[2, :]  # Second component of the solution
        # Plot N1 vs N2
        plot1 = plot(N1, N2, xlabel="N1", ylabel="N2", title="N1 vs N2", legend=false, 
                xlims = (0,1), ylims = (0,1), color = :black)
        if start_ind === nothing
            start_ind = round(Int,length(N1)/5)
        end
        plot!(N1[start_ind:start_ind+1], N2[start_ind:start_ind+1], 
                arrow=true, linewidth=2,  legend=false, color = :black)

        plot2 = plot(sol.t, [N1, N2], label=["N1" "N2"], 
            xlabel="Time", ylabel="Population", title="N1 and N2 over Time")
        # Add an arrow to indicate direction
        # Choose a point and the next point to determine the arrow's direction

        fullplot = plot(plot1, plot2, layout = (2,1))
        output = (sol, fullplot)
        return output
    end
    function plot_segments!(x_vec, y_vec, stability_vec; 
        color = :black, plot_legend = true)
        """
        Plots line segments based on stability.

        # Arguments
        - `x_vec`: A vector of x-coordinates.
        - `y_vec`: A vector of y-coordinates.
        - `stability_vec`: A vector of booleans indicating stability for each point.
                        `true` means stable, `false` means unstable.

        # Returns
        - A plot with solid lines for stable segments and dashed lines for unstable segments.
        """
        #plt = plot()  # Initialize the plot

        # Initialize temporary vectors for segments
        x_segment = []
        y_segment = []
        current_stability = stability_vec[1]  # Start with the stability of the first point

        # Track whether the labels have been added to the legend
        stable_label_added = false
        unstable_label_added = false
        for i in 1:length(stability_vec)
            # Add the current point to the segment
            push!(x_segment, x_vec[i])
            push!(y_segment, y_vec[i])

            # Check if stability changes or if it's the last point
            if i == length(stability_vec) || stability_vec[i] != current_stability
                # Plot the current segment
                linestyle = current_stability ? :solid : :dash
                linewidth = current_stability ? 6 : 2
                plot!(x_segment, y_segment, linestyle=linestyle, linewidth = linewidth, label = "",
                        legend = false, color = color)

                # Start a new segment
                x_segment = [x_vec[i]]
                y_segment = [y_vec[i]]
                current_stability = stability_vec[i]
            end
        end
        if plot_legend
            # manually add legend
            plot!([], [], linestyle=:solid, color=:black, label="stable", legend = true)
            plot!([], [], linestyle=:dash, color=:black, label="unstable", legend = true)
        end

    end
    function diagram_2_recursion(p_nt; lens = (@optic _.scale), 
        p_min = 0.1, p_max = 8.0, 
        systemfunction =system_scaled_nogroups, 
        dsmax = 0.01)
        
        if systemfunction == system_scaled_nogroups
            br_list = do_base_continuations_nog(
                p_nt; lens = lens, 
                p_min = p_min, p_max = p_max, 
                systemfunction =systemfunction, 
                dsmax = dsmax)
        else
            br_list = do_base_continuations(
                p_nt, p_nt.x_max; lens = lens, p_min = p_min, 
                p_max = p_max,systemfunction = systemfunction,
                dsmax = dsmax)
        end
        extra_branches = []
        for (name, br) in pairs(br_list)
            for (i, sp) in enumerate(br.specialpoint)
                if sp.type == :bp
                    push!(
                        extra_branches, 
                        continuation(
                            br, i; 
                            bothside = true)
                        )
                end
            end
        end
        extra_branches = filter(!isnothing, extra_branches);

        return (br_list, extra_branches)
    end
    function plot_nice_bif(br_list, y_axis_symbol, x_axis_symbol; 
        x_max = 5,
        has_g = true,
        colorblind_palette = [:blue, :green, 
        :purple, :pink, :brown, :gray, :orange, :magenta, :black],
        ylims = nothing)
        
        """
        Plots branches with a specified y-axis variable.

        # Arguments
        - `br_list`: A list of branches to plot.
        - `y_axis_symbol`: The symbol representing the y-axis variable (e.g., `:N1`, `:N2`).
        - `colorblind_palette`: A palette of colors for the branches (default is colorblind-friendly).

        # Returns
        - A plot of the branches with the specified y-axis variable.
        """
        fun_filter_mat = has_g ? extract_branch_matrix : extract_branch_matrix_nog
        plt = plot()
        colorblind_palette = [colorblind_palette..., # doing this allows us to handle more branches
        colorblind_palette..., colorblind_palette...]
        for (j, branch) in enumerate(br_list)
            print()
            out_nt = fun_filter_mat(branch, x_max)
            if !isempty(out_nt.param)  # Check if out_nt.param is not empty
                plot_segments!(out_nt.param, getfield(out_nt, y_axis_symbol), Bool.(out_nt.stable); 
                            plot_legend = false, color = colorblind_palette[j])
            end
        end
        # Add legend for stability
        #plot!([], [], linestyle=:solid, color=:black, label="stable", legend=true)
        #plot!([], [], linestyle=:dash, color=:black, label="unstable", legend=true)
        plot!(xlabel = "\n"*param_label_dic[x_axis_symbol],
                ylabel = ylabel_dic[y_axis_symbol], bottommargin=6mm)
        if ylims != nothing
            plot!(ylims = ylims)
        end
        plot!(plt,size = (600,480))
        return plt

    end
    function make_and_save_nice_plots(
        br_list, fn_string_base; param_key = :scale, has_g = true, 
        P_lims = nothing
        )

        bif_fig_path = "/Users/taliaborofsky/Documents/CH_GroupFormation/CH_manuscript/FIgures/BifurcationDiagrams/"
        pltP = plot_nice_bif(
            br_list, :P, param_key, has_g = has_g,
            ylims = P_lims)
        if P_lims !== nothing
            plot!(pltP, ylims=P_lims)
        end
        savefig(pltP, bif_fig_path*"p_"*fn_string_base*".pdf")

        pltN1 = plot_nice_bif(
            br_list, :N1, param_key, has_g = has_g
            )

        savefig(
            pltN1, bif_fig_path*"N1_"*fn_string_base*".pdf"
            )

        pltN2 = plot_nice_bif(
            br_list, :N2, param_key, has_g = has_g
            )
        savefig(
            pltN2, bif_fig_path*"N2_"*fn_string_base*".pdf"
            )

        pltNsum = plot_nice_bif(
            br_list, :Nsum, param_key, has_g = has_g
            )
        savefig(pltNsum, bif_fig_path*"Nsum_"*fn_string_base*".pdf")

        if has_g
            print("hi")
            pltxbar = plot_nice_bif(
                br_list, :mean_x, param_key, has_g = has_g
                )
            plot!(ylims=[0.9,4.6])
            savefig(pltxbar, bif_fig_path*"meanx_"*fn_string_base*".pdf")
            return (P=pltP, N1 = pltN1, N2 = pltN2, Nsum = pltNsum, mean_x = pltxbar)
        end

        return (P=pltP, N1 = pltN1, N2 = pltN2, Nsum = pltNsum)

    end

    

    function iterate_to_last_pt_scaled(
        u0, p_nt; 
        tf = 500, systemfunction = fullsystem_scaled!
        )
        tspan = (0.0, tf)
        prob = ODEProblem(systemfunction, u0, tspan, p_nt)
        sol = solve(prob)
        return sol[:,end]
    end
    function do_continuation(u0, p_nt; 
        systemfunction = fullsystem_scaled, lens = (@optic _.scale),
        p_min = 1.01, p_max = 8.0, dsmax = 0.01)

        # lens has to be in@optic form (whatever that is). 
        # it comes from Accesors.jl package
        # _ is the placeholder for the object, which is set to p_nt
        if systemfunction == system_scaled_nogroups
            record_from_solution = recordFromSolution_nog
        else
            record_from_solution = recordFromSolution
        end
        prob = BifurcationProblem(systemfunction, u0, p_nt,
            # specify the continuation parameter)
            lens, record_from_solution = record_from_solution)
        
        opts_br = ContinuationPar(
            # parameter interval
            p_min = p_min, p_max = p_max, dsmax = dsmax,
            # detect bifurcations with bisection method
            # we increase the precision of the bisection
            n_inversion = 8 )
        br = continuation(prob, PALC(), opts_br;
            # we want to compute both sides of the branch of the initial
            # value of E0 = -2
            bothside = true)
    end
end