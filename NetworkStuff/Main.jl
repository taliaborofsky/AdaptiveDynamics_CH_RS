#!/usr/bin/env julia

### Code for Borofsky et al, 2023. Based on code from Akcay et al. 2018 Nature.
### Executable file. Type "./Main.jl --help" in shell for usage. Note for mac that before doing ./Main.jl, need to run 'chmod + x Main.jl' (only need to do this once)

include("MainFunctions.jl")

using ArgParse
using JLD



### Defining the parameters, default values, and help text.
function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--pn"
        help = "probability of inheriting connections"
        arg_type = Float64
        default = 0.5
    "--pr"
        help = "probability of random connections"
        arg_type = Float64
        default = 0.01
    "--r0"
        help = "initial normalized density of small prey"
        arg_type = Float64
        default = 1.0
    "--R0"
        help = "initial normalized density of big prey"
        arg_type = Float64
        default = 1.0
    "--netsize"
        help = "network size"
        arg_type = Int64
        default = 100
    "--generations"
        help = "number of generations to iterate network. Total time-steps given by n*gen"
        arg_type = Int64
        default = 100
    "--br"
        help = "benefit from catching small prey"
        arg_type = Float64
        default = 0.2
    "--bR"
        help = "benefit from catching big prey. Must be larger than 2*br"
        arg_type = Float64
        default = 1.0
    "--betar"
        help = "depletion constant of small prey"
        arg_type = Float64
        default = 0.5
    "--betaR"
        help = "depletion constant of big prey"
        arg_type = Float64
        default = 0.5
    "--H"
        help = "horizontal learning"
        arg_type = Float64
        default = 0.01
    "--file"
        help = "name of the file to save things in"
        arg_type = String
        default = "output"
    "--retint"
        help = "Interval at which the simulation saves output, in terms of number of death/birth events. Default = 0 reverts to saving things at intervals equal to network size" # so retint = 0--> save once every generation
        arg_type = Int64
        default = 0
    "--replicates"
        help = "number of replicates to run for the same parameter value"
        arg_type = Int64
        default = 1
    "--saveall"
        help = "whether to save all trajectories, or calculate the mean across all replicates"
        arg_type = Bool
        default = true
    "--networksaveint"
        help = "by which frequency (when data is saved) the simulation saves the full adjacency matric and type vectors. Integer; default is 0, in which case these are never saved."
        arg_type = Int64
        default = 0
  end
  return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    sim_args = copy(parsed_args)
    # remove "file" and "saveall" from sim_args because not used till after do sim
    delete!(sim_args, "file")
    delete!(sim_args, "saveall")
    sim_params = Dict()
    for (arg, val) in sim_args
        sim_params[Symbol(arg)] = val
    end
    params = NetworkParam(;sim_params...)
    
    print(params.pn)
    if params.networksaveint == 0
        simfun = runSim
    else
        simfun = runSimNetSave
    end
    print(params.replicates)
    if parsed_args["saveall"] == true
        for i in 1:params.replicates
            typehist, degreehist, payoffhist, netw, types = simfun(params)
        end
    end
    
    
    #=
    if parsed_args["saveall"] == true
      for i in 1:params.replicates
            typehist, degreehist, payoffhist, netw, types = simfun(params)
          for i in 1:params.replicates
          typehist, pnhist, prhist, degreehist, payoffhist, finnet, fintype, pnshist, prshist = simfun(;pn=params.pn, pr=params.pr, netsize=params.netsize, generations=params.generations, b=params.b, c=params.c, d=params.d, mu=params.mu, evollink=params.evollink, mulink=params.mulink, sigmapn=params.sigmapn, sigmapr=params.sigmapr, clink=params.clink, retint=params.retint, funnoevollink=funnoevollinkin, funevollink=funevollinkin, delta=params.delta,payfun=payfunin,netsaveint=params.networksaveint)

          file = ismatch(r"\.jl", parsed_args["file"]) ? parsed_args["file"] : parsed_args["file"]*"-"*lpad(string(i), length(digits(params.replicates)), "0")*".jld"

          save(file, "params", params, "typehist", typehist, "pn", pnhist, "pr", prhist, "degree", degreehist, "payoff", payoffhist, "network", finnet, "types", fintype, "pns", pnshist, "prs", prshist)
      end
    =#
end

main()