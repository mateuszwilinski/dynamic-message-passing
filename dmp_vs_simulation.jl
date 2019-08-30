using DelimitedFiles

include("src/structures.jl")
include("src/functions.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "ba" end
    sim_n = try parse(Int64, ARGS[2]) catch e 1000 end
    N = try parse(Int64, ARGS[3]) catch e 200 end
    k = try parse(Int64, ARGS[4]) catch e 1 end
    s = try parse(Int64, ARGS[5]) catch e 1 end
    f = try parse(Int64, ARGS[5]) catch e 1 end
    r = try parse(Int64, ARGS[5]) catch e 1 end

    # Loading network
    if network_type == "ba"
        edges = readdlm(string("data/networks/ba_", k, "_200_", f, ".csv"), ' ', Int64) .+ 1
    elseif network_type == "er"
        edges = readdlm(string("data/networks/er_", k, "_200_", f, ".csv"), ' ', Int64) .+ 1
    end

    # Generating cascade model
    weights = rand(size(edges)[1])
    n = maximum(edges)

    edgelist = edgelist_from_array(edges, weights)
    m = length(edgelist)
    neighbors = neighbors_from_edges(edgelist, n)

    g = Graph(n, m, edgelist, neighbors)

    p0 = zeros(n)
    p0[s] = 1.0
    T = 100

    # Dynamic Message Passing
    marginals, messages = dynamic_messsage_passing(g, p0, T)

    # Simulation
    temp = zeros(Int64, T, n)
    for i in 1:sim_n
        temp .+= (cascade(g, p0, T) .> 0)
    end

    # Printing results
    println("d_vs_s;", network_type, ";", k, ";", N, ";", sim_n, ";", s, ";",
    sum(abs.(marginals - temp ./ sim_n)) / sum(marginals))
end

main()
