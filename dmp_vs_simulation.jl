using DelimitedFiles

include("src/structures.jl")
include("src/functions.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "ba" end
    sim_n = try parse(Int64, ARGS[2]) catch e 1000 end
    n = try parse(Int64, ARGS[3]) catch e 200 end
    k = try parse(Int64, ARGS[4]) catch e 1 end
    r = try parse(Int64, ARGS[5]) catch e 1 end
    s = try parse(Int64, ARGS[6]) catch e rand(1:n) end

    for f in 1:10
        # Loading network
        edges = readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", f, ".csv"), ' ', Int64) .+ 1
        p0 = zeros(Float64, n)
        p0[s] = 1.0
        T = 100

        for _ in 1:r
            # Generating IC model
            edge_weights = rand(size(edges)[1])

            edgelist = edgelist_from_array(edges, edge_weights)
            m = length(edgelist)
            neighbors = neighbors_from_edges(edgelist, n)

            g = Graph(n, m, edgelist, neighbors)

            # Dynamic Message Passing
            marginals, messages = dynamic_messsage_passing(g, p0, T)

            # Simulation
            temp = zeros(Int64, T, n)
            for i in 1:sim_n
                temp .+= (cascade(g, p0, T) .> 0)
            end

            # Printing results
            println("d_vs_s;", network_type, ";", k, ";", n, ";", sim_n, ";", s, ";", f, ";",
            sum(abs.(marginals - temp ./ sim_n)) / sum(marginals))
        end
    end
end

main()
