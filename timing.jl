using DelimitedFiles
using StatsBase

include("src/structures.jl")
include("src/network_tools.jl")
include("src/cascade_tools.jl")
include("src/dynamic_message_passing.jl")
include("src/lagrange_dmp_method.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "rr" end
    # n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    k = try parse(Int64, ARGS[2]) catch e 3 end  # degree for rr (er) or 'm' for bb
    T = try parse(Int64, ARGS[3]) catch e 20 end  # cascade time
    M = try parse(Int64, ARGS[4]) catch e 20 end  # number of cascades
    d = try parse(Int64, ARGS[5]) catch e 1 end  # number of unobserved nodes
    f_end = try parse(Int64, ARGS[6]) catch e 10 end  # number of files

    iter_num = 20

    for n in 50000:10000:100000
        unobserved = sample(1:n, d, replace=false)
        observed = filter(!in(unobserved), 1:n)
        for f in 1:f_end
            # Generating IC model
            edges = readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", f, ".csv"), ' ', Int64) .+ 1
            edge_weights = vec(readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", f, "_weights.csv"), ' ', Float64))
            edgelist = edgelist_from_array(edges, edge_weights)
            m = length(edgelist)
            neighbors = neighbors_from_edges(edgelist, n)
            g = Graph(n, m, edgelist, neighbors)

            # Generating cascades
            cascades = zeros(Int64, n, M)
            for i in 1:M
                s = rand(observed)
                p0 = zeros(Float64, n)
                p0[s] = 1.0
                temp_cascades = cascade_ic(g, p0, T)
                temp_cascades[1:end, unobserved] .= 0
                cascades[1:n, i] = times_from_cascade(temp_cascades)
            end
            cascades_classes = preprocess_cascades(cascades)

            # Inference
            multiplier = n / M / T / 80.0
            g_temp = Graph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])), neighbors)
            for _ in 1:iter_num
                t_0 = time()
                D, objective = get_lagrange_gradient(cascades_classes, g_temp, T)
                t_1 = time()

                difference = 0.0
                for (e, v) in g_temp.edgelist
                    step = D[e] * multiplier
                    while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                        step /= 2.0
                    end
                    g_temp.edgelist[e] = v - step
                end

                single_time = t_1 - t_0

                # Printing results
                println("timing;", network_type, ";", k, ";", n, ";", M, ";",
                        T, ";", d, ";", f, ";", iter_num, ";", single_time)
            end
        end
    end
end

main()
