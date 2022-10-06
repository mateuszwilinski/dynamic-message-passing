
using Random
using StatsBase
using DelimitedFiles

include("src/structures.jl")
include("src/cascade_tools.jl")
include("src/network_tools.jl")
include("src/lagrange_dmp_method.jl")


function main()
    # Parameters
    network_name = try ARGS[1] catch e "ba_3_100" end
    n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    T = try parse(Int64, ARGS[3]) catch e 20 end  # cascade length
    M = try parse(Int64, ARGS[4]) catch e 20 end  # number of cascades
    ts = try ARGS[5] catch e "2,3" end  # unobserved times
    f_end = try parse(Int64, ARGS[6]) catch e 5 end  # number of files

    # threshold = 1e-11
    max_iter = 1000
    iter_threshold = 400

    # Specify a Random Seed
    seed = 7

    Random.seed!(seed)

    unobserved_times = parse.(Int64, split(ts, ","))

    for f in 1:f_end
        Random.seed!(seed)

        # Generating IC model
        edges = readdlm(string("data/networks/", network_name, "_", f, ".csv"), ' ', Int64) .+ 1
        edge_weights = readdlm(string("data/networks/", network_name, "_", f, "_weights.csv"),
                               ' ', Float64)[1:end, 1]
        edgelist = edgelist_from_array(edges, edge_weights)

        m = size(edges)[1]
        neighbors = neighbors_from_edges(edgelist, n)
        g = Graph(n, m, edgelist, neighbors)

        # Generate Cascades
        cascades = zeros(Int64, n, M)
        for i in 1:M
            s = rand(1:n)
            p0 = zeros(Float64, n)
            p0[s] = 1.0

            temp_cascades = times_from_cascade(cascade_ic(g, p0, T))
            cascades[1:n, i] = temp_cascades
        end
        cascades_classes = preprocess_cascades(cascades)
        remove_times!(cascades_classes, unobserved_times)

        # SLICER Inference
        g_temp = Graph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])),
                       neighbors)

        ratio = 1.0
        multiplier = n / M / T / 120.0
        iter = 0
        # while (abs(ratio) > threshold) & (iter < max_iter)
        for iter in 1:max_iter
            D, objective_old = get_lagrange_gradient(cascades_classes, g_temp, T)

            for (e, v) in g_temp.edgelist
                step = D[e] * multiplier
                while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                    step /= 2.0
                end
                g_temp.edgelist[e] = v - step
            end
            objective_new = get_full_objective(cascades_classes, g_temp, T)
            ratio = (objective_new - objective_old) / abs(objective_old)

            # iter += 1
            if iter > iter_threshold
                multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
            end
        end

        # Comparison with true values
        error_on_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
        final_objective = get_full_objective(cascades_classes, g_temp, T)

        println("slicer;", network_name, ";", n, ";", M, ";", T, ";", ts, ";", f, ";",
                iter, ";", error_on_alpha, ";", ratio, ";", final_objective)
    end
end

main()
