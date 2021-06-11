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
    n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    k = try parse(Int64, ARGS[3]) catch e 3 end  # degree for rr (er) or 'm' for bb
    T = try parse(Int64, ARGS[4]) catch e 20 end  # cascade time
    M = try parse(Int64, ARGS[5]) catch e 20 end  # number of cascades
    d = try parse(Int64, ARGS[6]) catch e 1 end  # number of unobserved nodes
    f_end = try parse(Int64, ARGS[7]) catch e 10 end  # number of files
    s = try parse(Int64, ARGS[8]) catch e 1 end  # number of seeds

    threshold = 1e-8
    max_iter = 700
    iter_threshold = 200

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
            seed = sample(observed, s, replace=false)
            p0 = zeros(Float64, n)
            p0[seed] .= 1.0
            temp_cascades = cascade_ic(g, p0, T)
            temp_cascades[1:end, unobserved] .= 0
            cascades[1:n, i] = times_from_cascade(temp_cascades)
        end
        cascades_classes = preprocess_cascades(cascades)

        # Inference
        g_temp = Graph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])), neighbors)

        difference = 1.0
        multiplier = n / M / T / 80.0
        iter = 0
        objective_old = 0.0
        objective_new = 0.0
        # while (abs(difference) > threshold) & (iter < max_iter)
        for _ in 1:max_iter
            D, objective_old = get_lagrange_gradient(cascades_classes, g_temp, T)

            difference = 0.0
            for (e, v) in g_temp.edgelist
                step = D[e] * multiplier
                while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                    step /= 2.0
                end
                g_temp.edgelist[e] = v - step
            end
            objective_new = get_full_objective(cascades_classes, g_temp, T)
            difference = (objective_new - objective_old) / abs(objective_old)

            iter += 1
            if iter > iter_threshold
                multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
            end
        end

        # Printing results
        diff_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
        obj = get_full_objective(cascades_classes, g, T)
        obj_temp = get_full_objective(cascades_classes, g_temp, T)

        leaves = find_unobserved_leaves(g, unobserved)
        for leave in leaves
            g.edgelist[leave] = 0.0
            g_temp.edgelist[leave] = 0.0
        end
        diff_alpha_leaves = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / (m - length(leaves))

        println("compare_alphas;", network_type, ";", k, ";", n, ";", M, ";", T, ";", d, ";",
        f, ";", s, ";", iter, ";", diff_alpha, ";", diff_alpha_leaves, ";", obj, ";", obj_temp)
    end
end

main()
