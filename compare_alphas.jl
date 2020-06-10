using DelimitedFiles
using StatsBase

include("src/structures.jl")
include("src/functions.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "rr" end
    n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    k = try parse(Int64, ARGS[3]) catch e 3 end  # degree for rr (er) or 'm' for bb
    T = try parse(Int64, ARGS[4]) catch e 20 end  # cascade time
    M = try parse(Int64, ARGS[5]) catch e 20 end  # number of cascades
    d = try parse(Int64, ARGS[6]) catch e 1 end  # number of unobserved nodes
    f_end = try parse(Int64, ARGS[7]) catch e 10 end  # number of files

    threshold = 0.0001
    max_iter = 200
    iter_threshold = 100

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
        cascades = Array{Array{UInt8,2},1}()

        for _ in 1:M
            s = rand(observed)
            p0 = zeros(Float64, n)
            p0[s] = 1.0
            append!(cascades, [cascade(g, p0, T)])
            cascades[end][:, unobserved] .= 0
        end

        # Inference
        g_temp = Graph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])), neighbors)

        difference = 1.0
        multiplier = n / M / T / 80.0
        iter = 0
        cascades_classes = preprocess_cascades(cascades)
        while (difference > threshold) & (iter < max_iter)
            D, objective = get_gradient(cascades_classes, g_temp, T, unobserved)

            iter += 1
            difference = 0.0
            for (e, v) in g_temp.edgelist
                step = D[e] * multiplier
                new_v = min(1.0, max(0.0, v - step))
                g_temp.edgelist[e] = new_v
                difference += abs(step)
            end
            difference /= sum(abs.(values(g_temp.edgelist)))
            if iter > iter_threshold
                multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
            end
        end

        # Printing results
        diff_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
        _, obj = get_gradient(cascades_classes, g, T, unobserved)
        _, obj_temp = get_gradient(cascades_classes, g_temp, T, unobserved)

        leaves = find_unobserved_leaves(g, unobserved)
        for leave in leaves
            g.edgelist[leave] = 0.0
            g_temp.edgelist[leave] = 0.0
        end

        diff_alpha_leaves = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / (m - length(leaves))

        println("compare_alphas_leave;", network_type, ";", k, ";", n, ";", M, ";", T, ";", d, ";",
        f, ";", iter, ";", diff_alpha, ";", diff_alpha_leaves, ";", obj, ";", obj_temp)
    end
end

main()
