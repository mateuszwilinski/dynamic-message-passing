using DelimitedFiles
using StatsBase

include("src/structures.jl")
include("src/functions.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "ba" end
    n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    k = try parse(Int64, ARGS[3]) catch e 1 end  # degree for rr (er) or 'm' for bb
    T = try parse(Int64, ARGS[4]) catch e 20 end  # cascade time
    M = try parse(Int64, ARGS[5]) catch e 20 end  # number of cascades
    d = try parse(Int64, ARGS[6]) catch e 1 end  # number of unobserved nodes
    f_end = try parse(Int64, ARGS[7]) catch e 10 end  # number of files

    threshold = 0.001
    max_iter = 200
    iter_threshold = 35

    unobserved = [1:d;]
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
        g_temp =  Graph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])), neighbors)

        difference = 1.0
        multiplier = 1.0
        iter = 0
        cascades_classes = preprocess_cascades(cascades)
        while (difference > threshold) & (iter < max_iter)
            D, objective = get_gradient(cascades_classes, g_temp, T, unobserved)

            iter += 1
            difference = 0.0
            for (e, v) in g_temp.edgelist
                gamma = v * (1.0 - v) * multiplier
                step = (D[e] / M) * gamma
                new_v = v - step
                while (new_v < 0.0) | (new_v > 1.0)
                    gamma /= 2.0
#                     multiplier /= 2.0  # test
#                     gamma = v * (1.0 - v) * multiplier  # test
                    step = (D[e] / M) * gamma
                    new_v = v - step
                end
                g_temp.edgelist[e] = new_v
                difference += abs(step)
            end
            difference /= sum(abs.(values(g_temp.edgelist)))
            if iter > iter_threshold
                multiplier *= (iter - iter_threshold) / (iter + 1 - iter_threshold)
            end
        end

        # Printing results
        diff_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
        _, obj = get_gradient(cascades_classes, g, T, unobserved)
        _, obj_temp = get_gradient(cascades_classes, g_temp, T, unobserved)

        println("compare_alphas_new;", network_type, ";", k, ";", n, ";", M, ";", T, ";", d, ";",
        f, ";", iter, ";", diff_alpha, ";", obj, ";", obj_temp)
    end
end

main()
