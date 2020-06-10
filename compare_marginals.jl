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
    s_end = try parse(Int64, ARGS[7]) catch e 1 end  # cascade source
    f_end = try parse(Int64, ARGS[8]) catch e 10 end  # number of files

    threshold = 0.0001
    max_iter = 200
    iter_threshold = 100

#     unobserved = [1:d;]
    unobserved = [(n-d+1):n;]
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
            s_ = rand(observed)
            p0 = zeros(Float64, n)
            p0[s_] = 1.0
            append!(cascades, [cascade(g, p0, T)])
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
        for s in 1:s_end
            p0 = zeros(Float64, n)
            p0[s] = 1.0

            alpha_marginals, _ = dynamic_messsage_passing(g, p0, T)
            estimated_marginals, _ = dynamic_messsage_passing(g_temp, p0, T)
            real_marginals = readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", T, "_", f, "_", s, "_marginals.csv"), ';', Float64)

            diff_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
            _, obj = get_gradient(cascades_classes, g, T, unobserved)
            _, obj_temp = get_gradient(cascades_classes, g_temp, T, unobserved)

            diff_alpha_margins = sum(abs.(alpha_marginals - real_marginals)) / sum(real_marginals)
            diff_margins = sum(abs.(estimated_marginals - real_marginals)) / sum(real_marginals)
    #         diff_alpha_margins = sum(abs.(alpha_marginals - real_marginals)) / (n * T)
    #         diff_margins = sum(abs.(estimated_marginals - real_marginals)) / (n * T)

            println("compare_marginals_relative;", network_type, ";", k, ";", n, ";", M, ";", T, ";",
            d, ";", s, ";", f, ";", iter, ";", diff_alpha, ";", diff_alpha_margins, ";",
            diff_margins, ";", obj, ";", obj_temp)
        end
    end
end

main()
