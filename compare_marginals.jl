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
    s = try parse(Int64, ARGS[6]) catch e 1 end  # cascade source
    f_end = try parse(Int64, ARGS[7]) catch e 10 end  # number of files

    threshold = 0.001
    max_iter = 200
    iter_threshold = 35

    unobserved = Array{Int64, 1}()
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
        p0 = zeros(Float64, n)
        p0[s] = 1.0

        alpha_marginals, _ = dynamic_messsage_passing(g, p0, T)
        estimated_marginals, _ = dynamic_messsage_passing(g_temp, p0, T)
        real_marginals = readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", T, "_", f, "_", s, "_marginals.csv"), ';', Float64)

        diff_alpha = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m
        _, obj = get_gradient(cascades_classes, g, T, unobserved)
        _, obj_temp = get_gradient(cascades_classes, g_temp, T, unobserved)

        # TODO: sprawdz roznice marginali (!!!)
        diff_alpha_margins = sum(abs.(alpha_marginals - real_marginals)) / sum(real_marginals)
        diff_margins = sum(abs.(estimated_marginals - real_marginals)) / sum(real_marginals)
#         diff_alpha_margins = sum(abs.(alpha_marginals - real_marginals)) / (n * T)
#         diff_margins = sum(abs.(estimated_marginals - real_marginals)) / (n * T)
#         normalisation = copy(real_marginals)
#         normalisation[normalisation .== 0.0] /= 1.0
#         diff_alpha_margins = sum(abs.(alpha_marginals - real_marginals) ./ normalisation)
#         diff_margins = sum(abs.(estimated_marginals - real_marginals) ./ normalisation)

        println("compare_marginals_relative;", network_type, ";", k, ";", n, ";", M, ";", T, ";",
        s, ";", f, ";", iter, ";", diff_alpha, ";", diff_alpha_margins, ";", diff_margins, ";",
        obj, ";", obj_temp)
    end
end

main()
