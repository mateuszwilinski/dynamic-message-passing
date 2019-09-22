using DelimitedFiles
using StatsBase

include("src/structures.jl")
include("src/functions.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "ba" end
    n = try parse(Int64, ARGS[2]) catch e 200 end  # number of nodes
    k = try parse(Int64, ARGS[3]) catch e 1 end
    d = try parse(Int64, ARGS[4]) catch e 1 end  # number of unobserved nodes
    T = try parse(Int64, ARGS[5]) catch e 20 end
    M = try parse(Int64, ARGS[6]) catch e 20 end  # number of cascades

    for f in 1:10
        # Generating IC model
        edges = readdlm(string("data/networks/", network_type, "_", k, "_", n, "_", f, ".csv"), ' ', Int64) .+ 1
        edge_weights = rand(size(edges)[1])
        edgelist = edgelist_from_array(edges, edge_weights)
        m = length(edgelist)
        neighbors = neighbors_from_edges(edgelist, n)
        g = Graph(n, m, edgelist, neighbors)

        # Generating cascades
        cascades = Array{Array{UInt8,2},1}()

        unobserved = sample(1:n, d, replace=false)
        observed = filter(!in(unobserved), 1:n)
        for _ in 1:M
            s = rand(observed)
            p0 = zeros(Float64, n)
            p0[s] = 1.0
            append!(cascades, [cascade(g, p0, T)])
            cascades[end][:, unobserved] .= 0
        end

        # Inference
        temp_weights = rand(size(edges)[1])
        g_temp =  Graph(n, m, edgelist_from_array(edges, temp_weights), neighbors)

        difference = 1.0
        iter = 0
        threshold = 0.005
        max_iter = 300

        println("description;", network_type, ";", k, ";", n, ";", M, ";", d, ";", f)
        println("old_version")
        multiplier = 1.0
        cascades_classes = preprocess_cascades(cascades)
        while (difference > threshold) & (iter < max_iter)
            D, objective = get_gradient(cascades_classes, g_temp, T)

            iter += 1
            difference = 0.0
            for (e, v) in g_temp.edgelist
                gamma = v * (1.0 - v) * multiplier
                step = (D[e] / M) * gamma
                new_v = v - step
                while (new_v < 0.0) | (new_v > 1.0)
                    multiplier /= 2.0
                    gamma = v * (1.0 - v) * multiplier
                    step = (D[e] / M) * gamma
                    new_v = v - step
                end
                g_temp.edgelist[e] = new_v
                difference += abs(step)
            end
            difference /= sum(abs.(values(g_temp.edgelist)))
            println(iter, ";", difference, ";", objective)
        end

        # Printing results
        diff = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / sum(abs.(values(g.edgelist)))
        _, obj = get_gradient(cascades, g)
        _, obj_temp = get_gradient(cascades, g_temp)
        println("results;", diff, ";", obj, ";", obj_temp)
        println("new_version")
        g_temp =  Graph(n, m, edgelist_from_array(edges, temp_weights), neighbors)

        difference = 1.0
        iter = 0
        iter_threshold = 30
        multiplier = 1.0
        while (difference > threshold) & (iter < max_iter)
            D, objective = get_gradient(cascades_classes, g_temp, T)

            iter += 1
            difference = 0.0
            for (e, v) in g_temp.edgelist
                gamma = v * (1.0 - v) * multiplier
                step = (D[e] / M) * gamma
                new_v = v - step
                while (new_v < 0.0) | (new_v > 1.0)
                    multiplier /= 2.0
                    gamma = v * (1.0 - v) * multiplier
                    step = (D[e] / M) * gamma
                    new_v = v - step
                end
                g_temp.edgelist[e] = new_v
                difference += abs(step)
            end
            difference /= sum(abs.(values(g_temp.edgelist)))
            println(iter, ";", difference, ";", objective)
            if iter > iter_threshold
                multiplier *= (iter - iter_threshold) / (iter + 1 - iter_threshold)
            end
        end

        # Printing results
        diff = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / sum(abs.(values(g.edgelist)))
        _, obj = get_gradient(cascades, g)
        _, obj_temp = get_gradient(cascades, g_temp)
        println("results;", diff, ";", obj, ";", obj_temp)

    end
end

main()
