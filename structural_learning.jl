
using DelimitedFiles
using StatsBase
using Random

include("src/structures.jl")
include("src/cascade_tools.jl")
include("src/network_tools.jl")
include("src/dynamic_message_passing.jl")
include("src/lagrange_dmp_method.jl")

function main()
    # Parameters
    network_type = try ARGS[1] catch e "rr" end
    n = try parse(Int64, ARGS[2]) catch e 100 end  # number of nodes
    k = try parse(Int64, ARGS[3]) catch e 3 end  # degree for rr (er) or 'm' for bb
    T = try parse(Int64, ARGS[4]) catch e 20 end  # cascade time
    M = try parse(Int64, ARGS[5]) catch e 20 end  # number of cascades
    l = try parse(Int64, ARGS[6]) catch e 16 end  # number of unobserved nodes
    d = try parse(Int64, ARGS[7]) catch e 1 end  # number of unobserved nodes
    f_end = try parse(Int64, ARGS[8]) catch e 10 end  # number of files
    s = try parse(Int64, ARGS[9]) catch e 1 end  # number of seeds

    threshold = 1e-8
    min_val = 0.001
    max_iter = 400
    iter_threshold = 200
    m = Int64(n * k / 2.0)

    Random.seed!(100)

    unobserved = sample(1:n, d, replace=false)
    observed = filter(!in(unobserved), 1:n)

    for f in 1:f_end
        # Generating IC model
        edges = readdlm(string("data/networks/", network_type, "_ext_", k, "_", n, "_", f, ".csv"), ' ', Int64) .+ 1
        edge_weights = readdlm(string("data/networks/", network_type, "_ext_", k, "_", n, "_", f, "_weights.csv"),
                               ' ', Float64)[1:end, 1]
        edgelist = edgelist_from_array(edges[1:m, 1:end], edge_weights[1:m])
        neighbors = neighbors_from_edges(edgelist, n)

        g = Graph(n, m, edgelist_from_array(edges[1:m, 1:end], edge_weights[1:m]), neighbors)

        # Generating cascades
        cascades_times = zeros(Int64, n, M)
        for i in 1:M
            seed = sample(observed, s, replace=false)
            p0 = zeros(Float64, n)
            p0[seed] .= 1.0
            temp_cascades = cascade_ic(g, p0, T)
            temp_cascades[1:end, unobserved] .= 0
            cascades_times[1:n, i] = times_from_cascade(temp_cascades)
        end
        cascades = preprocess_cascades(cascades_times)

        # Inference
        ext_edge_weights = repeat([0.5], size(edges)[1])
        ext_edgelist = edgelist_from_array(edges[1:(m+l), 1:end], ext_edge_weights[1:(m+l)])
        ext_neighbors = neighbors_from_edges(ext_edgelist, n)

        g_temp = Graph(n, m+l, edgelist_from_array(edges[1:(m+l), 1:end],
                                                   ext_edge_weights[1:(m+l)]), ext_neighbors)

        difference = 1.0
        multiplier = n / M / T / 200.0
        iter = 0
        objective_old = 0.0
        objective_new = 0.0
        while (abs(difference) > threshold) & (iter < max_iter)
            D, objective_old = get_lagrange_gradient(cascades, g_temp, T)

            for (e, v) in g_temp.edgelist
                step = D[e] * multiplier
                while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                    step /= 2.0
                end
                g_temp.edgelist[e] = v - step
            end
            objective_new = get_full_objective(cascades, g_temp, T)
            difference = (objective_new - objective_old) / abs(objective_old)

            iter += 1
            if iter > iter_threshold
                multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
            end
        end

        vals = zeros(m+l)
        for i in 1:(m+l)
            e = edges[i, 1:end]
            vals[i] = g_temp.edgelist[e]
        end

        min_real = minimum(vals[1:m])
        max_fake = maximum(vals[(m+1):(m+l)])
        mean_fake = mean(vals[(m+1):(m+l)])
        count_real = sum(vals[1:m] .> min_val)
        count_fake = sum(vals[(m+1):(m+l)] .< min_val)

        println("structure;", network_type, ";", k, ";", n, ";", M, ";", T, ";",
                l, ";", d, ";", f, ";", s, ";", iter, ";", min_real, ";",
                max_fake, ";", mean_fake, ";", count_real, ";", count_fake)
    end
end

main()
