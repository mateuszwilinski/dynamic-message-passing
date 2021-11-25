
using Random
using StatsBase

include("../src/structures.jl")
include("../src/cascade_tools.jl")
include("../src/network_tools.jl")
include("../src/lagrange_dmp_method.jl")

function main()
    # Specify a Random Seed

    seed = 7

    Random.seed!(seed)

    # Create a Network

    edges = Int64[[1 2]; [1 3]; [1 4]; [1 5]; [1 6]; [1 7]; [2 8]; [5 9]; [5 10]]
    m = size(edges)[1]
    n = maximum(edges)
    edge_weights = rand(size(edges)[1])

    edgelist = edgelist_from_array(edges, edge_weights)
    neighbors = neighbors_from_edges(edgelist, n)

    g = Graph(n, m, edgelist, neighbors)

    # Unobserved Nodes

    d = 0

    unobserved = sample(1:n, d, replace=false)
    observed = filter(!in(unobserved), 1:n)

    # Generate Cascades

    M = 1000
    T = 4

    cascades = zeros(Int64, n, M)
    for i in 1:M
        s = rand(1:n)
        p0 = zeros(Float64, n)
        p0[s] = 1.0

        temp_cascades = cascade_ic(g, p0, T)
        cascades[1:n, i] = times_from_cascade(temp_cascades)
    end
    cascades_classes = preprocess_cascades(cascades)
    remove_unobserved!(cascades_classes, unobserved)

    # SLICER Inference

    threshold = 1e-6
    max_iter = 1000
    iter_threshold = 400

    g_temp = Graph(n, n * (n-1) / 2,
                   full_edgelist(n, repeat([0.5], Int64(n * (n-1) / 2))),
                   full_neighbors(n))

    ratio = 1.0
    multiplier = n / M / T / 80.0
    iter = 0
    while (abs(ratio) > threshold) & (iter < max_iter)
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

        iter += 1
        if iter > iter_threshold
            multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
        end
    end

    # Comparison with true structure

    edge_threshold = 1e-7
    false_positive = 0
    false_negative = 0
    for edge in keys(g_temp.edgelist)
        if edge in keys(g.edgelist)
            if g_temp.edgelist[edge] < edge_threshold
                false_negative += 1
            end
        else
            if g_temp.edgelist[edge] >= edge_threshold
                false_positive += 1
            end
        end
    end

    println("For M = ", M, ", number of false negative is equal to ",
            false_negative, ", number of false positive is equal to ",
            false_positive)
end

main()
