
import Random

include("../src/structures.jl")
include("../src/cascade_tools.jl")
include("../src/network_tools.jl")
include("../src/lagrange_dmp_method.jl")
include("../src/gradient_dmp_method.jl")


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
    out_neighbors, in_neighbors = dir_neighbors_from_edges(edgelist, n)

    g = DirGraph(n, m, edgelist, out_neighbors, in_neighbors)

    # Generate Cascades

    M = 1000
    T = 4

    observed = collect(1:n)

    cascades = zeros(Int64, n, M)
    for i in 1:M
        s = rand(1:n)
        p0 = zeros(Float64, n)
        p0[s] = 1.0

        temp_cascades = cascade_ic(g, p0, T)
        cascades[1:n, i] = times_from_cascade(temp_cascades)
    end
    cascades_classes = preprocess_cascades(cascades)

    # DMP-Rec Inference

    threshold = 1e-6
    max_iter = 1000
    iter_threshold = 400

    g_temp = DirGraph(n, m, edgelist_from_array(edges, repeat([0.5], size(edges)[1])),
                   out_neighbors, in_neighbors)

    ratio = 1.0
    multiplier = n / M / T / 80.0
    iter = 0
    while (abs(ratio) > threshold) & (iter < max_iter)
        D, objective_old = get_dmp_gradient(cascades_classes, g_temp, T, observed)

        for (e, v) in g_temp.edgelist
            step = D[e] * multiplier
            while ((v + step) <= 0.0) | ((v + step) >= 1.0)
                step /= 2.0
            end
            g_temp.edgelist[e] = v + step
        end
        objective_new = get_full_objective(cascades_classes, g_temp, T)
        ratio = (objective_new - objective_old) / abs(objective_old)

        iter += 1
        if iter > iter_threshold
            multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
        end
    end

    # Comparison with true values

    average_error_on_alphas = sum(abs.(values(merge(-, g.edgelist, g_temp.edgelist)))) / m

    println("For M = ", M, ", the average error on alphas is equal to ", average_error_on_alphas)
end

main()
