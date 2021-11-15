
import Random

include("../src/structures.jl")
include("../src/cascade_tools.jl")
include("../src/network_tools.jl")
include("../src/mixture_model_tools.jl")
include("../src/dynamic_message_passing.jl")


function main()
    # Specify a Random Seed

    seed = 7

    Random.seed!(seed)

    # Create a Network

    # In order to see the effect of this approach, we need a loopy graph
    edges = Int64[[1 2]; [2 3]; [3 4]; [1 4]; [1 5]; [2 6]; [3 7]; [4 8];
                  [5 6]; [6 7]; [7 8]; [5 8]; [5 9]; [6 10]; [7 11]; [8 12];
                  [9 10]; [10 11]; [11 12]; [9 12]; [1 9]; [2 10]; [3 11]; [4 12]]
    m = size(edges)[1]
    n = maximum(edges)
    edge_weights = rand(size(edges)[1])

    edgelist = edgelist_from_array(edges, edge_weights)
    neighbors = neighbors_from_edges(edgelist, n)

    g = Graph(n, m, edgelist, neighbors)

    # Generate Cascades

    M = 10000
    T = 20  # long time is needed for the loops to have significant effect

    cascades = zeros(Int64, n, M)
    for i in 1:M
        s = rand(1:n)
        p0 = zeros(Float64, n)
        p0[s] = 1.0

        temp_cascades = cascade_ic(g, p0, T)
        cascades[1:n, i] = times_from_cascade(temp_cascades)
    end
    cascades_classes = preprocess_cascades(cascades)

    # SLICER Mixture Inference

    threshold = 1e-6
    max_iter = 1000
    iter_threshold = 400
    number_of_layers = 2

    gs = Dict{Int64, Graph}()
    for l in 1:number_of_layers
        gs[l] = Graph(n, m, edgelist_from_array(edges, rand(size(edges)[1])),
                      neighbors)
    end

    objective_old = get_mixture_objective(cascades_classes, gs, T, n)

    ratio = 1.0
    multiplier = n / M / T / 80.0
    iter = 0
    while (abs(ratio) > threshold) & (iter < max_iter)
        D = get_mixture_gradient(cascades_classes, gs, T)

        difference = 0.0
        for k in keys(gs)
            for (e, v) in gs[k].edgelist
                step = D[k][e] * multiplier
                while ((v - step) <= 0.0) | ((v - step) >= 1.0)
                    step /= 2.0
                end
                gs[k].edgelist[e] = v - step
            end
        end
        objective_new = get_mixture_objective(cascades_classes, gs, T, n)
        ratio = (objective_new - objective_old) / abs(objective_old)
        objective_old = objective_new

        iter += 1
        if iter > iter_threshold
            multiplier *= sqrt(iter - iter_threshold) / sqrt(iter + 1 - iter_threshold)
        end
    end

    # Comparing marginals

    sim_n = 100000

    for s in 1:n
        p0_ = zeros(Float64, n)
        p0_[s] = 1.0

        estimated_marginals = get_mixture_marginals(gs, p0_, T) / number_of_layers

        real_marginals = zeros(Int64, T, n)
        for i in 1:sim_n
            real_marginals .+= (cascade_ic(g, p0_, T) .> 0)
        end
        real_marginals = real_marginals ./ sim_n

        dmp_marginals, _ = dmp_ic(g, p0_, T)

        relative_error_on_marginals = (sum(abs.(estimated_marginals - real_marginals)) /
                                       sum(real_marginals))

        dmp_error_on_marginals = (sum(abs.(dmp_marginals - real_marginals)) /
                                  sum(real_marginals))

        println("For M = ", M, " and source = ", s,
                ", the relative error on marginals is equal to ",
                relative_error_on_marginals, "
                The realtive error with true parameters and DMP: ",
                dmp_error_on_marginals)
    end
end

main()
