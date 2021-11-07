
import Random

include("../src/structures.jl")
include("../src/cascade_tools.jl")
include("../src/network_tools.jl")
include("../src/maximum_likelihood_method.jl")


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

    # MLE Inference

    threshold = 1e-6
    max_iter = 1000

    likelihood_edgelist = max_likelihood_params(g, cascades, T,
                                                threshold, max_iter)

    # Comparison with true values

    average_error_on_alphas = sum(abs.(values(merge(-, g.edgelist, likelihood_edgelist)))) / m

    println("For M = ", M, ", the average error on alphas is equal to ", average_error_on_alphas)
end

main()
