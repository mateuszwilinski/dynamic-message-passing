
using JuMP, Ipopt

include("structures.jl")
include("cascade_tools.jl")
include("network_tools.jl")

"""
    max_likelihood_params(g, cascades, T, tol, max_iter)

Returns edgelist with weights, obtained by maximum likelihood optimisation.
"""
function max_likelihood_params(g::Graph, cascades::Array{Int64, 2}, T::Int64,
                               tol::Float64, max_iter::Int64)
    edges = permutedims(hcat(collect(keys(g.edgelist))...))

    function cascades_likelihood(alphas...)
        g_temp = Graph(g.n, g.m, edgelist_from_array(edges, collect(alphas)),
                       g.neighbors)
        n_c = size(cascades)[2]
        p_c = ones(Real, n_c)
        for i in 1:n_c
            p_c[i] = cascade_likelihood(cascades[1:g.n, i], g_temp, T)
        end
        return sum(p_c) / n_c
    end

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
                                            "max_iter" => max_iter,
                                            "print_level" => 0))
    register(model, :cascades_likelihood, g.m,
             cascades_likelihood, autodiff=true)
    @variable(model, 0.0 <= x[1:g.m] <= 1.0)
    @NLobjective(model, Max, cascades_likelihood(x...))
    optimize!(model)

    return edgelist_from_array(edges, value.(x))
end

"""
    max_likelihood_params(g, cascades, T, tol, max_iter)

Returns edgelist with weights, obtained by maximum likelihood optimisation.
"""
function max_likelihood_params(g::DirGraph, cascades::Array{Int64, 2}, T::Int64,
                               tol::Float64, max_iter::Int64)
    edges = permutedims(hcat(collect(keys(g.edgelist))...))

    function cascades_likelihood(alphas...)
        g_temp = Graph(g.n, g.m, edgelist_from_array(edges, collect(alphas)),
                       g.out_neighbors, g.in_neighbors)
        n_c = size(cascades)[2]
        p_c = ones(Real, n_c)
        for i in 1:n_c
            p_c[i] = cascade_likelihood(cascades[1:g.n, i], g_temp, T)
        end
        return sum(p_c) / n_c
    end

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
                                            "max_iter" => max_iter,
                                            "print_level" => 0))
    register(model, :cascades_likelihood, g.m,
             cascades_likelihood, autodiff=true)
    @variable(model, 0.0 <= x[1:g.m] <= 1.0)
    @NLobjective(model, Max, cascades_likelihood(x...))
    optimize!(model)

    return edgelist_from_array(edges, value.(x))
end

"""
    loc_max_likelihood_params(node, g, cascades, T, tol, max_iter)

Optimise only node neighborhood using maximum likelihood.
"""
function loc_max_likelihood_params(node::Int64, g::Graph,
                                   cascades::Array{Int64, 2}, T::Int64,
                                   tol::Float64, max_iter::Int64)
    l = length(g.neighbors[node])
    local_edges = sort(Array{Int64, 2}([g.neighbors[node] repeat([node], l)]), dims=2)

    function local_cascades_likelihood(alphas...)
        for (k, neighbor) in enumerate(g.neighbors[node])
            g.edgelist[sort([neighbor, node])] = alphas[k]
        end
        n_c = size(cascades)[2]
        p_c = ones(Real, n_c)
        for i in 1:n_c
            p_c[i] = log(local_cascade_likelihood(node, cascades[1:n, i], g, T))
        end
        return sum(p_c) / n_c
    end

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
                                            "max_iter" => max_iter,
                                            "print_level" => 0))
    register(model, :local_cascades_likelihood, l,
             local_cascades_likelihood, autodiff=true)
    @variable(model, 0.0 <= x[1:l] <= 1.0)
    @NLobjective(model, Max, local_cascades_likelihood(x...))
    optimize!(model)

    return edgelist_from_array(local_edges, value.(x))
end

"""
    loc_max_likelihood_params(node, g, cascades, T, tol, max_iter)

Optimise only node neighborhood using maximum likelihood.
"""
function loc_max_likelihood_params(node::Int64, g::DirGraph,
                                   cascades::Array{Int64, 2}, T::Int64,
                                   tol::Float64, max_iter::Int64)
    l = length(g.in_neighbors[node])
    local_edges = Array{Int64, 2}([g.in_neighbors[node] repeat([node], l)])

    function local_cascades_likelihood(alphas...)
        for (k, neighbor) in enumerate(g.in_neighbors[node])
            g.edgelist[[neighbor, node]] = alphas[k]
        end
        n_c = size(cascades)[2]
        p_c = ones(Real, n_c)
        for i in 1:n_c
            p_c[i] = log(local_cascade_likelihood(node, cascades[1:n, i], g, T))
        end
        return sum(p_c) / n_c
    end

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
                                            "max_iter" => max_iter,
                                            "print_level" => 0))
    register(model, :local_cascades_likelihood, l,
             local_cascades_likelihood, autodiff=true)
    @variable(model, 0.0 <= x[1:l] <= 1.0)
    @NLobjective(model, Max, local_cascades_likelihood(x...))
    optimize!(model)

    return edgelist_from_array(local_edges, value.(x))
end
