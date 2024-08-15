
include("dynamic_message_passing.jl")

"""
    lambda_from_marginals(marginals, seed)

Computes marginals lagrange coefficients for a given initial condition (seed).
"""
function lambda_from_marginals(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}})
    T::Int64 = size(marginals)[1]
    n::Int64 = size(marginals)[2]
    lambda = zeros(Float64, T, n)
    for i in keys(seed)
        for t in keys(seed[i])
            if t == 1  # I assume that T > 1 (!!!)
                lambda[t, i] -= seed[i][t] / marginals[t, i]
            elseif t == T
                if marginals[t-1, i] < 1.0
                    lambda[t-1, i] += seed[i][t] / (1.0 - marginals[t-1, i])
                end
            elseif 1 < t < T
                if marginals[t, i] - marginals[t-1, i] > 1e-9
                    lambda[t, i] -= seed[i][t] / (marginals[t, i] - marginals[t-1, i])
                    lambda[t-1, i] += seed[i][t] / (marginals[t, i] - marginals[t-1, i])
                end
            end
        end
    end
    return lambda
end

"""
    lambda_from_marginals(marginals, seed, noise)

Computes marginals lagrange coefficients for a given initial condition (seed).
"""
function lambda_from_marginals(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}},
                               noise::TimeNoise)
    T::Int64 = size(marginals)[1]
    n::Int64 = size(marginals)[2]
    lambda = zeros(Float64, T, n)
    for i in keys(seed)
        for t in keys(seed[i])
            temp_sum = 0.0
            for k in noise.m_1:noise.m_2
                if t - k == 1
                    temp_sum += noise.p[k] * marginals[t-k, i]
                elseif t - k == T
                    temp_sum += noise.p[k] * (1.0 - marginals[t-1-k, i])
                elseif 1 < t - k < T
                    temp_sum += noise.p[k] * (marginals[t-k, i] - marginals[t-1-k, i])
                end
            end  # TODO: What if temp_sum == 0.0 ??
            for k in noise.m_1:noise.m_2
                if 1 <= t - k < T
                    lambda[t-k, i] -= noise.p[k] * seed[i][t] / temp_sum
                end
                if 1 < t - k <= T
                    lambda[t-1-k, i] += noise.p[k] * seed[i][t] / temp_sum
                end
            end
        end
    end
    return lambda
end

"""
    lambda_from_marginals(marginals, seed, unobs_times)

Computes marginals lagrange coefficients for a given initial condition (seed)
and assuming partly unobserved times.
"""
function lambda_from_marginals(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}},
                               unobs_times::Array{Int64, 1})
    T::Int64 = size(marginals)[1]
    n::Int64 = size(marginals)[2]
    lambda = zeros(Float64, T, n)
    for i in keys(seed)
        for t in keys(seed[i])
            if t == 1  # I assume that T > 1 (!!!)
                lambda[t, i] -= seed[i][t] / marginals[t, i]
            elseif t == T
                if marginals[t-1, i] < 1.0
                    lambda[t-1, i] += seed[i][t] / (1.0 - marginals[t-1, i])
                end
            elseif 1 < t < T
                if t in unobs_times
                    t_low, t_upp = unobserved_time_interval(t, unobs_times)
                else
                    t_low, t_upp = t, t
                end
                if marginals[t_upp, i] - marginals[t_low-1, i] > 1e-9
                    lambda[t_upp, i] -= seed[i][t] / (marginals[t_upp, i] - marginals[t_low-1, i])
                    lambda[t_low-1, i] += seed[i][t] / (marginals[t_upp, i] - marginals[t_low-1, i])
                end
            end
        end
    end
    return lambda
end

"""
    get_lambda_ij(lambda, g, messages, p0)

Calculates messages lagrange coefficients.
"""
function get_lambda_ij(lambda::Array{Float64, 2}, g::Graph,
                       messages::Dict{Array{Int64, 1}, Array{Float64, 1}}, p0::Array{Float64, 1})
    # initialising lambdas
    T = size(lambda)[1]
    lambda_j = zeros(Float64, T, g.n)
    lambda_ij = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        lambda_ij[edge] = zeros(Float64, T)
        lambda_ij[reverse(edge)] = zeros(Float64, T)
    end

    # computing lambdas
    for t in T-1:-1:1
        for j in 1:g.n
            for neighbor in g.neighbors[j]
                lambda_j[t, j] += (lambda_ij[Int64[j, neighbor]][t+1] *
                                   (1.0 - messages[Int64[j, neighbor]][t+1]))
            end
        end
        for e in keys(lambda_ij)
            temp_prob = 1.0 - g.edgelist[sort(e)] * messages[e][t]
            if temp_prob == 0.0
                lambda_ij[e][t] = get_lambda_ij_hard_way(lambda_ij, messages, p0,
                                                         g, lambda[t+1, e[2]], e, t)
            else
                lambda_ij[e][t] = (g.edgelist[sort(e)] *
                                   (lambda[t+1, e[2]] * (1.0 - messages[reverse(e)][t+1]) +
                                    (lambda_j[t, e[2]] - lambda_ij[reverse(e)][t+1] *
                                     (1.0 - messages[reverse(e)][t+1])) / temp_prob))
            end
        end
    end
    return lambda_ij
end

"""
    get_lambda_ij(lambda, g, marginals, messages, p0)

Calculates messages lagrange coefficients for a directed network.
"""
function get_lambda_ij(lambda::Array{Float64, 2}, g::DirGraph, marginals::Array{Float64, 2},
                       messages::Dict{Array{Int64,1}, Array{Float64,1}}, p0::Array{Float64, 1})
    # initialising lambdas
    T = size(lambda)[1]
    lambda_j = zeros(Float64, T, g.n)
    lambda_ij = Dict{Array{Int64,1}, Array{Float64,1}}()
    for edge in keys(g.edgelist)
        lambda_ij[edge] = zeros(Float64, T)
    end

    # computing lambdas
    for t in T-1:-1:1
        for j in 1:g.n
            for neighbor in g.out_neighbors[j]
                lambda_j[t, j] += (lambda_ij[Int64[j, neighbor]][t+1] *
                                   (1.0 - messages[Int64[j, neighbor]][t+1]))
            end
        end
        for e in keys(lambda_ij)
            temp_prob = 1.0 - g.edgelist[e] * messages[e][t]
            if temp_prob == 0.0
                lambda_ij[e][t] = get_lambda_ij_hard_way(lambda_ij, messages, p0,
                                                         g, lambda[t+1, e[2]], e, t)
            else
                if haskey(g.edgelist, reverse(e))
                    lambda_ij[e][t] = (g.edgelist[e] *
                                       (lambda[t+1, e[2]] * (1.0 - messages[reverse(e)][t+1]) +
                                        (lambda_j[t, e[2]] - lambda_ij[reverse(e)][t+1] *
                                         (1.0 - messages[reverse(e)][t+1])) / temp_prob))
                else  # TODO: this may require extra checking
                    lambda_ij[e][t] = (g.edgelist[e] *
                                       (lambda[t+1, e[2]] * (1.0 - marginals[t+1, e[2]]) +
                                        lambda_j[t, e[2]]) / temp_prob)
                end
            end
        end
    end
    return lambda_ij
end

"""
    get_lambda_ij(lambda, g, marginals, messages, p0)

Calculates messages lagrange coefficients for a simple graph.
"""
function get_lambda_ij(lambda::Array{Float64, 2}, g::SimpleGraph,
                       messages::Dict{Array{Int64, 1}, Array{Float64, 1}}, p0::Array{Float64, 1})
    # initialising lambdas
    T = size(lambda)[1]
    lambda_j = zeros(Float64, T, g.n)
    lambda_ij = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in eachrow(g.edgelist)
        lambda_ij[edge] = zeros(Float64, T)
        lambda_ij[reverse(edge)] = zeros(Float64, T)
    end

    # computing lambdas
    for t in T-1:-1:1
        for j in 1:g.n
            for neighbor in g.neighbors[j]
                lambda_j[t, j] += (lambda_ij[Int64[j, neighbor]][t+1] *
                                   (1.0 - messages[Int64[j, neighbor]][t+1]))
            end
        end
        for e in keys(lambda_ij)
            temp_prob = 1.0 - g.alpha[] * messages[e][t]
            if temp_prob == 0.0
                lambda_ij[e][t] = get_lambda_ij_hard_way(lambda_ij, messages, p0,
                                                         g, lambda[t+1, e[2]], e, t)
            else
                lambda_ij[e][t] = (g.alpha[] *
                                   (lambda[t+1, e[2]] * (1.0 - messages[reverse(e)][t+1]) +
                                    (lambda_j[t, e[2]] - lambda_ij[reverse(e)][t+1] *
                                     (1.0 - messages[reverse(e)][t+1])) / temp_prob))
            end
        end
    end
    return lambda_ij
end

"""
    get_lambda_ij_hard_way(lambda_ij, messages, p0, g, lambda_ti, e, t)

Computes the lambda_ij for edge 'e' at time 't', when the original implementation is indefinite.
"""
function get_lambda_ij_hard_way(lambda_ij::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                messages::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                p0::Array{Float64, 1}, g::Graph, lambda_ti::Float64,
                                e::Array{Int64, 1}, t::Int64)
    n_neighbors = size(g.neighbors[e[2]])[1]
    temp_j = lambda_ti * g.edgelist[sort(e)] * (1.0 - p0[e[2]])
    temp_ij = repeat([g.edgelist[sort(e)] * (1.0 - p0[e[2]])], n_neighbors)
    for k in 1:n_neighbors
        if g.neighbors[e[2]][k] != e[1]
            temp = (1.0 - g.edgelist[sort([g.neighbors[e[2]][k], e[2]])] *
                    messages[[g.neighbors[e[2]][k], e[2]]][t])
            temp_j *= temp
            temp_ij[1:end .!= k] *= temp
            temp_ij[k] *= lambda_ij[[e[2], g.neighbors[e[2]][k]]][t+1]
        else
            temp_ij[k] = 0.0
        end
    end
    return temp_j + sum(temp_ij)
end

"""
    get_lambda_ij_hard_way(lambda_ij, messages, p0, g, lambda_ti, e, t)

Computes the lambda_ij for edge 'e' at time 't', when the original implementation is indefinite
for a directed network.
"""
function get_lambda_ij_hard_way(lambda_ij::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                messages::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                p0::Array{Float64, 1}, g::DirGraph, lambda_ti::Float64,
                                e::Array{Int64, 1}, t::Int64)
    m_neighbors = size(g.in_neighbors[e[2]])[1]
    k_neighbors = size(g.out_neighbors[e[2]])[1]
    temp_j = lambda_ti * g.edgelist[e] * (1.0 - p0[e[2]])
    temp_ij = repeat([g.edgelist[e] * (1.0 - p0[e[2]])], n_neighbors)

    # the loops below correspond to equation (A7)
    for m in 1:m_neighbors
        if g.in_neighbors[e[2]][m] != e[1]
            temp_j *= (1.0 - g.edgelist[[g.in_neighbors[e[2]][m], e[2]]] *
                        messages[[g.in_neighbors[e[2]][m], e[2]]][t])
        end
    end
    for k in 1:k_neighbors
        if g.out_neighbors[e[2]][k] != e[1]
            temp_ij *= lambda_ij[[e[2], g.out_neighbors[e[2]][k]]][t+1]
            for m in 1:m_neighbors
                if !(g.in_neighbors[e[2]][m] in [e[1], k])
                    temp_ij *= (1.0 - g.edgelist[[g.in_neighbors[e[2]][m], e[2]]] *
                                messages[[g.in_neighbors[e[2]][m], e[2]]][t])
                end
            end
        end
    end
    return temp_j + sum(temp_ij)
end

"""
    get_lambda_ij_hard_way(lambda_ij, messages, p0, g, lambda_ti, e, t)

Computes the lambda_ij for edge 'e' at time 't', when the original implementation is indefinite
for a simple graph.
"""
function get_lambda_ij_hard_way(lambda_ij::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                messages::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                p0::Array{Float64, 1}, g::SimpleGraph, lambda_ti::Float64,
                                e::Array{Int64, 1}, t::Int64)
    n_neighbors = size(g.neighbors[e[2]])[1]
    temp_j = lambda_ti * g.alpha[] * (1.0 - p0[e[2]])
    temp_ij = repeat([g.alpha[] * (1.0 - p0[e[2]])], n_neighbors)
    for k in 1:n_neighbors
        if g.neighbors[e[2]][k] != e[1]
            temp = (1.0 - g.alpha[] *
                    messages[[g.neighbors[e[2]][k], e[2]]][t])
            temp_j *= temp
            temp_ij[1:end .!= k] *= temp
            temp_ij[k] *= lambda_ij[[e[2], g.neighbors[e[2]][k]]][t+1]
        else
            temp_ij[k] = 0.0
        end
    end
    return temp_j + sum(temp_ij)
end

"""
    get_lagrange_gradient(cascades_classes, g, T)

Computes gradient for alphas according to lagrange derivative summed over classes of cascades.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::Graph, T::Int64)
    objective = 0.0
    D_ij = Dict{Array{Int64, 1}, Float64}()
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds])
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds])
        for (edge, v) in g.edgelist
            if !haskey(D_ij, edge)
                if v == 0.0
                    D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                       lambda, lambda_ij, g, T)
                else
                    D_ij[edge] = sum(lambda_ij[edge] .* messages[edge] +
                                     lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            else
                if v == 0.0
                    D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                        lambda, lambda_ij, g, T)
                else
                    D_ij[edge] += sum(lambda_ij[edge] .* messages[edge] +
                                      lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_classes, g, T)

Computes gradient for alphas according to lagrange derivative summed over classes of cascades
in the case of directed graph.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::DirGraph, T::Int64)
    objective = 0.0
    D_ij = Dict{Array{Int64, 1}, Float64}()
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds])
        lambda_ij = get_lambda_ij(lambda, g, marginals, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds])
        for (edge, v) in g.edgelist
            if !haskey(D_ij, edge)
                if v == 0.0
                    D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                       lambda, lambda_ij, g, T)
                else
                    D_ij[edge] = sum(lambda_ij[edge] .* messages[edge]) / v
                end
            else
                if v == 0.0
                    D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                        lambda, lambda_ij, g, T)
                else
                    D_ij[edge] += sum(lambda_ij[edge] .* messages[edge]) / v
                end
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_classes, g, T)

Computes gradient for alphas according to lagrange derivative summed over classes of cascades
in the case of simple graph.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::SimpleGraph, T::Int64)
    objective = 0.0
    D = 0.0
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds])
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds])
        # TODO: One could ask what to do when alpha == 0.0, but it seems to me that
        #       in the case of simple graphs such situation is trivial and either
        #.      there is no dynamics, or you can start from a very small alpha.
        for edge in eachrow(g.edgelist)
            D += sum(lambda_ij[edge] .* messages[edge] +
                     lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / g.alpha[]
        end
    end
    return D, objective
end

"""
    get_lagrange_gradient(cascades_class, p0, g, T)

Computes gradient for alphas according to lagrange derivative when there is only one cascades
class (corresponding to stochastic initial condition p0).
"""
function get_lagrange_gradient(cascades_class::Dict{Int64, Dict{Int64, Int64}}, p0::Array{Float64, 1},
                               g::Graph, T::Int64)
    D_ij = Dict{Array{Int64, 1}, Float64}()
    marginals, messages = dmp_ic(g, p0, T)
    lambda = lambda_from_marginals(marginals, cascades_class)
    lambda_ij = get_lambda_ij(lambda, g, messages, p0)

    objective = get_ic_objective(marginals, cascades_class)
    for (edge, v) in g.edgelist
        if !haskey(D_ij, edge)
            if v == 0.0
                D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                   lambda, lambda_ij, g, T)
            else
                D_ij[edge] = sum(lambda_ij[edge] .* messages[edge] +
                                 lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
            end
        else
            if v == 0.0
                D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                    lambda, lambda_ij, g, T)
            else
                D_ij[edge] += sum(lambda_ij[edge] .* messages[edge] +
                                  lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_class, p0, g, T)

Computes gradient for alphas according to lagrange derivative when there is only one cascades
class (corresponding to stochastic initial condition p0).
"""
function get_lagrange_gradient(cascades_class::Dict{Int64, Dict{Int64, Int64}}, p0::Array{Float64, 1},
                               g::DirGraph, T::Int64)
    D_ij = Dict{Array{Int64, 1}, Float64}()
    marginals, messages = dmp_ic(g, p0, T)
    lambda = lambda_from_marginals(marginals, cascades_class)
    lambda_ij = get_lambda_ij(lambda, g, marginals, messages, p0)

    objective = get_ic_objective(marginals, cascades_class)
    for (edge, v) in g.edgelist
        if !haskey(D_ij, edge)
            if v == 0.0
                D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                   lambda, lambda_ij, g, T)
            else
                D_ij[edge] = sum(lambda_ij[edge] .* messages[edge]) / v
            end
        else
            if v == 0.0
                D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                    lambda, lambda_ij, g, T)
            else
                D_ij[edge] += sum(lambda_ij[edge] .* messages[edge]) / v
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_classes, g, T, noise)

Computes gradient for alphas according to lagrange derivative summed over classes of cascades,
in the case of noisy times.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::Graph, T::Int64, noise::TimeNoise)
    objective = 0.0
    D_ij = Dict{Array{Int64, 1}, Float64}()
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds], noise)
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds], noise)
        for (edge, v) in g.edgelist
            if !haskey(D_ij, edge)
                if v == 0.0
                    D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                       lambda, lambda_ij, g, T)
                else
                    D_ij[edge] = sum(lambda_ij[edge] .* messages[edge] +
                                     lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            else
                if v == 0.0
                    D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                        lambda, lambda_ij, g, T)
                else
                    D_ij[edge] += sum(lambda_ij[edge] .* messages[edge] +
                                      lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_classes, g, T, unobs_times)

Computes gradient for alphas according to lagrange derivative summed over classes of cascades
and assuming partly unobserved times.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::Graph, T::Int64, unobs_times::Array{Int64, 1})
    objective = 0.0
    D_ij = Dict{Array{Int64, 1}, Float64}()
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds], unobs_times)
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds], unobs_times)
        for (edge, v) in g.edgelist
            if !haskey(D_ij, edge)
                if v == 0.0
                    D_ij[edge] = get_gradient_hard_way(edge, p0, messages,
                                                       lambda, lambda_ij, g, T)
                else
                    D_ij[edge] = sum(lambda_ij[edge] .* messages[edge] +
                                     lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            else
                if v == 0.0
                    D_ij[edge] += get_gradient_hard_way(edge, p0, messages,
                                                        lambda, lambda_ij, g, T)
                else
                    D_ij[edge] += sum(lambda_ij[edge] .* messages[edge] +
                                      lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
                end
            end
        end
    end
    return D_ij, objective
end

"""
    get_lagrange_gradient(cascades_classes, g, T, unobs_times)

Computes gradient for alpha according to lagrange derivative summed over classes of cascades
and assuming partly unobserved times.
"""
function get_lagrange_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g::SimpleGraph, T::Int64, unobs_times::Array{Int64, 1})
    objective = 0.0
    D = 0.0
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seeds], unobs_times)
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_ic_objective(marginals, cascades_classes[seeds], unobs_times)
        # TODO: One could ask what to do when alpha == 0.0, but it seems to me that
        #       in the case of simple graphs such situation is trivial and either
        #.      there is no dynamics, or you can start from a very small alpha.
        for edge in eachrow(g.edgelist)
            D += sum(lambda_ij[edge] .* messages[edge] +
                     lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / g.alpha[]
        end
    end
    return D, objective
end

"""
    get_gradient_hard_way(edge, p0, messages, lambda, lambda_ij, g, T)

Computes gradient for a single edge (for a given cascade class).
"""
function get_gradient_hard_way(edge::Array{Int64,1}, p0::Array{Float64, 1},
        messages::Dict{Array{Int64,1}, Array{Float64,1}}, lambda::Array{Float64, 2},
        lambda_ij::Dict{Array{Int64,1}, Array{Float64,1}}, g::Graph, T::Int64)
    D_edge = 0.0
    i_neighbors = size(g.neighbors[edge[1]])[1]
    j_neighbors = size(g.neighbors[edge[2]])[1]
    for t in 2:T
        temp_i = lambda[t, edge[1]] * messages[[edge[2], edge[1]]][t-1] * (1.0 - p0[edge[1]])
        temp_j = lambda[t, edge[2]] * messages[[edge[1], edge[2]]][t-1] * (1.0 - p0[edge[2]])
        temp_ij = repeat([messages[[edge[1], edge[2]]][t-1] * (1.0 - p0[edge[2]])], j_neighbors)
        temp_ji = repeat([messages[[edge[2], edge[1]]][t-1] * (1.0 - p0[edge[1]])], i_neighbors)
        for k in 1:i_neighbors
            if g.neighbors[edge[1]][k] != edge[2]
                temp = (1.0 - g.edgelist[sort([g.neighbors[edge[1]][k], edge[1]])] *
                        messages[[g.neighbors[edge[1]][k], edge[1]]][t-1])
                temp_i *= temp
                temp_ji[1:end .!= k] *= temp
                temp_ji[k] *= lambda_ij[[edge[1], g.neighbors[edge[1]][k]]][t]
            else
                temp_ji[k] = 0.0
            end
        end
        for k in 1:j_neighbors
            if g.neighbors[edge[2]][k] != edge[1]
                temp = (1.0 - g.edgelist[sort([g.neighbors[edge[2]][k], edge[2]])] *
                        messages[[g.neighbors[edge[2]][k], edge[2]]][t-1])
                temp_j *= temp
                temp_ij[1:end .!= k] *= temp
                temp_ij[k] *= lambda_ij[[edge[2], g.neighbors[edge[2]][k]]][t]
            else
                temp_ij[k] = 0.0
            end
        end
        D_edge += temp_j + sum(temp_ij) + temp_i + sum(temp_ji)
    end
    return D_edge
end

"""
    get_gradient_hard_way(edge, p0, messages, lambda, lambda_ij, g, T)

Computes gradient for a single edge (for a given cascade class) in case
of a directed network.
"""
function get_gradient_hard_way(edge::Array{Int64,1}, p0::Array{Float64, 1},
        messages::Dict{Array{Int64,1}, Array{Float64,1}}, lambda::Array{Float64, 2},
        lambda_ij::Dict{Array{Int64,1}, Array{Float64,1}}, g::DirGraph, T::Int64)
    D_edge = 0.0
    j_neighbors = size(g.in_neighbors[edge[2]])[1]
    for t in 2:T
        temp_j = lambda[t, edge[2]] * messages[edge][t-1] * (1.0 - p0[edge[2]])
        temp_ij = repeat([messages[edge][t-1] * (1.0 - p0[edge[2]])], j_neighbors)
        for k in 1:j_neighbors
            if g.in_neighbors[edge[2]][k] != edge[1]
                temp = (1.0 - g.edgelist[[g.in_neighbors[edge[2]][k], edge[2]]] *
                        messages[[g.in_neighbors[edge[2]][k], edge[2]]][t-1])
                temp_j *= temp
                temp_ij[1:end .!= k] *= temp
                temp_ij[k] *= lambda_ij[[edge[2], g.in_neighbors[edge[2]][k]]][t]
            else
                temp_ij[k] = 0.0
            end
        end
        D_edge += temp_j + sum(temp_ij)
    end
    return D_edge
end

"""
    get_full_objective(cascades_classes, g, T)

Calculates the objective of a given graph, with respect to a set of cascades.
"""
function get_full_objective(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                            g::Union{Graph, DirGraph, SimpleGraph}, T::Int64)
    objective = 0.0
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, _ = dmp_ic(g, p0, T)
        objective += get_ic_objective(marginals, cascades_classes[seeds])
    end
    return objective
end

"""
    get_full_objective(cascades_class, p0, g, T)

Calculates the objective of a given graph, with respect to a set of cascades, assuming
that all the cascades have common stochastic initial condition p0.
"""
function get_full_objective(cascades_class::Dict{Int64, Dict{Int64, Int64}}, p0::Array{Float64, 1},
                            g::Union{Graph, DirGraph, SimpleGraph}, T::Int64)
    marginals, _ = dmp_ic(g, p0, T)
    objective = get_ic_objective(marginals, cascades_class)
    return objective
end

"""
    get_full_objective(cascades_class, g, T, noise)

Calculates the objective of a given graph, with respect to a set of cascades,
in the case of noisy times.
"""
function get_full_objective(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                            g::Union{Graph, DirGraph}, T::Int64, noise::TimeNoise)
    objective = 0.0
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, _ = dmp_ic(g, p0, T)
        objective += get_ic_objective(marginals, cascades_classes[seeds], noise)
    end
    return objective
end

"""
    get_full_objective(cascades_classes, g, T, unobs_times)

Calculates the objective of a given graph, with respect to a set of cascades.
"""
function get_full_objective(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                            g::Union{Graph, DirGraph, SimpleGraph}, T::Int64, unobs_times::Array{Int64, 1})
    objective = 0.0
    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, _ = dmp_ic(g, p0, T)
        objective += get_ic_objective(marginals, cascades_classes[seeds], unobs_times)
    end
    return objective
end
