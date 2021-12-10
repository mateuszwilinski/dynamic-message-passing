
include("lagrange_dmp_method.jl")

"""
    lambda_from_mixed_marginals(mixture_marginals, seed, T, n)

Computes mixture marginals lagrange coefficients for a given initial condition (seed)
"""
function lambda_from_mixed_marginals(mixture_marginals::Dict{Int64, Array{Float64, 2}},
                                     seed::Dict{Int64, Dict{Int64, Int64}}, T::Int64, n::Int64)
    marginals = zeros(Float64, T, n)
    L = length(mixture_marginals)
    for k in keys(mixture_marginals)
        marginals += mixture_marginals[k]
    end

    lambda = zeros(Float64, T, n)
    for i in keys(seed)
        for t in keys(seed[i])
            if t == 1  # I assume that T > 1 (!!!)
                lambda[t, i] -= seed[i][t] / marginals[t, i]
            elseif t == T
                if marginals[t-1, i] < L
                    lambda[t-1, i] -= seed[i][t] / (marginals[t-1, i] - L)
                end
            elseif t > 1
                if marginals[t, i] - marginals[t-1, i] > 1e-9
                    lambda[t, i] -= seed[i][t] / (marginals[t, i] - marginals[t-1, i])
                    lambda[t-1, i] -= seed[i][t] / (marginals[t-1, i] - marginals[t, i])
                end
            end
        end
    end
    return lambda
end

"""
    get_mixture_gradient(cascades_classes, g_mixture, T)

Computes gradient for alphas of the mixture model, using lagrange equations
"""
function get_mixture_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                              g_mixture::Dict{Int64, Graph}, T::Int64)
    D_kij = Dict{Int64, Dict{Array{Int64, 1}, Float64}}()
    for k in keys(g_mixture)
        D_kij[k] = Dict{Array{Int64, 1}, Float64}()
    end

    for seed in keys(cascades_classes)
        marginals = Dict{Int64, Array{Float64, 2}}()
        messages = Dict{Int64, Dict{Array{Int64,1}, Array{Float64,1}}}()
        p0 = Dict{Int64, Array{Float64, 1}}()
        for k in keys(g_mixture)
            p0[k] = zeros(Float64, g_mixture[k].n)
            p0[k][seed] .= 1.0
            marginals[k], messages[k] = dmp_ic(g_mixture[k], p0[k], T)
        end
        lambda = lambda_from_mixed_marginals(marginals, cascades_classes[seed], T, g_mixture[1].n)

        for k in keys(g_mixture)
            lambda_ij = get_lambda_ij(lambda, g_mixture[k], messages[k], p0[k])
            for (edge, v) in g_mixture[k].edgelist
                if !haskey(D_kij[k], edge)
                    if v == 0.0
                        D_kij[k][edge] = get_gradient_hard_way(edge, p0[k], messages[k], lambda,
                                                               lambda_ij, g_mixture[k], T)
                    else
                        D_kij[k][edge] = sum(lambda_ij[edge] .* messages[k][edge] +
                                             lambda_ij[reverse(edge)] .* messages[k][reverse(edge)]) / v
                    end
                else
                    if v == 0.0
                        D_kij[k][edge] += get_gradient_hard_way(edge, p0[k], messages[k], lambda,
                                                                lambda_ij, g_mixture[k], T)
                    else
                        D_kij[k][edge] += sum(lambda_ij[edge] .* messages[k][edge] +
                                              lambda_ij[reverse(edge)] .* messages[k][reverse(edge)]) / v
                    end
                end
            end
        end
    end
    return D_kij
end

"""
    get_mixture_marginals(g_mixture, p0, T)

Computes marginals of a mixture model
"""
function get_mixture_marginals(g_mixture::Dict{Int64, Graph}, p0::Array{Float64, 1}, T::Int64)
    n = length(p0)
    marginals = zeros(Float64, T, n)
    for k in keys(g_mixture)
        temp, _ = dmp_ic(g_mixture[k], p0, T)
        marginals += temp
    end
    return marginals
end

"""
    get_mixture_objective(cascades_classes, g_mixture, T, n)

computes the objective of a mixture model
"""
function get_mixture_objective(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                               g_mixture::Dict{Int64, Graph}, T::Int64, n::Int64)
    objective = 0.0
    L = length(g_mixture)
    for seed in keys(cascades_classes)
        p0 = zeros(Float64, n)
        p0[seed] .= 1.0
        marginals = get_mixture_marginals(g_mixture, p0, T)
        for i in keys(cascades_classes[seed])
            for t in keys(cascades_classes[seed][i])
                if t == 1  # I assume that T > 1 (!!!)
                    objective += log(marginals[t, i] / L) * cascades_classes[seed][i][t]
                elseif t == T
                    objective += log(1.0 - marginals[t-1, i] / L) * cascades_classes[seed][i][t]
                elseif t > 1
                    objective += log(marginals[t, i] / L - marginals[t-1, i] / L) * cascades_classes[seed][i][t]
                end
            end
        end
    end
    return objective
end
