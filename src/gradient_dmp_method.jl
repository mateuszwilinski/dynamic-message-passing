
include("dynamic_message_passing.jl")

"""
    messages_derivatives(g, messages, p0, T)

Compute the derivatives over messages with respect to spreading parameters
"""
function messages_derivatives(g::DirGraph, messages::Dict{Array{Int64,1}, Array{Float64,1}},
                              p0::Array{Float64, 1}, T::Int64)
    phi = Dict{Array{Int64,1}, Dict{Array{Int64,1}, Array{Float64,1}}}()
    for (k, v) in messages
        phi[k] = Dict{Array{Int64,1}, Array{Float64,1}}()
        for edge in keys(g.edgelist)
            phi[k][edge] = zeros(Float64, T)
        end
    end

    for t in 2:T
        for (k, v) in messages
            temp_prod = ones(Float64, length(g.in_neighbors[k[1]]))
            for (n, neighbor) in enumerate(g.in_neighbors[k[1]])
                if neighbor != k[2]  # so that k[2] is not taken into account 'x' lines below
                    temp_prod[n] = 1.0 - g.edgelist[Int64[neighbor, k[1]]] * messages[Int64[neighbor, k[1]]][t-1]
                end
            end
            for edge in keys(g.edgelist)
                for (n, neighbor) in enumerate(g.in_neighbors[k[1]])
                    if neighbor != k[2]
                        temp_sum = g.edgelist[Int64[neighbor, k[1]]] * phi[Int64[neighbor, k[1]]][edge][t-1]
                        if Int64[neighbor, k[1]] == edge
                            temp_sum += messages[edge][t-1]
                        end
                        phi[k][edge][t] += temp_sum * prod(temp_prod[1:end .!= n])
                    end
                end
                phi[k][edge][t] *= (1.0 - p0[k[1]])
            end
        end
    end
    return phi
end

"""
    part_derivative!(dm, i, g, t, messages, phi, p0, action)

Compute part of the marginal derivative with respect to spreading parameters
"""
function part_derivative!(dm::Dict{Array{Int64,1}, Float64}, i::Int64, g::DirGraph,
                          t::Int64, messages::Dict{Array{Int64,1}, Array{Float64,1}},
                          phi::Dict{Array{Int64,1}, Dict{Array{Int64,1}, Array{Float64,1}}},
                          p0::Array{Float64, 1}, action::Char)
    temp_prod = ones(Float64, length(g.in_neighbors[i]))
    for (n, neighbor) in enumerate(g.in_neighbors[i])
        temp_prod[n] = 1.0 - g.edgelist[Int64[neighbor, i]] * messages[Int64[neighbor, i]][t]
    end
    for edge in keys(g.edgelist)
        if !haskey(dm, edge)
            dm[edge] = 0.0
        end
        for (n, neighbor) in enumerate(g.in_neighbors[i])
            temp_sum = g.edgelist[Int64[neighbor, i]] * phi[Int64[neighbor, i]][edge][t]
            if Int64[neighbor, i] == edge
                temp_sum += messages[edge][t]
            end
            if action == '+'
                dm[edge] += temp_sum * prod(temp_prod[1:end .!= n]) * (1.0 - p0[i])
            elseif action == '-'
                dm[edge] -= temp_sum * prod(temp_prod[1:end .!= n]) * (1.0 - p0[i])
            end
        end
    end
end

"""
    marginal_derivatives(i, g, messages, phi, p0, tau)

Compute the marginal derivative with respect to spreading parameters
"""
function marginal_derivatives(i::Int64, g::DirGraph, messages::Dict{Array{Int64,1}, Array{Float64,1}},
                              phi::Dict{Array{Int64,1}, Dict{Array{Int64,1}, Array{Float64,1}}},
                              p0::Array{Float64, 1}, tau::Int64, T::Int64)
    dm = Dict{Array{Int64,1}, Float64}()
    if tau == 1
        for edge in keys(g.edgelist)
            dm[edge] = 0.0
        end
    elseif tau > T
        throw(ArgumentError("Parameter 'tau' is greater than 'T'."))
    else
        if (tau > 1) & (tau < T)
            part_derivative!(dm, i, g, tau-1, messages, phi, p0, '+')
        end
        if (tau > 2) & (tau <= T)
            part_derivative!(dm, i, g, tau-2, messages, phi, p0, '-')
        end
    end
    return dm
end

"""
    get_dmp_gradient(cascades_classes, g, T, observed)

Calculates gradient for alphas according to lagrange derivative summed over classes of cascades
"""
function get_dmp_gradient(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                          g::DirGraph, T::Int64, observed::Array{Int64, 1})
    objective = 0.0
    D_ij = Dict{Array{Int64, 1}, Float64}()
    for edge in keys(g.edgelist)
        D_ij[edge] = 0.0
    end

    for seeds in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seeds] .= 1.0
        marginals, messages = dmp_ic(g, p0, T)
        phi = messages_derivatives(g, messages, p0, T)

        objective += get_ic_objective(marginals, cascades_classes[seeds])
        for i in observed
            for tau_i in keys(cascades_classes[seeds][i])
                m_i = 0.0
                if tau_i < T
                    m_i += marginals[tau_i, i]
                end
                if tau_i > 1
                    m_i -= marginals[tau_i-1, i]
                end
                if tau_i == T
                    m_i += 1.0
                end
                c_i = cascades_classes[seeds][i][tau_i]
                dm_i = marginal_derivatives(i, g, messages, phi, p0, tau_i, T)
                for edge in keys(g.edgelist)
                    D_ij[edge] += c_i * dm_i[edge] / m_i
                end
            end
        end
    end
    return D_ij, objective
end
