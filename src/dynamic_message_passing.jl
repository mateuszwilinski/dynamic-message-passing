
"""
    dmp_ic(g, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with initial condition p0
"""
function dmp_ic(g::Graph, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    marginals = zeros(Float64, T, g.n)
    messages = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        messages[edge] = zeros(Float64, T)
        messages[reverse(edge)] = zeros(Float64, T)
    end

    # initial conditions
    marginals[1, :] = p0
    for (k, v) in messages
        v[1] = p0[k[1]]
    end

    # DMP
    for t in 2:T
        for i in 1:g.n
            marginals[t, i] = 1.0 - p0[i]
            for neighbor in g.neighbors[i]
                marginals[t, i] *= (1.0 - g.edgelist[sort(Int64[neighbor, i])] *
                                    messages[Int64[neighbor, i]][t-1])
            end  # TODO: Przyjrzyj sie ponizszemu, czy nie da sie tego poprawic
            marginals[t, i] = max(marginals[t-1, i], 1.0 - marginals[t, i])  # numerical safeguard
        end
        for (e, v) in messages
            temp_prob = 1.0 - g.edgelist[sort(e)] * messages[Int64[e[2], e[1]]][t-1]
            if temp_prob == 0.0
                v[t] = get_ic_message_hard_way(messages, g, p0, e, t)
            else
                v[t] = 1.0 - (1.0 - marginals[t, e[1]]) / temp_prob
            end  # TODO: Przyjrzyj sie ponizszemu, czy nie da sie tego poprawic
            v[t] = max(v[t], v[t-1])  # numerical safeguard
        end
    end
    return marginals, messages
end

"""
    get_ic_message_hard_way(messages, g, p0, e, t)

Computes the message for edge 'e' at time 't', when the original implementation is indefinite
"""
function get_ic_message_hard_way(messages::Dict{Array{Int64, 1}, Array{Float64, 1}}, g::Graph,
                                 p0::Array{Float64, 1}, e::Array{Int64, 1}, t::Int64)
    message = 1.0 - p0[e[1]]
    for neighbor in g.in_neighbors[e[1]]
        if neighbor != e[2]
            message *= 1.0 - g.edgelist[sort(Int64[neighbor, e[1]])] * messages[Int64[neighbor, e[1]]][t-1]
        end
    end
    return 1.0 - message
end

"""
    dmp_si(g, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with initial condition p0
"""
function dmp_si(g::Graph, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    marginals = zeros(Float64, T, g.n)
    msg_phi = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    msg_theta = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        msg_phi[edge] = zeros(Float64, T)
        msg_phi[reverse(edge)] = zeros(Float64, T)
        msg_theta[edge] = ones(Float64, T)
        msg_theta[reverse(edge)] = ones(Float64, T)
    end

    # initial conditions
    marginals[1, :] = 1.0 .- p0
    for (k, v) in msg_phi
        v[1] = p0[k[1]]
    end

    # DMP
    for t in 2:T
        # first theta
        for (e, v) in msg_theta
            v[t] = v[t-1] - g.edgelist[sort(e)] * msg_phi[e][t-1]
        end
        # then marginals
        for i in 1:g.n
            marginals[t, i] = 1.0 - p0[i]
            for neighbor in g.neighbors[i]
                marginals[t, i] *= msg_theta[Int64[neighbor, i]][t]
            end  # TODO: moze potrzebny jest safeguard jak dla IC?
        end
        # theta phi
        for (e, w) in msg_phi
            w[t] = (1.0 - g.edgelist[sort(e)]) * w[t-1]
            w[t] += marginals[t-1, e[1]] / msg_theta[reverse(e)][t-1]
            w[t] -= marginals[t, e[1]] / msg_theta[reverse(e)][t]
        end
    end

    return marginals, msg_phi, msg_theta
end

"""
    get_ic_objective(marginals, seed)

Caclulates the objective function of given activation times paired with an initial condition (seed)
"""
function get_ic_objective(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}})
    T::Int64 = size(marginals)[1]
    objective::Float64 = 0.0
    for i in keys(seed)
        for t in keys(seed[i])
            if t == 1  # I assume that T > 1 (!!!)
                objective += log(marginals[t, i]) * seed[i][t]
            elseif t == T
                objective += log(1.0 - marginals[t-1, i]) * seed[i][t]
            elseif t > 1
                objective += log(marginals[t, i] - marginals[t-1, i]) * seed[i][t]
            end
        end
    end
    return objective
end
