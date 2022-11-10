
include("structures.jl")
include("additional_tools.jl")

"""
    dmp_ic(g, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with initial condition p0.
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
            end  # TODO: can we improve line below?
            marginals[t, i] = max(marginals[t-1, i], 1.0 - marginals[t, i])  # numerical safeguard
        end
        for (e, v) in messages
            temp_prob = 1.0 - g.edgelist[sort(e)] * messages[reverse(e)][t-1]
            if temp_prob == 0.0
                v[t] = get_ic_message_hard_way(messages, g, p0, e, t)
            else
                v[t] = 1.0 - (1.0 - marginals[t, e[1]]) / temp_prob
            end  # TODO: can we improve line below?
            v[t] = max(v[t], v[t-1])  # numerical safeguard
        end
    end
    return marginals, messages
end

"""
    dmp_ic(g, p0, T)

Computes the marginals and messages for cascade of length T on a directed graph g
with initial condition p0.
"""
function dmp_ic(g::DirGraph, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    marginals = zeros(Float64, T, g.n)
    messages = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        messages[edge] = zeros(Float64, T)
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
            for neighbor in g.in_neighbors[i]
                marginals[t, i] *= (1.0 - g.edgelist[Int64[neighbor, i]] *
                                    messages[Int64[neighbor, i]][t-1])
            end  # TODO: can we improve line below?
            marginals[t, i] = max(marginals[t-1, i], 1.0 - marginals[t, i])  # numerical safeguard
        end
        for (e, v) in messages
            if haskey(g.edgelist, reverse(e))
                temp_prob = 1.0 - g.edgelist[reverse(e)] * messages[reverse(e)][t-1]
                if temp_prob == 0.0
                    v[t] = get_ic_message_hard_way(messages, g, p0, e, t)
                else
                    v[t] = 1.0 - (1.0 - marginals[t, e[1]]) / temp_prob
                end
            else
                v[t] = marginals[t, e[1]]
            end  # TODO: can we improve line below?
            v[t] = max(v[t], v[t-1])  # numerical safeguard
        end
    end
    return marginals, messages
end

"""
    dmp_ic(g, p0, T)

Computes the marginals and messages for cascade of length T on a simple graph g
with initial condition p0.
"""
function dmp_ic(g::SimpleGraph, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    marginals = zeros(Float64, T, g.n)
    messages = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in eachrow(g.edgelist)
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
                marginals[t, i] *= (1.0 - g.alpha[] * messages[Int64[neighbor, i]][t-1])
            end  # TODO: can we improve line below?
            marginals[t, i] = max(marginals[t-1, i], 1.0 - marginals[t, i])  # numerical safeguard
        end
        for (e, v) in messages
            temp_prob = 1.0 - g.alpha[] * messages[reverse(e)][t-1]
            if temp_prob == 0.0
                v[t] = get_ic_message_hard_way(messages, g, p0, e, t)
            else
                v[t] = 1.0 - (1.0 - marginals[t, e[1]]) / temp_prob
            end  # TODO: can we improve line below?
            v[t] = max(v[t], v[t-1])  # numerical safeguard
        end
    end
    return marginals, messages
end

"""
    get_ic_message_hard_way(messages, g, p0, e, t)

Computes the message for edge 'e' at time 't', when the original implementation is indefinite.
"""
function get_ic_message_hard_way(messages::Dict{Array{Int64, 1}, Array{Float64, 1}}, g::Graph,
                                 p0::Array{Float64, 1}, e::Array{Int64, 1}, t::Int64)
    message = 1.0 - p0[e[1]]
    for neighbor in g.neighbors[e[1]]
        if neighbor != e[2]
            message *= 1.0 - g.edgelist[sort(Int64[neighbor, e[1]])] * messages[Int64[neighbor, e[1]]][t-1]
        end
    end
    return 1.0 - message
end

"""
    get_ic_message_hard_way(messages, g, p0, e, t)

Computes the message for edge 'e' at time 't', when the original implementation is indefinite
in the case of directed graphs.
"""
function get_ic_message_hard_way(messages::Dict{Array{Int64, 1}, Array{Float64, 1}},
                                 g::DirGraph, p0::Array{Float64, 1}, e::Array{Int64, 1},
                                 t::Int64)
    message = 1.0 - p0[e[1]]
    for neighbor in g.in_neighbors[e[1]]
        if neighbor != e[2]
            message *= 1.0 - g.edgelist[Int64[neighbor, e[1]]] * messages[Int64[neighbor, e[1]]][t-1]
        end
    end
    return 1.0 - message
end

"""
    get_ic_message_hard_way(messages, g, p0, e, t)

Computes the message for edge 'e' at time 't', when the original implementation is indefinite
in the case of simple graphs.
"""
function get_ic_message_hard_way(messages::Dict{Array{Int64, 1}, Array{Float64, 1}}, g::SimpleGraph,
                                 p0::Array{Float64, 1}, e::Array{Int64, 1}, t::Int64)
    message = 1.0 - p0[e[1]]
    for neighbor in g.neighbors[e[1]]
        if neighbor != e[2]
            message *= 1.0 - g.alpha[] * messages[Int64[neighbor, e[1]]][t-1]
        end
    end
    return 1.0 - message
end

"""
    dmp_si(g, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with
initial condition p0.
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
            end  # TODO: maybe we need a safeguard, like for IC?
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
    dmp_sir(g, gamma, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with
initial condition p0 and removal probability gamma.
"""
function dmp_sir(g::Graph, gamma::Float64, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    mrg_s = zeros(Float64, T, g.n)
    mrg_r = zeros(Float64, T, g.n)
    msg_phi = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    msg_theta = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        msg_phi[edge] = zeros(Float64, T)
        msg_phi[reverse(edge)] = zeros(Float64, T)
        msg_theta[edge] = ones(Float64, T)
        msg_theta[reverse(edge)] = ones(Float64, T)
    end

    # initial conditions
    mrg_s[1, :] = 1.0 .- p0
    for (k, v) in msg_phi
        v[1] = p0[k[1]]
    end

    # DMP
    for t in 2:T
        # theta
        for (e, v) in msg_theta
            v[t] = v[t-1] - g.edgelist[sort(e)] * msg_phi[e][t-1]
        end
        # susceptible marginals
        for i in 1:g.n
            mrg_s[t, i] = 1.0 - p0[i]
            for neighbor in g.neighbors[i]
                mrg_s[t, i] *= msg_theta[Int64[neighbor, i]][t]
            end  # TODO: maybe we need a safeguard, like for IC?
        end
        # phi
        for (e, w) in msg_phi
            w[t] = (1.0 - g.edgelist[sort(e)]) * (1.0 - gamma) * w[t-1]
            w[t] += mrg_s[t-1, e[1]] / msg_theta[reverse(e)][t-1]
            w[t] -= mrg_s[t, e[1]] / msg_theta[reverse(e)][t]
        end
        # removed marginals
        for i in 1:g.n
            mrg_r[t, i] = mrg_r[t-1, i] + gamma * (1.0 - mrg_s[t-1, i] - mrg_r[t-1, i])
        end
    end

    return mrg_s, mrg_r, msg_phi, msg_theta
end

"""
    dmp_sir(g, gamma, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with
initial condition p0 and removal probabilities gamma.
"""
function dmp_sir(g::Graph, gamma::Array{Float64, 1}, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    mrg_s = zeros(Float64, T, g.n)
    mrg_r = zeros(Float64, T, g.n)
    msg_phi = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    msg_theta = Dict{Array{Int64, 1}, Array{Float64, 1}}()
    for edge in keys(g.edgelist)
        msg_phi[edge] = zeros(Float64, T)
        msg_phi[reverse(edge)] = zeros(Float64, T)
        msg_theta[edge] = ones(Float64, T)
        msg_theta[reverse(edge)] = ones(Float64, T)
    end

    # initial conditions
    mrg_s[1, :] = 1.0 .- p0
    for (k, v) in msg_phi
        v[1] = p0[k[1]]
    end

    # DMP
    for t in 2:T
        # theta
        for (e, v) in msg_theta
            v[t] = v[t-1] - g.edgelist[sort(e)] * msg_phi[e][t-1]
        end
        # susceptible marginals
        for i in 1:g.n
            mrg_s[t, i] = 1.0 - p0[i]
            for neighbor in g.neighbors[i]
                mrg_s[t, i] *= msg_theta[Int64[neighbor, i]][t]
            end  # TODO: maybe we need a safeguard, like for IC?
        end
        # phi
        for (e, w) in msg_phi
            w[t] = (1.0 - g.edgelist[sort(e)]) * (1.0 - gamma[e[1]]) * w[t-1]
            w[t] += mrg_s[t-1, e[1]] / msg_theta[reverse(e)][t-1]
            w[t] -= mrg_s[t, e[1]] / msg_theta[reverse(e)][t]
        end
        # removed marginals
        for i in 1:g.n
            mrg_r[t, i] = mrg_r[t-1, i] + gamma[i] * (1.0 - mrg_s[t-1, i] - mrg_r[t-1, i])
        end
    end

    return mrg_s, mrg_r, msg_phi, msg_theta
end

"""
    get_ic_objective(marginals, seed)

Caclulates the objective function of given activation times paired with
an initial condition (seed).
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

"""
    get_ic_objective(marginals, seed, noise)

Caclulates the objective function of given activation times paired with
an initial condition (seed) in the case of noisy times.
"""
function get_ic_objective(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}}, noise::TimeNoise)
    T::Int64 = size(marginals)[1]
    objective::Float64 = 0.0
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
            end
            objective += log(temp_sum) * seed[i][t]  # TODO: What if temp_sum == 0.0 ??
        end
    end
    return objective
end

"""
    get_ic_objective(marginals, seed, unobs_times)

Caclulates the objective function of given activation times paired with
an initial condition (seed) and assuming partly unobserved times.
"""
function get_ic_objective(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}},
                          unobs_times::Array{Int64, 1})
    T::Int64 = size(marginals)[1]
    objective::Float64 = 0.0
    for i in keys(seed)
        for t in keys(seed[i])
            if t == 1  # I assume that T > 1 (!!!)
                objective += log(marginals[t, i]) * seed[i][t]
            elseif t == T
                objective += log(1.0 - marginals[t-1, i]) * seed[i][t]
            elseif t > 1
                if t in unobs_times
                    t_low, t_upp = unobserved_time_interval(t, unobs_times)
                else
                    t_low, t_upp = t, t
                end
                objective += log(marginals[t_upp, i] - marginals[t_low-1, i]) * seed[i][t]
            end
        end
    end
    return objective
end
