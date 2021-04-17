
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
    cascade(g, p0, T)

Generate a cascade of length T on a graph g with initial condition p0
"""
function cascade_si(g::Graph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(1), inner=T-1), (T-1, g.n))
    # active[2:end, :] = reshape(repeat(active[1, :] * UInt8(2), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t, j] .== 0
                    if rand() < g.edgelist[sort([i, j])]
                        active[t:end, j] .= 1
                        # active[t, j] = 1
                        # active[(t+1):end, j] = repeat(UInt8[2], inner=T-t)
                    end
                end
            end
        end
    end
    return active
end
