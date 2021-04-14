
"""
    dynamic_messsage_passing(g, p0, T)

Compute the marginals and messages for cascade of length T on a graph g with initial condition p0
"""
function dynamic_messsage_passing(g::Graph, p0::Array{Float64, 1}, T::Int64)
    # initialising marginals and messages
    marginals = zeros(Float64, T, g.n)
    messages = Dict{Array{Int64,1}, Array{Float64,1}}()
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
            marginals[t, i] = (1.0 - p0[i])
            for neighbor in g.neighbors[i]
                marginals[t, i] *= (1.0 - g.edgelist[sort(Int64[neighbor, i])] *
                                    messages[Int64[neighbor, i]][t-1])
            end  # TODO: Przyjrzyj sie ponizszemu, czy nie da sie tego poprawic
            marginals[t, i] = max(marginals[t-1, i], 1.0 - marginals[t, i])
        end
        for (k, v) in messages  # TODO: uwzglednij mozliwosc, ze bedzie dzielenie przez zero (!!!)
            v[t] = (1.0 - (1.0 - marginals[t, k[1]]) /
                    (1.0 - g.edgelist[sort(Int64[k[1], k[2]])] *
                     messages[Int64[k[2], k[1]]][t-1]))
            v[t] = max(v[t], v[t-1])  # TODO: Przyjrzyj sie temu, czy nie da sie tego poprawic
        end
    end
    return marginals, messages
end

"""
    get_objective(marginals, seed)

Caclulates the objective function of given activation times paired with an initial condition (seed)
"""
function get_objective(marginals::Array{Float64, 2}, seed::Dict{Int64, Dict{Int64, Int64}})
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
