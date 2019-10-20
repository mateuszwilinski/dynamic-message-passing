
"""
    edgelist_from_array(edges, wights)

Generate a proper edgelist (as a dictionary) for graph structure, from an array of edges
"""
function edgelist_from_array(edges::Array{Int64, 2}, edge_weights::Array{Float64, 1})
    edgelist = Dict{Array{Int64, 1}, Float64}()
    for i in 1:length(edge_weights)
        edgelist[[edges[i, 1], edges[i, 2]]] = edge_weights[i]
    end
    return edgelist
end

"""
    neighbors_from_edges(edges, n)

Compute neighbors for each of n nodes, based on the list of edges
"""
function neighbors_from_edges(edges::Dict{Array{Int64, 1}, Float64}, n::Int64)
    neighbors = [Int64[] for i in 1:n]
    for edge in edges
        push!(neighbors[edge[1][1]], edge[1][2])
        push!(neighbors[edge[1][2]], edge[1][1])
    end
    return neighbors
end

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
        for (k, v) in messages
            v[t] = (1.0 - p0[k[1]])
            for neighbor in g.neighbors[k[1]]
                if neighbor != k[2]
                    v[t] *= 1.0 - g.edgelist[sort(Int64[neighbor, k[1]])] * messages[Int64[neighbor, k[1]]][t-1]
                end
            end
            v[t] = 1.0 - v[t]
        end
        for i in 1:g.n
            marginals[t, i] = (1.0 - p0[i])
            for neighbor in g.neighbors[i]
                marginals[t, i] *= 1.0 - g.edgelist[sort(Int64[neighbor, i])] * messages[Int64[neighbor, i]][t-1]
            end
            marginals[t, i] = 1.0 - marginals[t, i]
        end
    end

    return marginals, messages
end

"""
    cascade(g, p0, T)

Generate a cacade of length T on a graph g with initial condition p0
"""
function cascade(g::Graph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(2), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t, j] .== 0
                    if rand() < g.edgelist[sort([i, j])]
                        active[t, j] = 1
                        active[(t+1):end, j] = repeat(UInt8[2], inner=T-t)
                    end
                end
            end
        end
    end
    return active
end

"""
    times_from_cascade(cascade)

Returns activation times of a given cacade
"""
function times_from_cascade(cascade::Array{UInt8, 2})
    n = size(cascade)[2]
    T = size(cascade)[1]
#     tau = zeros(Int64, n)
    tau = repeat([T], inner=n)
    for i in 1:n
        temp = findfirst(isequal(1), cascade[:, i])
        if temp != nothing
            tau[i] = temp
        end
    end
    return tau
end

"""
    add_activations_to_seed!(seed, tau)

Generates a list of activation times for a given cascade initial point (seed)
"""
function add_activations_to_seed!(seed::Dict{Int64, Dict{Int64, Int64}}, tau::Array{Int64, 1})
    for i in 1:length(tau)
        t = tau[i]
        if !haskey(seed, i)
            seed[i] = Dict{Int64, Int64}()
            seed[i][t] = 1
        else
            if !haskey(seed[i], t)
                seed[i][t] = 1
            else
                seed[i][t] += 1
            end
        end
    end
end

"""
    preprocess_cascades(cascades)

Gets classes of cascades, based on their initial points (seeds), and their activation times
"""
function preprocess_cascades(cascades::Array{Array{UInt8, 2}, 1})
    cascades_classes = Dict{Int64, Dict{Int64, Dict{Int64, Int64}}}()
    for cascade in cascades
        T = size(cascade)[1]
        tau = times_from_cascade(cascade)
        p0 = convert(Array{Float64, 1}, cascade[1, :])
        seed = findfirst(isequal(1.0), p0)
        if !haskey(cascades_classes, seed)
            cascades_classes[seed] = Dict{Int64, Dict{Int64, Int64}}()
        end
        add_activations_to_seed!(cascades_classes[seed], tau)
    end
    return cascades_classes
end

"""
    get_objective(marginals, tau)

Caclulates the objective function of given activation times tau according to given marginals
"""
function get_objective(marginals::Array{Float64, 2}, tau::Array{Int64,1})
    T = size(marginals)[1]
    objective = 0.0
    for (i, t) in enumerate(tau)
        if t == 1  # I assume that T > 1 (!!!)
            objective += log(marginals[t, i])
        elseif t == T
            objective += log(1.0 - marginals[t-1, i])
        elseif t > 1
            objective += log(marginals[t, i] - marginals[t-1, i])
        end
    end
    return objective
end

"""
    get_objective(marginals, seed)

Caclulates the objective function of given activation times paired with an initial point (seed)
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

"""
    lambda_from_marginals(marginals, tau)

Calculates marginals lagrange coefficients for a given cascade with activation times tau
"""
function lambda_from_marginals(marginals::Array{Float64, 2}, tau::Array{Int64,1})
    T = size(marginals)[1]
    n = size(marginals)[2]
    lambda = zeros(Float64, T, n)
    for (i, t) in enumerate(tau)
        if t == 1  # I assume that T > 1 (!!!)
            lambda[t, i] = -1.0 / marginals[t, i]
        elseif t == T
#             lambda[t, i] = -1.0 / (1 - marginals[t-1, i])
            lambda[t-1, i] = -1.0 / (marginals[t-1, i] - 1)
        elseif t > 1
            lambda[t, i] = -1.0 / (marginals[t, i] - marginals[t-1, i])
            lambda[t-1, i] = -1.0 / (marginals[t-1, i] - marginals[t, i])
        end
    end
    return lambda
end

"""
    lambda_from_marginals(marginals, seed)

Calculates marginals lagrange coefficients for a given initial condition (seed)
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
                lambda[t-1, i] -= seed[i][t] / (marginals[t-1, i] - 1.0)
            elseif t > 1
                lambda[t, i] -= seed[i][t] / (marginals[t, i] - marginals[t-1, i])
                lambda[t-1, i] -= seed[i][t] / (marginals[t-1, i] - marginals[t, i])
            end
        end
    end
    return lambda
end

"""
    get_lambda_ij(lambda, g, messages, p0)

Calculates messages lagrange coefficients
"""
function get_lambda_ij(lambda::Array{Float64, 2}, g::Graph, messages::Dict{Array{Int64,1}, Array{Float64,1}}, p0::Array{Float64, 1})
    T = size(lambda)[1]
    lambda_ij = Dict{Array{Int64,1}, Array{Float64,1}}()
    for edge in keys(g.edgelist)
        lambda_ij[edge] = zeros(Float64, T)
        lambda_ij[reverse(edge)] = zeros(Float64, T)
    end

    for t in T-1:-1:1
        for e in keys(lambda_ij)
            n_neighbors = size(g.neighbors[e[2]])[1]
            temp_j = lambda[t+1, e[2]] * g.edgelist[sort(e)] * (1.0 - p0[e[2]])
            temp_ij = repeat([g.edgelist[sort(e)] * (1.0 - p0[e[2]])], n_neighbors)
            for k in 1:n_neighbors
                if g.neighbors[e[2]][k] != e[1]
                    temp = (1.0 - g.edgelist[sort([g.neighbors[e[2]][k], e[2]])] * messages[[g.neighbors[e[2]][k], e[2]]][t])
                    temp_j *= temp
                    temp_ij[1:end .!= k] *= temp
                    temp_ij[k] *= lambda_ij[[e[2], g.neighbors[e[2]][k]]][t+1]
#                     temp_ij *= temp
#                     temp_ij[k] *= lambda_ij[[e[2], g.neighbors[e[2]][k]]][t+1] / temp
                else
                    temp_ij[k] = 0.0
                end
            end
            lambda_ij[e][t] += temp_j + sum(temp_ij)
        end
    end
    return lambda_ij
end

"""
    get_gradient(cascades, g, unobserved)

Calculates gradient for alphas according to lagrange derivative summed over cascades
"""
function get_gradient(cascades::Array{Array{UInt8,2},1}, g::Graph, unobserved::Array{Int64, 1})
    D_ij = Dict{Array{Int64, 1}, Float64}()
    all_marginals = Dict{Int64, Array{Float64,2}}()
    all_messages = Dict{Int64, Dict{Array{Int64,1}, Array{Float64,1}}}()
    objective = 0.0
    for c in 1:size(cascades)[1]
        T = size(cascades[c])[1]
        tau = times_from_cascade(cascades[c])
        p0 = convert(Array{Float64, 1}, cascades[c][1, :])
        seed = findfirst(isequal(1.0), p0)
        if !haskey(all_marginals, seed)
            all_marginals[seed], all_messages[seed] = dynamic_messsage_passing(g, p0, T)
        end
        lambda = lambda_from_marginals(all_marginals[seed], tau)
        lambda[:, unobserved] .= 0.0
        lambda_ij = get_lambda_ij(lambda, g, all_messages[seed], p0)

        objective += get_objective(all_marginals[seed], tau)
        for (edge, v) in g.edgelist  # bare in mind that 'v' could be zero (maybe 'if' is needed?)
            if c == 1
                D_ij[edge] = sum(lambda_ij[edge] .* all_messages[seed][edge] +
                    lambda_ij[reverse(edge)] .* all_messages[seed][reverse(edge)]) / v
            else
                D_ij[edge] += sum(lambda_ij[edge] .* all_messages[seed][edge] +
                    lambda_ij[reverse(edge)] .* all_messages[seed][reverse(edge)]) / v
            end
        end
    end
return D_ij, objective
end

"""
    get_gradient(cascades_classes, g, T, unobserved)

Calculates gradient for alphas according to lagrange derivative summed over classes of cascades
"""
function get_gradient(cascades_classes::Dict{Int64, Dict{Int64, Dict{Int64, Int64}}}, g::Graph, T::Int64, unobserved::Array{Int64, 1})
    D_ij = Dict{Array{Int64, 1}, Float64}()
    # moze szybciej bedzie tutaj zaalokowac marginals i messages?
    objective = 0.0
    for seed in keys(cascades_classes)
        p0 = zeros(Float64, g.n)
        p0[seed] = 1.0
        marginals, messages = dynamic_messsage_passing(g, p0, T)
        lambda = lambda_from_marginals(marginals, cascades_classes[seed])
        lambda[:, unobserved] .= 0.0
        lambda_ij = get_lambda_ij(lambda, g, messages, p0)

        objective += get_objective(marginals, cascades_classes[seed])
        for (edge, v) in g.edgelist  # bare in mind that 'v' could be zero (maybe an 'if' is needed?)
            if !haskey(D_ij, edge)
                D_ij[edge] = sum(lambda_ij[edge] .* messages[edge] +
                    lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
            else
                D_ij[edge] += sum(lambda_ij[edge] .* messages[edge] +
                    lambda_ij[reverse(edge)] .* messages[reverse(edge)]) / v
            end
        end
    end
    return D_ij, objective
end

# TODO: popraw 'unobserved', zeby bylo zgrabniejsze (!!!)
