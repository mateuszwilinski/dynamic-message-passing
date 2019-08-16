
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
            for neighbor in neighbors[k[1]]
                if neighbor != k[2]
                    v[t] *= 1.0 - edgelist[sort(Int64[neighbor, k[1]])] * messages[Int64[neighbor, k[1]]][t-1]
                end
            end
            v[t] = 1.0 - v[t]
        end
        for i in 1:g.n
            marginals[t, i] = (1.0 - p0[i])
            for neighbor in neighbors[i]
                marginals[t, i] *= 1.0 - edgelist[sort(Int64[neighbor, i])] * messages[Int64[neighbor, i]][t-1]
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
    active[1, :] = (rand(n) .< p0)
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
    edgelist_from_array(edges, wights)

Generate a proper edgelist (as a dictionary) for graph structure, from an array of edges
"""
function edgelist_from_array(edges::Array{Int64, 2}, weights::Array{Float64, 1})
    edgelist = Dict{Array{Int64, 1}, Float64}()
    for i in 1:length(weights)
        edgelist[[edges[i, 1], edges[i, 2]]] = weights[i]
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
