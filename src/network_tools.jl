
"""
    edgelist_from_array(edges, edge_weights)

Generate a proper edgelist (as a dictionary) for graph structure, from an array of edges.
"""
function edgelist_from_array(edges::Array{Int64, 2}, edge_weights::Array{<:Real,1})
    edgelist = Dict{Array{Int64, 1}, Real}()
    for i in 1:length(edge_weights)
        edgelist[[edges[i, 1], edges[i, 2]]] = edge_weights[i]
    end
    return edgelist
end

"""
    full_edgelist(n, edge_weights)

Generate a proper edgelist (as a dictionary) for a full graph structure, with n nodes.
"""
function full_edgelist(n::Int64, edge_weights::Array{<:Real,1})
    edgelist = Dict{Array{Int64, 1}, Real}()
    k = 0
    for i in 1:(n-1)
        for j in (i+1):n
            k += 1
            edgelist[[i, j]] = edge_weights[k]
        end
    end
    return edgelist
end

"""
    neighbors_from_edges(edges, n)

Compute neighbors for each of n nodes, based on the edgelist dictionary.
"""
function neighbors_from_edges(edgelist::Dict{Array{Int64, 1}, Real}, n::Int64)
    neighbors = [Int64[] for i in 1:n]
    for edge in edgelist
        push!(neighbors[edge[1][1]], edge[1][2])
        push!(neighbors[edge[1][2]], edge[1][1])
    end
    return neighbors
end

"""
    full_neighbors(n)

Compute neighbors for each of n nodes, for a full graph.
"""
function full_neighbors(n::Int64)
    neighbors = [filter(x -> x != i, 1:n) for i in 1:n]
    return neighbors
end

"""
    dir_neighbors_from_edges(edges, n)

Compute in- and out-neighbors for each of n nodes, based on the list of edges for
directed network.
"""
function dir_neighbors_from_edges(edges::Dict{Array{Int64, 1}, Real}, n::Int64)
    out_neighbors = [Int64[] for i in 1:n]
    in_neighbors = [Int64[] for i in 1:n]
    for edge in edges
        push!(out_neighbors[edge[1][1]], edge[1][2])
        push!(in_neighbors[edge[1][2]], edge[1][1])
    end
    return out_neighbors, in_neighbors
end

"""
    neighbors_from_edges(edges, n)

Compute neighbors for each of n nodes, based on the list of edges.
"""
function neighbors_from_edges(edges::Array{Int64, 2}, n::Int64)
    neighbors = [Int64[] for i in 1:n]
    for edge in eachrow(edges)
        push!(neighbors[edge[1]], edge[2])
        push!(neighbors[edge[2]], edge[1])
    end
    return neighbors
end

"""
    remove_edge!(g, edge)

Removes an edge from graph g.
"""
function remove_edge!(g::Graph, edge::Array{Int64, 1})
    delete!(g.edgelist, edge)
    deleteat!(g.neighbors[edge[1]], findfirst(x -> x == edge[2], g.neighbors[edge[1]]))
    deleteat!(g.neighbors[edge[2]], findfirst(x -> x == edge[1], g.neighbors[edge[2]]))
end

"""
    find_leaves(g)

Returns an array of leaves.
"""
function find_leaves(g::Graph)
    leaves = Int64[]
    for (i, ns) in enumerate(g.neighbors)
        if (length(ns) == 1)
            append!(leaves, [i])
        end
    end
    return leaves
end

"""
    find_unobserved_leaves(g, unobserved)

Returns an array of edges, which connect unobserved leaves.
"""
function find_unobserved_leaves(g::Graph, unobserved::Array{Int64, 1})
    leaves = Array{Array{Int64, 1}, 1}()
    for (i, ns) in enumerate(g.neighbors)
        if (length(ns) == 1) & (i in unobserved)
            append!(leaves, [sort([ns[1], i])])
        end
    end
    return leaves
end

# TODO: Add degree function (both directed and undirected)
