
"""
    cascade_ic(g, p0, T)

Generate a cascade of length T on a graph g with initial condition p0.
"""
function cascade_ic(g::Graph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(2), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t-1, j] .== 0
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
    cascade_ic(g, p0, T)

Generate a cascade of length T on a directed graph g with initial condition p0.
"""
function cascade_ic(g::DirGraph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(2), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.out_neighbors[i]
                if active[t-1, j] .== 0
                    if rand() < g.edgelist[Int64[i, j]]
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
    cascade_ic(g, p0, T)

Generate a cascade of length T on a simple graph g with initial condition p0.
"""
function cascade_ic(g::SimpleGraph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(2), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t-1, j] .== 0
                    if rand() < g.alpha[]
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
    cascade_si(g, p0, T)

Generate a cascade of length T on a graph g with initial condition p0.
"""
function cascade_si(g::Graph, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(1), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t-1, j] .== 0
                    if rand() < g.edgelist[sort([i, j])]
                        active[t:end, j] .= 1
                    end
                end
            end
        end
    end
    return active
end

"""
    cascade_sir(g, gamma, p0, T)

Generate a cascade of length T on a graph g with initial condition p0
and removal probability gamma.
"""
function cascade_sir(g::Graph, gamma::Float64, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(1), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t-1, j] .== 0
                    if rand() < g.edgelist[sort([i, j])]
                        active[t:end, j] .= 1
                    end
                end
            end
            if rand() < gamma
                active[t:end, i] .= 2
            end
        end
    end
    return active
end

"""
    cascade_sir(g, gamma, p0, T)

Generate a cascade of length T on a graph g with initial condition p0
and removal probabilities gamma.
"""
function cascade_sir(g::Graph, gamma::Array{Float64, 1}, p0::Array{Float64, 1}, T::Int64)
    active = zeros(UInt8, T, g.n)
    active[1, :] = (rand(g.n) .< p0)
    active[2:end, :] = reshape(repeat(active[1, :] * UInt8(1), inner=T-1), (T-1, g.n))
    for t in 2:T
        actives = findall(x -> x == 1, active[t-1, :])  # active nodes
        for i in actives
            for j in g.neighbors[i]
                if active[t-1, j] .== 0
                    if rand() < g.edgelist[sort([i, j])]
                        active[t:end, j] .= 1
                    end
                end
            end
            if rand() < gamma[i]
                active[t:end, i] .= 2
            end
        end
    end
    return active
end

"""
    times_from_cascade(cascade)

Returns activation times of a given cascade.
"""
function times_from_cascade(cascade::Array{UInt8, 2})
    n::Int64 = size(cascade)[2]
    T::Int64 = size(cascade)[1]
    tau = repeat([T], inner=n)
    for i in 1:n
        temp = findfirst(isequal(1), cascade[:, i])
        if !isnothing(temp)
            tau[i] = temp
        end
    end
    return tau
end

"""
    add_activations_to_seed!(seed, tau)

Generates a list of activation times for a given cascade initial condition (seed).
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

Gets classes of cascades, based on their initial points (seeds), and their activation times.
"""
function preprocess_cascades(cascades::Array{Int64, 2})
    cascades_classes = Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}}()
    for i in 1:size(cascades)[2]
        tau = cascades[:, i]
        seeds = findall(isequal(1), tau)
        if !haskey(cascades_classes, seeds)
            cascades_classes[seeds] = Dict{Int64, Dict{Int64, Int64}}()
        end
        add_activations_to_seed!(cascades_classes[seeds], tau)
    end
    return cascades_classes
end

"""
    preprocess_single_class(cascades)

Generates cascades in a class format, but with only one class. It is supposed to be used
when there is a known, stochastic initial condition for all the cascades.
"""
function preprocess_single_class(cascades::Array{Int64, 2})
    cascades_class = Dict{Int64, Dict{Int64, Int64}}()
    for i in 1:size(cascades)[2]
        tau = cascades[:, i]
        add_activations_to_seed!(cascades_class, tau)
    end
    return cascades_class
end

"""
    remove_unobserved!(cascades_classes, unobserved)

Removes unobserved nodes from the dictionary of activated nodes.
"""
function remove_unobserved!(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                            unobserved::Array{Int64, 1})
    for seed in keys(cascades_classes)
        for key in unobserved
            delete!(cascades_classes[seed], key)
        end
    end
end

"""
    remove_unobserved!(cascades_class, unobserved)

Removes unobserved nodes from the dictionary of activated nodes.
"""
function remove_unobserved!(cascades_class::Dict{Int64, Dict{Int64, Int64}},
                            unobserved::Array{Int64, 1})
    for key in unobserved
        delete!(cascades_class, key)
    end
end

"""
    local_cascade_likelihood(node, times, g, T)

Calculates the local (for a given node in-connections) likelihood of a given cascade.
"""
function local_cascade_likelihood(node::Int64, times::Array{Int64,1}, g::Graph, T::Int64)
    p_i = 1.0
    if (times[node] > 1)
        temp = 1.0
        for neighbor in g.neighbors[node]
            if times[neighbor] < (times[node] - 1)
                p_i *= 1.0 - g.edgelist[sort(Int64[neighbor, node])]
            elseif times[neighbor] == (times[node] - 1)
                temp *= 1.0 - g.edgelist[sort(Int64[neighbor, node])]
            end
        end
        if times[node] != T
            p_i *= 1 - temp
        end
    end
    return p_i
end

"""
    local_cascade_likelihood(node, times, g, T)

Calculates the local (for a given node in-connections) likelihood of a given cascade.
"""
function local_cascade_likelihood(node::Int64, times::Array{Int64,1}, g::DirGraph, T::Int64)
    p_i = 1.0
    if (times[node] > 1)
        temp = 1.0
        for neighbor in g.in_neighbors[node]
            if times[neighbor] < (times[node] - 1)
                p_i *= 1.0 - g.edgelist[Int64[neighbor, node]]
            elseif times[neighbor] == (times[node] - 1)
                temp *= 1.0 - g.edgelist[Int64[neighbor, node]]
            end
        end
        if times[node] != T
            p_i *= 1 - temp
        end
    end
    return p_i
end

"""
    cascade_likelihood(times, g, T)

Calculates the likelihood of a given cascade (defined by activation times).
"""
function cascade_likelihood(times::Array{Int64,1}, g::Union{Graph, DirGraph}, T::Int64)
    p_i = zeros(Real, g.n)
    for i in 1:g.n
        p_i[i] = log(local_cascade_likelihood(i, times, g, T))
    end
    return sum(p_i)
end
