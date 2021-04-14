
"""
    cascade(g, p0, T)

Generate a cascade of length T on a graph g with initial condition p0
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

Returns activation times of a given cascade
"""
function times_from_cascade(cascade::Array{UInt8, 2})
    n::Int64 = size(cascade)[2]
    T::Int64 = size(cascade)[1]
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

Generates a list of activation times for a given cascade initial condition (seed)
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
    remove_unobserved!(cascades_classes, unobserved)

Removes unobserved nodes from the dictionary of activated nodes
"""
function remove_unobserved!(cascades_classes::Dict{Array{Int64, 1}, Dict{Int64, Dict{Int64, Int64}}},
                            unobserved::Array{Int64, 1})
    for seed in keys(cascades_classes)
        for key in unobserved
            delete!(cascades_classes[seed], key)
        end
    end
end
