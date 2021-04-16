
"""
    dmp_si(g, p0, T)

Computes the marginals and messages for cascade of length T on a graph g with initial condition p0
"""
function dmp_si(g::Graph, p0::Array{Float64, 1}, T::Int64)
    marginals = zeros(Float64, T, g.n)
    
    return marginals
end
