
using Test

include("../src/structures.jl")
include("../src/dynamic_message_passing.jl")
include("../src/network_tools.jl")

# Data

edges = Int64[[1 2]; [1 3]]
m = size(edges)[1]
n = maximum(edges)
edge_weights = [0.5, 0.3]

edgelist = edgelist_from_array(edges, edge_weights)
neighbors = neighbors_from_edges(edgelist, n)

g = Graph(n, m, edgelist, neighbors)

p0 = zeros(3)
p0[1] = 0.4
T = 3
margins, msgs = dmp_ic(g, p0, T)

# DMP Tests

@testset "dmp_ic" begin
    for i in 1:3
        @test isapprox(margins[i, 1], 0.4)
    end
    for i in 2:3
        @test isapprox(margins[1, i], 0.0)
    end
    for i in 2:3
        @test isapprox(margins[i, 2], 0.2)
    end
    for i in 2:3
        @test isapprox(margins[i, 3], 0.12)
    end
end
