using DelimitedFiles

include("src/structures.jl")
include("src/functions.jl")

edges = readdlm("data/barabasi_tree.csv", ' ', Int64) .+ 1

weights = rand(size(edges)[1])
n = maximum(edges)

edgelist = edgelist_from_array(edges, weights)
m = length(edgelist)
neighbors = neighbors_from_edges(edgelist, n)

g = Graph(n, m, edgelist, neighbors)

k = 1
p0 = zeros(n)
p0[k] = 1.0
T = 10

# DMP

marginals, messages = dynamic_messsage_passing(g, p0, T)

# Simulation

sim_n = 1000000

temp = zeros(Int64, T, n)
for i in 1:sim_n
    temp .+= (cascade(g, p0, T) .> 0)
end

println("The relative difference between DMP and simulations is:")
println(sum(abs.(marginals - temp ./ sim_n)) / sum(marginals))
