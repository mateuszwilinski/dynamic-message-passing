
struct Graph
    n::Int64
    m::Int64
    edgelist::Dict{Array{Int64, 1}, Float64}
    neighbors::Array{Array{Int32, 1}, 1}
end
