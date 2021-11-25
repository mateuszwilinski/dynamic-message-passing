
struct Graph
    n::Int64
    m::Int64
    edgelist::Dict{Array{Int64, 1}, Real}
    neighbors::Array{Array{Int64, 1}, 1}
end

struct DirGraph
    n::Int64
    m::Int64
    edgelist::Dict{Array{Int64, 1}, Real}
    out_neighbors::Array{Array{Int64, 1}, 1}
    in_neighbors::Array{Array{Int64, 1}, 1}
end

struct SimpleGraph
    n::Int64
    m::Int64
    alpha::Ref{Real}
    edgelist::Array{Int64, 2}
    neighbors::Array{Array{Int64, 1}, 1}
end

struct TimeNoise
    m_1::Int64
    m_2::Int64
    p::Dict{Int64, Float64}
end
