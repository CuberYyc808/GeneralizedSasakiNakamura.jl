# Writes the T1 baseline. Usage (checkout root):
#   GSN_SOURCE=$PWD julia --threads=1 test/extended/baseline/collect.jl [out.tsv] [--no-flux]
include(joinpath(@__DIR__, "..", "load_package.jl"))
include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(data_file("radial_references.jl"))
include(joinpath(@__DIR__, "points.jl"))
out = something(findfirst(a -> !startswith(a, "--"), ARGS) |> i -> i === nothing ? nothing : ARGS[i],
    data_file("t1_baseline.tsv"))
rows = t1_rows(; flux="--no-flux" ∉ ARGS)
open(out, "w") do io
    for (key, value) in rows
        println(io, key, '\t', value)
    end
end
println("wrote ", length(rows), " rows to ", out)
