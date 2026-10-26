# T1: bitwise comparison with the stored baseline (data/t1_baseline.tsv, written by collect.jl), and T2: every
# counter of ISEM.Certification stays at zero on these points (the second pass and its branches never run).
# Optional group (not part of "full"): about 20 minutes on one thread.

if !@isdefined(GSN)
    include(joinpath(@__DIR__, "..", "load_package.jl"))
end
include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(data_file("radial_references.jl"))
include(joinpath(@__DIR__, "points.jl"))

const T1_BASELINE = Dict(Pair(split(line, '\t'; limit=2)...)
    for line in eachline(data_file("t1_baseline.tsv")) if !isempty(line))
const T1_CERTIFICATION = isdefined(GSN.ISEM, :Certification) ? GSN.ISEM.Certification : nothing

@testset "baseline" begin
    T1_CERTIFICATION === nothing || T1_CERTIFICATION.reset_counters!()
    rows = t1_rows()
    @testset "T1 bitwise" begin
        @test Set(first.(rows)) == Set(keys(T1_BASELINE))
        changed = [key for (key, value) in rows if get(T1_BASELINE, key, nothing) != value]
        isempty(changed) || println("T1 changed rows:\n  ", join(first(changed, 20), "\n  "))
        @test isempty(changed)
    end
    if T1_CERTIFICATION !== nothing
        @testset "T2 counters" begin
            counters = T1_CERTIFICATION.counters()
            nonzero = filter(p -> last(p) != 0, collect(counters))
            isempty(nonzero) || println("T2 nonzero counters: ", nonzero)
            @test isempty(nonzero)
        end
    end
end
