using GeneralizedSasakiNakamura
using Test

@testset "GeneralizedSasakiNakamura.jl" begin
    include("direct_gsn_smoke.jl")
    include("direct_gsn_complex_frequency.jl")
    include("direct_auto_inhomogeneous.jl")
end
