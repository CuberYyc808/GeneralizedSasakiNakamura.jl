# Small reflection above the former complex-path ratio 1e-6. References: a 256-bit Frobenius and Taylor solution of
# the Teukolsky equation (independent of the package), Y at the transmission amplitude of the Y_radial. At (a, omega) =
# (0.99, 1.15) of (l, m) = (2, 1), |B_ref / B_inc| = 4.5e-6 and the matched B_ref was off by 1.1e-5, which R_out / R_in
# ~ r^4 brought into Y as 8e-7 at r = 10 and 1.1e-5 at r = 50.
@testset "Y_radial with a small reflection (complex path)" begin
    Y = Y_radial(-2, 2, 1, 0.99, 1.15, IN)
    @test Y.Y_solution(10.0)[1] ≈ -2.6970325277921525e-5 - 8.666135783741672e-5im rtol=1e-8
    @test Y.Y_solution(50.0)[1] ≈ 2.2779902278298815e-5 + 2.384758130773702e-5im rtol=1e-8
end

# The near-horizon re-solve of Y (RadialFunctions.y_radial_function) keeps the first solution when the re-solve
# fails a certificate, and rethrows programming errors. (l, m, a, omega) = (2, 2, 0.99, 1.15) takes the re-solve
# (mismatch 1e-8); its coefficient backend is swapped for one that throws.
@testset "Y re-solve: certificate failures fall back, programming errors propagate" begin
    route = D.radial_route(-2, 2, 2, 0.99, 1.15, :IN;
        small_frequency_expansion=:auto, large_frequency_expansion=:auto)
    reference = D.y_radial_function(route).Y_solution(1.5)[1]
    backend = D.CoefficientTables._BACKEND[]
    try
        D.CoefficientTables._BACKEND[] = (args...; kwargs...) -> error("no certified seed (test)")
        @test isfinite(D.y_radial_function(route).Y_solution(1.5)[1])
        D.CoefficientTables._BACKEND[] = (args...; kwargs...) -> throw(MethodError(+, ("test",)))
        @test_throws MethodError D.y_radial_function(route)
    finally
        D.CoefficientTables._BACKEND[] = backend
    end
    @test D.y_radial_function(route).Y_solution(1.5)[1] == reference
end
