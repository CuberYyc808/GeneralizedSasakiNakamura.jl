if !@isdefined(GSN)
    include(joinpath(@__DIR__, "..", "load_package.jl"))
end

@testset "complex certificate semantics" begin
    CF = GSN.ISEM.ComplexFrequency
    good = (certified=true, estimated_error=1e-14, uncertified_stages=())
    finish = CF._finalize_complex_certificate
    @test finish(good, missing, 1e-14) == good
    @test finish(good, 1e-15, 1e-14) == good
    @test finish(good, 2e-14, 2e-14).estimated_error == 2e-14
    @test finish(good, 2e-14, 2e-14).certified
    @test_logs (:warn, r"not certified.*estimated_error=.*tolerance=") begin
        uncertified = finish(good, 3.8e-11, 1e-14)
        @test !uncertified.certified
        @test uncertified.estimated_error == 3.8e-11
    end
    relaxed = (certified=false, estimated_error=1e-15,
        uncertified_stages=(:complex_rational_patch_limit,))
    @test !finish(relaxed, missing, 1e-14).certified
    @test finish(relaxed, missing, 1e-14).uncertified_stages == relaxed.uncertified_stages
    for bad in (1e-2, 700.0, Inf, NaN)
        @test_throws ErrorException finish(good, bad, 1e-14)
        @test_throws ErrorException finish(merge(good, (estimated_error=bad,)), missing, 1e-14)
        @test_throws ErrorException finish(relaxed, bad, 1e-14)
    end
    # A metadata-only route fixture verifies policy-independent finalization,
    # including an already-uncertified patch-limit return.
    for policy in (:junction_search, :balanced_junction, :rstar_zero,
            :infinity_rotation, :infinity_rotation_relaxed, :offpole_match_plateau)
        metadata = merge(relaxed, (incidence_error=2e-14, reflection_error=3e-14,
            match_policy=policy))
        plan = (matching=(policy=policy,), controls=(tolerance=1e-13,))
        route = CF.ComplexRoute(:UP, nothing, nothing, nothing, nothing,
            metadata, plan, 1.0+0im, 1.0+0im, 1.0+0im, 1.0+0im, 1.0+0im, 1.0+0im)
        final = CF._complex_route_certificate(route)
        @test final.estimated_error == 3e-14
        @test !final.certified
        @test final.uncertified_stages == relaxed.uncertified_stages
    end
end

@testset "uncertified Y certificate survives conjugation" begin
    C = GSN.ISEM.Certification
    f = C.UncertifiedSolution(r -> (complex(r, 1.0), complex(2r, -1.0)), 3.8e-11,
        (:complex_rational_patch_limit,))
    @test f isa Function
    g = GSN.ISEM._conjugate_solution_function(f)
    @test C.certificate(g) == C.certificate(f)
    @test g(2.0) == conj.(f(2.0))
    mode = GSN.Mode(-2, 2, 2, 0.68, 0.5-0.08im, 4.0+0im)
    y = GSN.YRadialFunction(mode, GSN.IN, 1.0+0im, 1.0+0im, 1.0+0im,
        missing, missing, missing, r -> first(f(r)), f, GSN.UNIT_TEUKOLSKY_TRANS)
    @test C.certificate(y) == C.certificate(f)
    reflected = GSN.ISEM._conjugate_y_radial_function(y, -2, -conj(mode.omega))
    @test C.certificate(reflected) == C.certificate(y)
    @test reflected.Y_solution(2.0) == conj.(y.Y_solution(2.0))
end