@testset "basis coverage failure is retried, then recorded" begin
    CI = GSN.ConvolutionIntegrals
    calls = Ref(0)
    covered_when_ordinary = () -> begin
        calls[] += 1
        get(task_local_storage(), CI._FORCE_ORDINARY_KEY, false) ? Dict("EnergyFlux" => 1.0) :
            error("ISEM GSN basis does not cover x=0.93")
    end
    @test CI._with_coverage_retry(covered_when_ordinary, 2, 2, 2, 5, 0)["EnergyFlux"] == 1.0
    @test calls[] == 2
    failed = @test_logs (:warn,) CI._with_coverage_retry(() -> error("ISEM GSN basis does not cover x=0.93"), 2, 2, 2, 5, 0)
    @test failed["FailedMode"] && failed["EnergyFlux"] == 0.0
    @test_throws ErrorException CI._with_coverage_retry(() -> error("unrelated"), 2, 2, 2, 5, 0)
    @test_throws InterruptException CI._with_coverage_retry(() -> throw(InterruptException()), 2, 2, 2, 5, 0)
end

@testset "polar orbits with x = 0 are finite" begin
    f0 = GSN.Teukolsky_pointparticle_flux(0.9, 8.0, 0.0, 0.0; tol = 1e-6)
    fm = GSN.Teukolsky_pointparticle_flux(0.9, 8.0, 0.0, -0.0; tol = 1e-6)
    @test isfinite(f0.infinity_energy_flux) && isfinite(f0.horizon_energy_flux)
    @test f0.infinity_energy_flux == fm.infinity_energy_flux
    @test isapprox(f0.infinity_energy_flux, 1.869535e-4; rtol = 1e-4)
end

@testset "retrograde equatorial fallbacks keep the sign of L" begin
    f = GSN.Teukolsky_pointparticle_flux(0.9, 10.0, 0.0, -1.0; tol = 1e-6)
    g = MS.inclined_mode_summation(0.9, 10.0, -1.0; tol = 1e-6)
    @test f.infinity_angular_momentum_flux < 0
    @test g.infinity_angular_momentum_flux ≈ f.infinity_angular_momentum_flux rtol = 1e-6  # minimum_consecutive 2 vs 5
end
