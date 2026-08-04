@testset "auto direct ISEM dispatch" begin
    xin = GSN_radial(-2, 2, 2, 0.9, 0.5, IN)
    rin = Teukolsky_radial(-2, 2, 2, 0.9, 0.5, IN)
    yup = Y_radial(2, 2, 2, 0.9, 0.5, UP)

    @test xin.method == "direct_ISEM"
    @test rin.GSN_solution.method == "direct_ISEM"
    @test isfinite(xin(0.0))
    @test isfinite(rin(4.0))
    @test isfinite(yup(4.0))
end

@testset "point-particle direct radial input" begin
    mode = Teukolsky_pointparticle_mode(
        -2, 2, 2, 0, 0, 0.5, 8.0, 0.0, 1.0)
    @test mode.method.radial_method == "direct_ISEM"
    @test mode.Y_solution.P_solution === missing
    @test mode.Y_solution.Teukolsky_solution === missing
    @test isfinite(mode.amplitude)
    @test isfinite(mode.energy_flux)
end
