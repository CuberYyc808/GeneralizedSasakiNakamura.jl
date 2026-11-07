# Infinity endpoint at tiny real omega on the ordinary route (small-frequency expansion off, which the point-particle
# modes take after the MST certificate rejects a state at the orbit radius): the y coefficients grow like omega^-n and
# the recurrence lost credibility before order 2, so the mode was recorded as failed with zero flux (four 0PA Kerr
# spherical modes, omega = 2.4e-8 to 6.0e-7). The dimensionless Y = y / omega selection is taken instead. Reference for
# Y: the MST route of the same mode at a radius where its certificate accepts the state (independent representation).
@testset "ordinary infinity endpoint at tiny real omega" begin
    modes = (
        (5, -5, 0.9979160269150359, 2.4672579251438703e-8, 245435.73306292985),
        (4, 4, 0.702648271571811, 2.4395044794180108e-8, 299575.01045117085),
        (5, -5, 0.9979160269150359, 6.016469129908073e-7, 35357.27177474854),
        (5, 4, 0.9979645726597075, 2.83780649771205e-7, 67710.8284474255),
    )
    for (l, m, a, omega, p) in modes
        Y = Y_radial(-2, l, m, a, omega, IN; sfe=false)
        @test all(r -> isfinite(Y(r)), (0.8p, p, 1.25p))
    end
    l, m, a, omega, p = modes[2]
    r = 0.8p
    @test Y_radial(-2, l, m, a, omega, IN; sfe=false)(r) ≈ Y_radial(-2, l, m, a, omega, IN)(r) rtol=1e-9
    mode = Teukolsky_pointparticle_mode(-2, 5, 4, 0, 1, 0.9979645726597075, 67710.8284474255, 0.0, 0.9995241107909288;
        tol=1e-8, threaded_sampling=false)
    @test mode.energy_flux > 0 && isfinite(mode.energy_flux)
end
