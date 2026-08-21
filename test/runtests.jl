using GeneralizedSasakiNakamura
using Test

@testset "GeneralizedSasakiNakamura.jl" begin
    @testset "Public radial interfaces" begin
        X = GSN_radial(-2, 2, 2, 0.68, 0.3, IN)
        R = Teukolsky_radial(-2, 2, 2, 0.68, 0.3, IN)
        Y = Y_radial(-2, 2, 2, 0.68, 0.3, IN)
        r = 10.0
        rstar = rstar_from_r(0.68, r)

        @test X.method == "GSN-ISEM"
        @test R.GSN_solution.method == "GSN-ISEM"
        @test isfinite(X(rstar))
        @test isfinite(R(r))
        @test isfinite(Y(r))
        @test r_from_rstar(0.68, rstar) ≈ r
    end

    @testset "Exact-extremal radial interface" begin
        X = GSN_radial(
            -2, 2, 2, 1.0, 0.3, IN, -40.0, 60.0;
            method="GSN-ISEM", tolerance=1.0e-10)
        @test X.method == "GSN-ISEM"
        @test all(isfinite, X.GSN_solution(0.0))
        @test r_from_rstar(-1.0, rstar_from_r(-1.0, 2.0)) ≈ 2.0
        @test_throws DomainError GSN_radial(
            -2, 2, 2, 1.0, 1.0, IN; method="GSN-ISEM")
    end

    @testset "QNM interface" begin
        ordinary_mode = qnm(0.68, -2, 2, 2, 0)
        mirror_mode = qnm(0.68, -2, 2, 2, 0, mirror)

        @test ordinary_mode isa QNMResult
        @test mirror_mode isa QNMResult
        @test ordinary_mode.status == mirror_mode.status == :accepted
        @test real(ordinary_mode.omega) > 0
        @test real(mirror_mode.omega) < 0
        @test imag(ordinary_mode.omega) < 0
        @test imag(mirror_mode.omega) < 0
        @test ordinary_mode.excitation_factor ≈
            ordinary_mode.reflection_amplitude /
            (2 * ordinary_mode.omega * ordinary_mode.incidence_derivative)

        display_text = sprint(show, MIME("text/plain"), ordinary_mode)
        @test length(split(display_text, '\n')) == 9
        @test startswith(display_text, "QuasiNormalMode(\n")
        @test endswith(display_text, "    formalism = GSN)")

        detailed = qnm(
            0.68, -2, 2, 2, 0;
            primary_guess=ordinary_mode.omega, detailed=true)
        @test detailed.X isa GSNRadialFunction
        @test detailed.Y isa YRadialFunction
        @test detailed.R isa TeukolskyRadialFunction

        overtone7 = qnm_frequency(QNMMode(-2, 2, 2, 7), 0.0)
        overtone10 = qnm_frequency(QNMMode(-2, 2, 2, 10), 0.0)
        @test overtone7.status == overtone10.status == :accepted
        @test abs(overtone10.omega - overtone7.omega) > 0.1

        special = qnm_frequency(QNMMode(-2, 2, 2, 8), 0.0)
        endpoint = qnm(1.0, -2, 2, 2, 0)
        @test special.omega == -2.0im
        @test endpoint isa QNMEndpointResult
        @test ismissing(endpoint.excitation_factor)
    end

    @testset "Point-particle mode" begin
        mode = Teukolsky_pointparticle_mode(
            -2, 2, 2, 0, 0, 0.5, 8.0, 0.0, 1.0)
        @test mode.method.radial_method == "GSN-ISEM"
        @test isfinite(mode.amplitude)
        @test isfinite(mode.energy_flux)
    end
end
