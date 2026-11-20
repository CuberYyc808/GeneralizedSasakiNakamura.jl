# Y of s = -2 IN between the horizon series and the matching point from the Teukolsky Taylor chain (NearHorizonTeukolsky,
# G-005). Before it, the Y converted from X was off by 8e-7 at r = 2 - 3.4 for (l, m, a, omega) = (3, 2, 0.9, 2.37),
# 1e-7 for (2, 1, 0.9, 1.2595) and 3e-8 for (3, 3, 0.99, 1.6987). Float64 self-consistency first; then stored references
# from H-008 (128-bit arbitrary-precision route; nothing above 64 bits is computed here).
const G005_POINTS = ((3, 2, 0.9, 2.37), (2, 1, 0.9, 1.2595), (3, 3, 0.99, 1.6987))
_nh_route(l, m, a, omega) = D.radial_route(-2, l, m, a, omega, :IN; small_frequency_expansion=:auto, large_frequency_expansion=:auto)

@testset "Teukolsky Taylor chain: self-consistency (Float64)" begin
    # Inside r_+ + 2 kappa the horizon series still converges and, for these modes, hardly cancels (sum |terms| / |sum|
    # <= 10 at r_+ + 1.2 kappa): the chain, started from it at r_+ + kappa / 2, must agree with it there.
    for (l, m, a, omega) in ((2, 2, 0.7, 0.5), G005_POINTS[1], G005_POINTS[2])
        Y = Y_radial(-2, l, m, a, omega, IN)
        kappa = sqrt(1 - a^2); rplus = 1 + kappa; rminus = 1 - kappa
        r = rplus + 1.2kappa
        R, _ = D.RadialFunctions.teukolsky_horizon_pair(-2, m, a, omega, Y.mode.lambda, r)
        phase = omega * rstar_from_r(a, r) - a * m / (rplus - rminus) * log((r - rplus) / (r - rminus))
        @test Y.Y_solution(r)[1] ≈ Y.transmission_amplitude * exp(-im * phase) * R / r^2 rtol=1e-12
    end
    # The chain is used, and hands over to the converted Y at r_check (the self-check mismatch; 1.5e-11 - 3e-11 on
    # arm64), and on both sides of r_start (series -> chain) and r_check (chain -> converted Y).
    for p in G005_POINTS
        route = _nh_route(p...)
        y, mismatch = D.RadialFunctions._route_y_solution_checked(route, D.RadialFunctions._route_y_amplitudes(route))
        @test y isa D.RadialFunctions.RstarYSolution && y.chain !== nothing
        @test mismatch <= 1e-9
        for r in (y.r_start, y.r_check)
            @test y(r * (1 - 1e-12))[1] ≈ y(r * (1 + 1e-12))[1] rtol=1e-9
        end
    end
    # No step along omega: (3, 2, 0.9), omega in [2.36, 2.39], 20 points; Y at r = 2.5 with the phase e^{-i omega r*}
    # taken out, against the cubic through its four neighbours (the converted Y scattered by 1e-6 here).
    ws = range(2.36, 2.39; length=20)
    v = [(y = Y_radial(-2, 3, 2, 0.9, w, IN); y.Y_solution(2.5)[1] / y.transmission_amplitude * exp(im * w * rstar_from_r(0.9, 2.5))) for w in ws]
    for i in 3:18
        interpolated = (-v[i-2] + 4v[i-1] + 4v[i+1] - v[i+2]) / 6
        @test abs(v[i] / interpolated - 1) <= 1e-11
    end
end

@testset "Y_radial near the horizon from the Teukolsky Taylor chain (H-008)" begin
    # Y / T at unit Teukolsky transmission, 128-bit arbitrary-precision route (hp_requests/H-008/H-008_Y.csv; 160-bit
    # check 4e-37); the chain gives 3e-14 - 1.2e-12 on arm64.
    references = (
        (3, 2, 0.9, 2.37, 2.5567496226930975, 2.31622112370605121e-02 + 4.00448945557220076e-02im),
        (3, 2, 0.9, 2.37, 3.2417194566780605, 3.09227048064913031e-02 - 2.71424689444733531e-02im),
        (2, 1, 0.9, 1.2595, 2.5567496226930975, -3.33619283589444496e-02 - 4.04463546504071625e-02im),
        (2, 1, 0.9, 1.2595, 3.2417194566780605, 4.52346976734674805e-02 - 8.32597160084934222e-03im),
        (3, 3, 0.99, 1.6987, 1.5038119992737817, -1.88756307418191027e-03 + 4.76128570438547885e-04im),
        (3, 3, 0.99, 1.6987, 1.7254892789542458, 1.42948067208686783e-03 - 2.16265169298574889e-03im),
    )
    for (l, m, a, omega, r, reference) in references
        Y = Y_radial(-2, l, m, a, omega, IN)
        @test Y.Y_solution(r)[1] / Y.transmission_amplitude ≈ reference rtol=1e-11
    end
end

@testset "Y_rstar: Y at the tortoise coordinate" begin
    # Y_rstar against Y_solution(r_from_rstar(r*)) where r is far enough from r_+ to hold r*. Both take the same piece
    # below r_check (horizon series, Taylor chain) and from r_two on (two infinity series), where they differ by the
    # rounding of r, about eps |r dlog Y/dr|, and X by that of x next to 1; between r_check and r_two both convert X,
    # which passes the rounding of its inputs on enlarged (to 1e-9 at (2, 1, 0.9, 1.2595)).
    # Deep in the horizon region Y_solution has no digits left; there Y ~ exp((2 kappa / r_+ - 2 i k) r*) exactly, and
    # Y_rstar is continuous where its pieces hand over.
    # Y_radial types Y by its solution, also for the mirror mode (omega < 0), so that Y_rstar is inferred; inside the
    # package a YRadialFunction keeps the untyped layout. lambda = the computed eigenvalue changes nothing.
    for (l, m, a, omega) in ((2, 2, 0.9, 0.6), (2, -2, 0.9, 0.6), (3, 1, 0.5, 1.3), G005_POINTS[3], (2, 2, 0.9, 0.01))
        for w in (omega, -omega)
            Y = Y_radial(-2, l, m, a, w, IN)
            y = w > 0 ? Y.Y_solution : Y.Y_solution.solution
            @test y isa D.RadialFunctions.RstarYSolution && y.chain !== nothing
            S = w > 0 ? D.RadialFunctions.RstarYSolution : D.ConjugatedYSolution
            @test Y isa YRadialFunction{<:S}
            @test Base.return_types(Y_rstar, (typeof(Y), Float64)) == [NTuple{3,ComplexF64}]
            @test D.y_radial_function(D.radial_route(-2, l, m, a, w, :IN)) isa YRadialFunction{Function}
            for rs in (-5.0, -1.0, 0.0, 3.0, 10.0, 40.0)
                r = r_from_rstar(a, rs)
                old = Y.Y_solution(r)
                new = Y_rstar(Y, rs)
                tol = r < y.r_check ? 1e-13 : r < y.r_two ? 1e-8 : 1e-11
                @test all(abs(new[i] / old[i] - 1) <= tol for i in 1:3)
            end
            kappa = sqrt(1 - a^2); rplus = 1 + kappa; k = w - m * a / (2rplus)
            @test Y_rstar(Y, -400.0)[1] / Y_rstar(Y, -300.0)[1] ≈ exp(-100 * (2kappa / rplus - 2im * k)) rtol=1e-11
            for r in (y.r_start, y.r_check, y.r_two)
                isfinite(r) || continue
                rs = rstar_from_r(a, r)
                @test Y_rstar(Y, rs - 1e-9)[1] ≈ Y_rstar(Y, rs + 1e-9)[1] rtol=1e-7
            end
            @test Y_rstar(Y, [-5.0, 3.0]) == [Y_rstar(Y, -5.0), Y_rstar(Y, 3.0)]
            @test Y_rstar(Y_radial(-2, l, -m, -a, w, IN), -50.0) == Y_rstar(Y, -50.0)
            Yl = Y_radial(-2, l, m, a, w, IN; lambda=Y.mode.lambda)
            @test Yl.incidence_amplitude == Y.incidence_amplitude && Yl.Y_solution(3.0) == Y.Y_solution(3.0)
        end
    end
    # Other solutions are evaluated at r_from_rstar(a, rstar) and stay untyped.
    Yup = Y_radial(2, 2, 2, 0.7, 0.5, UP)
    @test Yup isa YRadialFunction{Function}
    @test Y_rstar(Yup, 3.0) == Yup.Y_solution(r_from_rstar(0.7, 3.0))[1:3]
end
