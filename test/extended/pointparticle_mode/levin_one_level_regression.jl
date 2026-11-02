# Adaptive Levin acceptance of a single passing level (bde7768): a level is final when every active segment
# resolves its own Chebyshev samples (tail <= tol), otherwise a second passing level is needed.
@testset "Levin one passing level" begin
    CI = GSN.ConvolutionIntegrals
    # Chebyshev tail of samples on Lobatto nodes: tiny for a smooth function, large for an under-resolved
    # oscillation, Inf for fewer than 4 nodes or non-finite samples, the largest over several sample sets.
    nodes(n) = [cospi(k / (n - 1)) for k in 0:n-1]
    smooth = ComplexF64.(exp.(nodes(33)))
    wiggly = ComplexF64.(cos.(40 .* nodes(33)))
    @test CI._cheb_tail(smooth) < 1e-14
    @test CI._cheb_tail(wiggly) > 1e-2
    @test CI._cheb_tail(ComplexF64[1, 2, 3]) == Inf
    @test CI._cheb_tail(ComplexF64[1, 2, NaN, 4, 5]) == Inf
    @test CI._cheb_tail(smooth, wiggly) == CI._cheb_tail(wiggly)

    scope, owned = MS.ensure_public_submission(nothing, 0.9, 8.0, 0.95, 1.0)
    MS._with_public_scope(scope, false) do
        MS.with_submission_operation(scope) do
            master = MS.submission_master!(scope)
            levin(mode; kw...) = MS.with_submission_mode(scope; mode=mode) do cache
                CI.convolution_integral_eccentric_adaptive_levin_isem(master, mode[1], mode[2], mode[3], mode[4];
                    max_depth=8, cache=cache, kw...)
            end
            # At (a, p, e, x) = (0.9, 8, 0.95, 1) the s = -2 mode (12, -11, 208) once stopped at a coarse level with
            # E = 2.1e-13 against 1.33e-34 (two-level rule with tol0 = 1e-12, 155 segments); the neighbours set the
            # scale. The floor is lowered below these fluxes so that the convergence rule, not low_flux, ends them.
            # These fluxes are round-off: the integrands cancel to 1e-17 of their size, and independent routes scatter
            # by a factor of a few (e.g. (12, -11, 208): adaptive Levin 1.3e-34 with local_n = 16 and 6.8e-35 with
            # local_n = 32, trapezoid 2.3e-35; (11, -11, 208): 2.9e-30, 1.9e-30, 8.3e-30), so only the order of
            # magnitude is checked.
            for (l, m, n, Elo, Ehi) in ((12, -11, 208, 1e-36, 1e-33), (12, -11, 207, 1e-36, 1e-33), (11, -11, 208, 1e-31, 1e-28))
                r = levin((-2, l, m, n, 0); tol=1e-8, mode_abs_floor=1e-40)
                @test r["AdaptiveLevinStopReason"] == "global_converged"
                @test Elo < r["EnergyFlux"] < Ehi
            end
            # The single-level rule is in effect: tol = -1 never accepts on the tail, which is the two-level rule.
            for mode in ((-2, 2, 2, 10, 0), (-2, 3, 1, 60, 0))
                one = levin(mode; tol=1e-8)
                two = levin(mode; tol=-1.0)
                @test one["AdaptiveLevinSegmentEvaluations"] < two["AdaptiveLevinSegmentEvaluations"]
                @test abs(one["EnergyFlux"] - two["EnergyFlux"]) <= 1e-8 * abs(two["EnergyFlux"])
            end
        end
    end
end
