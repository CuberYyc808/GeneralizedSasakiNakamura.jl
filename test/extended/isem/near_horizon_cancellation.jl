# Near-horizon Y at a large horizon exponent u = r_+ k / kappa: (l, m, a) = (2, 2, 0.99) at omega = -14 (mirror
# m = -2, u = 120) and -8 (u = 72). The Teukolsky horizon series cancels there (sum |terms| / |sum| up to 1e17 at
# r_+ + kappa) and Y was off by 8e-3 at r_+ + 0.75 kappa (omega = -14). References: a 256-bit Frobenius and Taylor
# solution of the Teukolsky equation (independent of the package; GSN_ISSUES_LIVE H-006), Y at unit Teukolsky
# transmission, compared as Y / transmission_amplitude. rtol 1e-6: these points take the converted Y, whose cancellation
# (about 2e6 at omega = -14) spreads the result by 3 - 5 times between platforms (1.9e-8 on arm64, 1e-7 on x86_64);
# still far below the former 8e-3. To be tightened once the G-005 fix gives Y there directly.
@testset "Y_radial near the horizon at a large horizon exponent" begin
    kappa = sqrt(1 - 0.99^2)
    references = (
        (-14.0, 0.5, 3.10914192971845939e-04 - 1.77234442642776509e-04im),
        (-14.0, 0.75, 1.93398850116654407e-04 - 8.25839677120952928e-04im),
        (-8.0, 0.5, 2.95964267993184191e-04 + 2.06191637260898420e-04im),
        (-8.0, 0.75, -3.43704929640185485e-05 - 8.57473486194463576e-04im),
    )
    for omega in (-14.0, -8.0)
        Y = Y_radial(-2, 2, 2, 0.99, omega, IN)
        for (w, f, reference) in references
            w == omega || continue
            @test Y.Y_solution(1 + kappa + f * kappa)[1] / Y.transmission_amplitude ≈ reference rtol=1e-6
        end
    end
end
