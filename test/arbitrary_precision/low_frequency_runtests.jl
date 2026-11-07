# Arbitrary-precision IN solutions at real |omega| ~ 1e-10 (opt-in, not part of test/runtests.jl).
#
# There nu - l ~ 1e-20: the binary64 seed l + (nu - l) rounded to l, where the MST characteristic function is NaN
# (poles of the recurrence coefficients of n = -l, -l - 1), and the asymptotic-series fallback cancelled its matching
# determinant to exactly zero, which was measured as an infinite cancellation (InexactError: Int64(Inf) from Y_radial).
# The frequencies are two of the historical failures (s, l, m, a) = (-2, 2, 2, 0.9) read in 128 bits, plus other modes.
# Checks: Y_radial returns; the seed is not l; B_inc and (Y, dY/dr, X) at r = 3 agree between 128 and 160 bits to
# 1e-25; at 128 bits the incidence amplitudes of the MST and of the asymptotic-series infinity sides agree to 1e-25.
#
# Running: GSN_SOURCE=<checkout> julia --project=<env> test/arbitrary_precision/low_frequency_runtests.jl
# (the dependency SpinWeightedSpheroidalHarmonics must support BigFloat, v1.4.1 or later; about two minutes).

using Test

const _LF_SOURCE = get(ENV, "GSN_SOURCE", "")
if isempty(_LF_SOURCE)
    using GeneralizedSasakiNakamura
else
    include(joinpath(_LF_SOURCE, "src", "GeneralizedSasakiNakamura.jl"))
    using .GeneralizedSasakiNakamura
end
const GSN = GeneralizedSasakiNakamura
const AP = GSN.ISEM.ArbitraryPrecisionRoute

const LOW_FREQUENCY_CASES = (
    (-2, 2, 2, 0.9, "-1.480669115615162773503919452867372002534e-10"),
    (-2, 2, 2, 0.9, "1.31621544694598528971700476509155229353e-10"),
    (-2, 3, 2, 0.5, "1e-10"),
    (2, 2, 2, 0.9, "-1.3e-10"),
)

function _lf_values(s, l, m, a, omega_string, bits)
    return setprecision(BigFloat, bits) do
        omega = parse(BigFloat, omega_string)
        Y = GSN.Y_radial(s, l, m, BigFloat(a), omega, s == -2 ? GSN.IN : GSN.UP)
        Yv, dYv, Xv = Y.Y_solution(BigFloat(3))
        (Y.incidence_amplitude, Yv, dYv, Xv)
    end
end

_lf_relative(x, y) = Float64(abs(x - y) / abs(y))

@testset "AP real frequency |omega| ~ 1e-10" begin
    @test AP._cancellation(big"1.5", big"1.5") == Float64(1 / eps(BigFloat))
    for (s, l, m, a, omega_string) in LOW_FREQUENCY_CASES
        @testset "$((s, l, m, a, omega_string))" begin
            setprecision(BigFloat, 128) do
                omega = parse(BigFloat, omega_string)
                mirror = omega < 0
                p = GSN.ISEM.Parameters.gsn_parameters(s, l, mirror ? -m : m, BigFloat(a), abs(omega))
                @test first(AP._nu_seeds(p)) != l
                branch = s == -2 ? :IN : :UP
                q = GSN.ISEM.Parameters.gsn_parameters(s, l, m, BigFloat(a), omega)
                mst = AP.arbitrary_precision_route(q, branch; infinity=:mst)
                series = AP.arbitrary_precision_route(q, branch; infinity=:series)
                @test _lf_relative(series.incidence, mst.incidence) <= 1e-25
            end
            low = _lf_values(s, l, m, a, omega_string, 128)
            high = _lf_values(s, l, m, a, omega_string, 160)
            @test maximum(_lf_relative.(low, high)) <= 1e-25
        end
    end
end
