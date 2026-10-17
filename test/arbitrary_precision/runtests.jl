# Arbitrary-precision acceptance tests of the ISEM route against the Black Hole Perturbation Toolkit
# Teukolsky paclet (MST), opt-in and not part of test/runtests.jl.
#
# For every stored reference case (a, l, m, omega) and every working precision in GSN_AP_BITS the IN
# and UP solutions are computed with the public `Teukolsky_radial`, and the Teukolsky amplitude ratios
# and the radial values at r' = 3.1126842514806725 are compared with the paclet. The pass condition for
# each quantity is
#     digits agreed >= min(N log10(2) - 6 - documented loss, certified reference digits - 1),
# where the certified reference digits are the agreement between the two stored paclet working
# precisions of the same case (60 and 120 digits; 200 and 400 for omega = 10). The table printed for
# every case gives the working bits, the digits agreed, the digits expected and the time.
#
# Running (the dependency SpinWeightedSpheroidalHarmonics must support BigFloat at complex c, i.e. the
# ArbitraryPrecision branch, v1.4.1 or later):
#     GSN_AP_BITS=113,256 julia --project=<env> test/arbitrary_precision/runtests.jl
#     GSN_AP_BITS=53,113,192,256 GSN_AP_CASES=all julia --project=<env> test/arbitrary_precision/runtests.jl
# GSN_AP_CASES selects "quick" (default: a = 2/5, l = m = 2, omega = 1/2 and 1/2 - 3i/10), "real",
# "complex" or "all". GSN_SOURCE=<checkout> loads the package by include as in test/extended.

using Test
using Printf

const _AP_SOURCE = get(ENV, "GSN_SOURCE", "")
if isempty(_AP_SOURCE)
    using GeneralizedSasakiNakamura
else
    include(joinpath(_AP_SOURCE, "src", "GeneralizedSasakiNakamura.jl"))
    using .GeneralizedSasakiNakamura
end
const GSN = GeneralizedSasakiNakamura
include(joinpath(@__DIR__, "bhpt_references.jl"))

const REFERENCE_DIR = joinpath(@__DIR__, "references")
const RPRIME = 3.1126842514806725      # the binary double, as the paclet's SetPrecision pads it
const ACCEPTANCE_MARGIN_DIGITS = 6     # N log10(2) - 6, as in the brief
const KEYS = ("Incidence/Transmission", "Reflection/Transmission", "R(rprime)/Transmission", "dR(rprime)/Transmission")

# Reference pairs (lower, higher working precision) per case group.
const REFERENCE_PAIRS = (
    ("bhpt_60_a04l2.txt", "bhpt_120_a04l2.txt"),
    ("bhpt_60_a09l2.txt", "bhpt_120_a09l2.txt"),
    ("bhpt_60_l3m1.txt", "bhpt_120_l3m1.txt"),
    ("bhpt_200_omega10.txt", "bhpt_400_omega10.txt"),
    ("bhpt_60_axis.txt", "bhpt_120_axis.txt"),
)

bits_list() = parse.(Int, split(get(ENV, "GSN_AP_BITS", "113,256"), ","))

# Parse the case tag written by bhpt_acceptance_reference.wls into (a, l, m, omega) as exact numbers.
function parse_tag(tag::AbstractString)
    fields = Dict(split(part, "=", limit=2) for part in split(tag, ";"))
    a = _parse_exact(fields["a"])
    l = parse(Int, fields["l"])
    m = parse(Int, fields["m"])
    omega = _parse_exact(fields["omega"])
    return a, l, m, omega
end
function _parse_exact(text::AbstractString)
    # Mathematica InputForm of an exact number: rationals, integers, "x - (y*I)/z", "I/2", "2 - 2*I"
    t = replace(strip(text), " " => "")
    occursin("I", t) || return _parse_rational(t)
    # split at the sign before the imaginary part (the real part may be absent)
    idx = 0
    for i in length(t):-1:2
        if (t[i] == '+' || t[i] == '-') && t[i-1] != '('
            idx = i
            break
        end
    end
    real_text = idx == 0 ? "0" : t[1:idx-1]
    imag_text = idx == 0 ? t : t[idx:end]
    imag_text = replace(imag_text, "*I" => "", "I" => "1")
    imag_text = replace(imag_text, "(" => "", ")" => "")
    return _parse_rational(real_text) + im * _parse_rational(imag_text)
end
function _parse_rational(t::AbstractString)
    t = replace(t, "(" => "", ")" => "")
    occursin("/", t) || return parse(BigInt, t) // 1
    num, den = split(t, "/")
    return parse(BigInt, num) // parse(BigInt, den)      # 10^-30 appears in the axis cases
end

function selected(tag, selection)
    a, l, m, omega = parse_tag(tag)
    selection == "all" && return true
    selection == "real" && return isreal(omega)
    selection == "complex" && return !isreal(omega)
    selection == "axis" && return !isreal(omega) && iszero(real(omega)) || abs(real(omega)) < 1 // 10^20
    # quick
    return a == 2 // 5 && l == 2 && m == 2 && (omega == 1 // 2 || omega == 1 // 2 - 3im // 10)
end

certified_digits(low, high) = (low === nothing || high === nothing) ? NaN : digits_agreed(low, high)

function run_case(tag, ref_high, ref_low, bits)
    a, l, m, omega = parse_tag(tag)
    setprecision(BigFloat, bits) do
        ab = BigFloat(numerator(a)) / BigFloat(denominator(a))
        wb = isreal(omega) ? BigFloat(numerator(real(omega))) / BigFloat(denominator(real(omega))) :
            Complex{BigFloat}(BigFloat(numerator(real(omega))) / BigFloat(denominator(real(omega))),
                BigFloat(numerator(imag(omega))) / BigFloat(denominator(imag(omega))))
        rprime = BigFloat(RPRIME)
        expected = bits * log10(2) - ACCEPTANCE_MARGIN_DIGITS
        for (bc, key) in ((GSN.IN, "In"), (GSN.UP, "Up"))
            t0 = time()
            R = GSN.Teukolsky_radial(-2, l, m, ab, wb, bc)
            Rv = R.Teukolsky_solution(rprime)
            dt = time() - t0
            values = (R.incidence_amplitude, R.reflection_amplitude, Rv[1], Rv[2])
            for (quantity, value) in zip(KEYS, values)
                refkey = "$key.$quantity"
                ref = ref_high[refkey]
                ref === nothing && continue
                certified = certified_digits(ref_low[refkey], ref)
                agreed = digits_agreed(value, ref)
                required = min(expected, certified - 1)
                @printf("%-42s %3s %-26s %4d bits  agreed %6.1f  expected %6.1f  reference %6.1f  %6.2f s\n",
                    tag, key, quantity, bits, agreed, expected, certified, dt)
                @test agreed >= required
            end
        end
    end
end

@testset "arbitrary-precision ISEM vs BHPT" begin
    selection = get(ENV, "GSN_AP_CASES", "quick")
    for (low_file, high_file) in REFERENCE_PAIRS
        low = load_bhpt_refs(joinpath(REFERENCE_DIR, low_file))
        high = load_bhpt_refs(joinpath(REFERENCE_DIR, high_file))
        for tag in sort(collect(keys(high)))
            selected(tag, selection) || continue
            haskey(low, tag) || continue
            for bits in bits_list()
                run_case(tag, high[tag], low[tag], bits)
            end
        end
    end
end
