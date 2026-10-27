"""
    QNMMode(s, l, m, n, branch=:positive_real)

Label one QNM, accepted by [`qnm_frequency`](@ref) and [`qnm_sequence`](@ref)
as an alternative to the plain `(s, l, m, n, branch)` arguments. In this form
`branch=:negative_real` requires an explicit `guess`, from which the root is
polished on the negative-real branch. The plain-argument form with `mirror` and
no `guess` instead uses the reflection `-conj(ω)` of the ordinary
`(s, l, -m, n)` root at the same spin, as described in [`qnm_frequency`](@ref).
"""
struct QNMMode
    s::Int
    l::Int
    m::Int
    n::Int
    branch::Symbol

    function QNMMode(s::Int, l::Int, m::Int, n::Int,
            branch::Symbol=:positive_real)
        l >= max(abs(s), abs(m)) ||
            throw(ArgumentError("l must satisfy l >= max(abs(s), abs(m))."))
        n >= 0 || throw(ArgumentError("n must be nonnegative."))
        branch in (:positive_real, :negative_real) || throw(ArgumentError(
            "branch must be :positive_real or :negative_real."))
        return new(s, l, m, n, branch)
    end
end

function _qnm_convention(value)
    convention = Symbol(lowercase(String(value)))
    convention in (:overtone, :complete_spectrum, :leaver) ||
        throw(ArgumentError(
            "convention must be :overtone, :complete_spectrum, or :leaver."))
    return convention
end

"""
    QNMBranch

Branch selector for [`qnm`](@ref). `ordinary` selects the QNM with positive
real frequency and `mirror` the one with negative real frequency.
`Ordinary` and `Mirror` are capitalised aliases of `ordinary` and `mirror`.
"""
@enum QNMBranch begin
    ordinary
    mirror
end

const Ordinary = ordinary
const Mirror = mirror

"""
    QNMResult

Result of [`qnm`](@ref). `status` is `:accepted` when every check passes,
`:estimated` when the observables are finite but one target is missed
(`limiting_gate` names it) and `:failed` when a step fails (`stage` names the
step and `stop_reason` the reason). `error` is the estimated relative error of
the observables (the refinement drift `achieved_relative_precision`, an
indicator rather than a bound), `missing` when no observable was computed.

The `limiting_gate` of an estimated result is `:richardson_step_gate` (the
Richardson step drift of the frequency derivative exceeds its target),
`:conditioned_root_gate` (the root itself has status `:estimated`) or
`:scaled_incidence_gate` (a near-extremal or strongly damped root passes only
the relaxed incidence check `|A_inc/α|/max(1, |ω|) ≤ 1e-8`).

The stages of a failed result are `:root` (the root search failed),
`:pole_skipping_endpoint` (the frequency `-2i` at `a = 0`, where the excitation
factor is not defined), `:radial_evaluation` (an error was thrown while
computing the amplitudes; its text is in `error_message`), `:ISEM_validation`
(the incidence amplitude did not vanish to `1e-8`, or the Wronskian check
failed) and `:excitation_factor`. A failed result carries only the fields its
stage reached; with `stage = :root` a finite `omega` is a diagnostic value of
the failed root search, not a QNM frequency.
"""
struct QNMResult{D}
    data::D
end

"""
    QNMExtremal

Result of [`qnm`](@ref) for a zero-damping mode (ZDM) at exact extremality,
`a = ±1`. These co-rotating families end at the synchronous frequency
`ω = m a/2`, which is a branch point of the Green function rather than a simple
pole, so the amplitude fields are `missing`. Damped modes (DM) at `a = ±1`
return a `QNMResult`.
"""
struct QNMExtremal{D}
    data::D
end

"""
    QNMPairResult

Result of [`qnm_pair`](@ref): both branches of one mode. `ordinary` and
`mirror` each hold a `QNMResult` or `QNMExtremal`. `status` is `:failed` if either branch failed, otherwise
`:estimated` if either is estimated, otherwise `:accepted`; `stop_reason` is
`:accepted` or the reason of the first branch that is not accepted, prefixed
with `ordinary_` or `mirror_`. `a`, `s`, `l`, `m` and `n` repeat the inputs,
and `metadata` records how the mirror root was obtained.
"""
struct QNMPairResult{D}
    data::D
end

for ResultType in (QNMExtremal, QNMPairResult)
    @eval begin
        function Base.getproperty(result::$ResultType, name::Symbol)
            name === :data && return getfield(result, :data)
            return getproperty(getfield(result, :data), name)
        end

        Base.propertynames(result::$ResultType, private::Bool=false) =
            propertynames(getfield(result, :data), private)
    end
end

function Base.show(io::IO, result::QNMExtremal)
    print(io, "QNMExtremal(mode=", result.mode,
        ", frequency=", result.frequency,
        ", spectrum_object=", result.spectrum_object, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::QNMExtremal)
    println(io, "ExtremalQuasiNormalMode(")
    println(io, "    parameters = (a = ", result.a,
        ", s = ", result.mode.s, ", l = ", result.mode.l,
        ", m = ", result.mode.m, ", n = ", result.mode.n,
        _qnm_segment_suffix(result), ")")
    println(io, "    omega = ", _qnm_summary_number(result.frequency))
    println(io, "    lambda = ", _qnm_summary_number(result.lambda))
    println(io, "    family = ZDM")
    println(io, "    simple_pole = false")
    print(io, "    formalism = GSN)")
end

function Base.getproperty(result::QNMResult, name::Symbol)
    name === :data && return getfield(result, :data)
    data = getfield(result, :data)
    name === :error && !haskey(data, :error) && return _qnm_error(data)
    return getproperty(data, name)
end

Base.propertynames(result::QNMResult, private::Bool=false) =
    (propertynames(getfield(result, :data), private)..., :error)

# The estimated relative error: the refinement drift where it was reached.
_qnm_error(data) = haskey(data, :achieved_relative_precision) ?
    data.achieved_relative_precision : missing

function Base.show(io::IO, result::QNMResult)
    print(io, "QNMResult(mode=", result.mode,
        ", frequency=", result.frequency,
        ", status=", result.status, ", error=", result.error, ")")
end

_qnm_summary_number(value::Real) = @sprintf("%.17g", value)

# Segment 0 adds nothing to the display; later segments are named inline.
function _qnm_segment_suffix(result)
    segment = :segment in propertynames(result) ? result.segment : 0
    return segment == 0 ? "" : string(", segment = ", segment)
end
function _qnm_summary_number(value::Complex)
    real_text = _qnm_summary_number(real(value))
    imaginary = imag(value)
    separator = signbit(imaginary) ? " - " : " + "
    imaginary_text = _qnm_summary_number(abs(imaginary))
    return string(real_text, separator, imaginary_text, "im")
end

# Exact-extremal (a = +-1) damped modes are labelled as such; zero-damping ones are
# QNMExtremal.
_qnm_extremal(result) = abs(result.a) == 1

function Base.show(io::IO, ::MIME"text/plain", result::QNMResult)
    _qnm_has_observables(result) || return _qnm_show_partial(io, result)
    println(io, _qnm_extremal(result) ? "ExtremalQuasiNormalMode(" : "QuasiNormalMode(")
    println(io, "    parameters = (a = ", result.a,
        ", s = ", result.mode.s, ", l = ", result.mode.l,
        ", m = ", result.mode.m, ", n = ", result.mode.n,
        _qnm_segment_suffix(result), ")")
    println(io, "    omega = ", _qnm_summary_number(result.frequency))
    println(io, "    lambda = ", _qnm_summary_number(result.lambda))
    println(io, "    incidence_amplitude = ",
        _qnm_summary_number(result.incidence_amplitude))
    println(io, "    reflection_amplitude = ",
        _qnm_summary_number(result.reflection_amplitude))
    println(io, "    incidence_derivative = ",
        _qnm_summary_number(result.incidence_derivative))
    println(io, "    excitation_factor = ",
        _qnm_summary_number(result.excitation_factor))
    _qnm_extremal(result) && println(io, "    family = DM")
    _qnm_show_status(io, result)
    print(io, "    formalism = GSN)")
end

_qnm_has_observables(result) = :excitation_factor in propertynames(result)

function _qnm_show_status(io, result)
    error = result.error
    range = error === missing || !isfinite(error) ? "relative error unknown" :
        iszero(error) ? "relative error below 1e-16" :
        string("relative error ~ 1e", floor(Int, log10(error)), " to 1e",
            floor(Int, log10(error)) + 1)
    note = result.status === :accepted ? "" :
        result.status === :estimated ? string(" (", range, ")") :
        string(" (", range, "; check the result or use higher precision)")
    println(io, "    status = ", result.status, note)
    result.status === :accepted || println(io, "    stop_reason = ", result.stop_reason)
end

function Base.show(io::IO, result::QNMPairResult)
    print(io, "QNMPairResult(a=", result.a,
        ", s=", result.s, ", l=", result.l,
        ", m=", result.m, ", n=", result.n,
        ", status=", result.status, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::QNMPairResult)
    println(io, "QNMPairResult")
    println(io, "  labels:")
    println(io, "    a = ", _qnm_summary_number(result.a))
    println(io, "    s = ", result.s, ", l = ", result.l,
        ", m = ", result.m, ", n = ", result.n)
    println(io, "  ordinary:")
    println(io, "    frequency = ",
        _qnm_summary_number(result.ordinary.frequency))
    println(io, "    status = ", result.ordinary.status)
    println(io, "  mirror:")
    println(io, "    frequency = ",
        _qnm_summary_number(result.mirror.frequency))
    println(io, "    status = ", result.mirror.status)
    println(io, "  status = ", result.status)
    print(io, "  stop_reason = ", result.stop_reason)
end

"""
    LeaverResult

QNM frequency of one mode, returned by [`qnm_frequency`](@ref) and
[`qnm_sequence`](@ref) without amplitudes. `omega` is the frequency
(`ComplexF64`, or `Complex{BigFloat}` when the root needed extended precision,
see `precision_bits`), `angular_A` the angular eigenvalue in Leaver's
convention and `lambda` the separation constant of the radial Teukolsky
equation; `cf_value`, `cf_error`, `root_residual` and `angular_residual` are
residuals. `status` is `:accepted`, `:estimated` or `:failed`, with the reason
in `stop_reason`; a failed result can hold a finite `omega`, which is then a
diagnostic value of the root search, not a QNM frequency. `overtone_index` is
`missing` for `convention=:leaver`, and `provenance` records the seed, the
continuation path and the values of the checks.
"""
struct LeaverResult{T,C,A,M,P}
    mode::QNMMode
    convention::Symbol
    overtone_index::Union{Int,Missing}
    inversion_index::Union{Int,Missing}
    a::T
    omega::C
    angular_A::A
    lambda::A
    mixing::M
    cf_value::C
    cf_error::T
    cf_iterations::Int
    root_residual::T
    angular_residual::T
    precision_bits::Int
    status::Symbol
    stop_reason::Symbol
    provenance::P
end

"""
    ISEMValidationResult

Check that the GSN incidence amplitude vanishes at a QNM frequency. `incidence`,
`reflection` and `transmission` are the amplitudes of the `IN` solution at
`omega`, `absolute_incidence` is `abs(incidence)` and `scaled_incidence` is
`abs(incidence)/max(1, abs(reflection), abs(transmission))`. `selected_method`
names the radial method that produced the amplitudes (`:not_run` when the root
was neither accepted nor estimated). `status` is `:accepted` when
`scaled_incidence` is at most the incidence tolerance (`1e-8` by default) and,
where a Wronskian is used, the Wronskian is constant to its tolerance;
otherwise it is `:failed`, with the reason in `stop_reason`. For some
ill-conditioned roots the incidence check is made instead on the implied root
shift `abs(incidence/α)/max(1, abs(omega))`, with `α` the frequency derivative
of the incidence amplitude, recorded in `metadata`, which also holds the
tolerances and the evaluation details.

[`validate_qnm_with_isem`](@ref) builds it from a `GSN_radial` solution. The
`validation` field of a [`qnm`](@ref) result holds the check made by the route
that computed the excitation factor. When that is the direct route, used by
default for `s = -2, …, 2`, `|s| ≤ l ≤ 15`, `n ≤ 10` and a damped root with
nonzero real part (`selected_method = :variational`), the amplitudes come from
the same integration that gives the frequency derivative, and `transmission`
is 1.
"""
struct ISEMValidationResult{O,C,T,M}
    omega::O
    incidence::C
    reflection::C
    transmission::C
    absolute_incidence::T
    scaled_incidence::T
    selected_method::Symbol
    status::Symbol
    stop_reason::Symbol
    metadata::M
end

"""
    ExcitationFactorResult

Excitation factor of one QNM in the GSN and the Teukolsky normalization,
returned by [`qnm_excitation_factor`](@ref) and held in the `excitation` field
of a `qnm(...)` result. Fields:

- `omega`: the QNM frequency;
- `alpha`, `alpha_teukolsky`: the frequency derivative of the incidence
  amplitude at `omega`, with unit GSN and unit Teukolsky transmission
  amplitude;
- `reflection`, `reflection_teukolsky`: the reflection amplitude in the same
  two normalizations;
- `B_gsn = reflection/(2 omega alpha)` and
  `B_teukolsky = reflection_teukolsky/(2i omega alpha_teukolsky)`: the
  excitation factors;
- `step_drift`: the relative change between the two resolutions of the direct
  route, or between the Richardson steps;
- `direction_drift`: the mismatch between the derivatives along real and
  imaginary frequency (`NaN` when the derivative is not taken by differences);
- `bridge_residual`: the relative mismatch between the Teukolsky values and
  those converted from the GSN values;
- `status` (`:accepted` or `:failed`), `stop_reason` and `metadata`, which
  holds the incidence check in `metadata.validation` and the tolerances.

The drifts and the residual are indicators, not error bounds.
"""
struct ExcitationFactorResult{O,A,AT,R,RT,B,BT,T,M}
    omega::O
    alpha::A
    alpha_teukolsky::AT
    reflection::R
    reflection_teukolsky::RT
    B_gsn::B
    B_teukolsky::BT
    direction_drift::T
    step_drift::T
    bridge_residual::T
    status::Symbol
    stop_reason::Symbol
    metadata::M
end

# Thrown by `leaver_cf_inversion` when the continued fraction does not converge
# within `maximum_iterations` and convergence is required.
struct LentzConvergenceError <: Exception
    error_estimate
    iterations::Int
    maximum_iterations::Int
end

function Base.showerror(io::IO, error::LentzConvergenceError)
    print(io, "Leaver continued fraction did not converge: estimate=",
        error.error_estimate, ", iterations=", error.iterations,
        ", maximum_iterations=", error.maximum_iterations)
end

# A failed result that stopped before the observables: its stage and the root.
function _qnm_show_partial(io, result)
    println(io, _qnm_extremal(result) ? "ExtremalQuasiNormalMode(" : "QuasiNormalMode(")
    println(io, "    parameters = (a = ", result.a,
        ", s = ", result.mode.s, ", l = ", result.mode.l,
        ", m = ", result.mode.m, ", n = ", result.mode.n,
        _qnm_segment_suffix(result), ")")
    if result.frequency isa Number && isfinite(result.frequency)
        println(io, "    omega = ", _qnm_summary_number(result.frequency))
    else
        println(io, "    omega = unavailable")
    end
    :stage in propertynames(result) && println(io, "    stage = ", result.stage)
    _qnm_show_status(io, result)
    print(io, "    formalism = GSN)")
end
