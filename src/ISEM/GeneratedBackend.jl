module GeneratedBackend

using ..Parameters: GSNParameters
import ..CoefficientTables
using ..CoefficientTables:
    RationalCoefficients,
    EndpointCoefficientSet,
    OrdinaryCoefficientEvaluator,
    CoefficientSet

export generated_backend_loaded

# Lengths of the A numerator, A denominator, B numerator and B denominator
# blocks in a packed generated coefficient vector.
struct _PQLayout
    ap_len::Int
    aq_len::Int
    bp_len::Int
    bq_len::Int
end

const _ALL_GENERATED_TABLES = (
    ("h_hc_sminus2", "endpoint_basis_matrix/gsn_endpoint_h_hc_sminus2.jl", -2),
    ("h_hc_sminus1", "endpoint_basis_matrix/gsn_endpoint_h_hc_sminus1.jl", -1),
    ("h_hc_s0", "endpoint_basis_matrix/gsn_endpoint_h_hc_s0.jl", 0),
    ("h_hc_splus1", "endpoint_basis_matrix/gsn_endpoint_h_hc_splus1.jl", 1),
    ("h_hc_splus2", "endpoint_basis_matrix/gsn_endpoint_h_hc_splus2.jl", 2),
    ("i_ic_sminus2", "endpoint_basis_matrix/gsn_endpoint_i_ic_sminus2.jl", -2),
    ("i_ic_sminus1", "endpoint_basis_matrix/gsn_endpoint_i_ic_sminus1.jl", -1),
    ("i_ic_s0", "endpoint_basis_matrix/gsn_endpoint_i_ic_s0.jl", 0),
    ("i_ic_splus1", "endpoint_basis_matrix/gsn_endpoint_i_ic_splus1.jl", 1),
    ("i_ic_splus2", "endpoint_basis_matrix/gsn_endpoint_i_ic_splus2.jl", 2),
)

const _GENERATED_TABLES = Tuple((key, file) for (key, file, _) in _ALL_GENERATED_TABLES)

for (_, file) in _GENERATED_TABLES
    include(joinpath(@__DIR__, "Generated", file))
end

const _LOADED_KEYS = Set(first.(collect(_GENERATED_TABLES)))

generated_backend_loaded() = !isempty(_LOADED_KEYS)

function _spin_tag(s::Integer)
    s == -2 && return "sminus2"
    s == -1 && return "sminus1"
    s == 0 && return "s0"
    s == 1 && return "splus1"
    s == 2 && return "splus2"
    throw(ArgumentError("ISEM GSN generated coefficients support s = -2, -1, 0, 1, 2."))
end

# Lengths of the A-numerator, A-denominator, B-numerator and B-denominator blocks
# of a generated value vector, per group (:H horizon, :I infinity, :O ordinary);
# the s = 0 tables are shorter.
function _layout_for(s::Integer, group::Symbol)
    if s == 0
        group == :H && return _PQLayout(11, 11, 11, 13)
        group == :I && return _PQLayout(10, 10, 9, 9)
        group == :O && return _PQLayout(10, 11, 9, 13)
    else
        group == :H && return _PQLayout(11, 11, 17, 19)
        group == :I && return _PQLayout(10, 10, 15, 15)
        group == :O && return _PQLayout(10, 11, 15, 19)
    end
    throw(ArgumentError("unsupported coefficient group $group."))
end

function _split_starts(layout::_PQLayout)
    ap = 1
    aq = ap + layout.ap_len
    bp = aq + layout.aq_len
    bq = bp + layout.bp_len
    return ap, aq, bp, bq
end

_view_range(values::AbstractVector, start::Int, len::Int) = @view values[start:(start + len - 1)]

function _split_endpoint(values::AbstractVector, s::Integer, group::Symbol, basis::Symbol)
    layout = _layout_for(s, group)
    ap, aq, bp, bq = _split_starts(layout)
    A = RationalCoefficients(_view_range(values, ap, layout.ap_len), _view_range(values, aq, layout.aq_len))
    B = RationalCoefficients(_view_range(values, bp, layout.bp_len), _view_range(values, bq, layout.bq_len))
    return EndpointCoefficientSet(A, B, basis)
end

# Return false if the horizon and infinity tables for s were included at load time; otherwise throw, since they
# cannot be added later.
function _ensure_generated_loaded!(s::Integer)
    tag = _spin_tag(s)
    required = ("h_hc_$tag", "i_ic_$tag")
    all(key -> key in _LOADED_KEYS, required) && return false
    throw(ArgumentError(
        "no generated coefficient tables for spin weight s=$s (supported: -2, -1, 0, 1, 2)"))
end

function _endpoint_module_symbol(s::Integer, group::Symbol)
    tag = _spin_tag(s)
    group == :H && return Symbol("GSNEndpoint_h_hc_", tag)
    group == :I && return Symbol("GSNEndpoint_i_ic_", tag)
    throw(ArgumentError("unsupported endpoint group $group."))
end

# Each generated table is looked up by name (a dynamic call), so a solve
# compiles only the tables for its spin and basis. The arguments are converted
# to the table element type before the call (ComplexF64/Float64 for binary64
# inputs, so real and complex frequencies share one compiled method per table;
# the promoted wide type otherwise), and the result is asserted to that type.
_generated_function(name::Symbol, table::Symbol) =
    getfield(getfield(@__MODULE__, table), name)

# Complex element type of the generated tables for these inputs.
@inline function _table_eltype(lambda, nu, omega)
    return promote_type(typeof(complex(float(lambda))), typeof(complex(float(omega))),
        typeof(complex(float(nu))))
end

@inline function _table_arguments(lambda, m, nu, omega)
    CT = _table_eltype(lambda, nu, omega)
    RT = real(CT)
    return CT(lambda), RT(m), RT(nu), CT(omega), CT
end

function _endpoint_values(s::Integer, group::Symbol, lambda, m, nu, omega)
    construct = _generated_function(
        :construct_gsn_endpoint_basis, _endpoint_module_symbol(s, group))
    lambda_t, m_t, nu_t, omega_t, CT = _table_arguments(lambda, m, nu, omega)
    return construct(lambda_t, m_t, nu_t, omega_t)::Vector{CT}
end

function _control_endpoint_basis(controls)
    controls === nothing && return (horizon=:hc, infinity=:ic)
    return controls.endpoint_basis
end

function _construct_coefficients(params, s::Integer, lambda, m, nu, omega; controls=nothing)
    _ensure_generated_loaded!(s)
    return _construct_coefficients_loaded(params, s, lambda, m, nu, omega, controls)
end

# Horner ranges of the ordinary-point blocks (s = 0 has shorter B polynomials), in the order of the packed P/Q
# values of the :O layout.
const _ORDINARY_RANGES = (
    1:9, 10:17, 18:24, 25:30, 31:35, 36:39, 40:42, 43:44, 45:45, 1:0, 46:55, 56:64, 65:72, 73:79, 80:85, 86:90,
    91:94, 95:97, 98:99, 100:100, 1:0, 101:115, 116:129, 130:142, 143:154, 155:165, 166:175, 176:184, 185:192,
    193:199, 200:205, 206:210, 211:214, 215:217, 218:219, 220:220, 221:237, 238:253, 254:268, 269:282, 283:295,
    296:307, 308:318, 319:328, 329:337, 338:345, 346:352, 353:358, 359:363, 364:367, 368:370, 371:372, 373:373, 1:0,
    1:0,
)
const _ORDINARY_RANGES_S0 = (
    1:4, 5:7, 8:9, 10:10, 1:0, 1:0, 1:0, 1:0, 1:0, 1:0, 11:15, 16:19, 20:22, 23:24, 25:25, 1:0, 1:0, 1:0, 1:0, 1:0,
    1:0, 26:34, 35:42, 43:49, 50:55, 56:60, 61:64, 65:67, 68:69, 70:70, 71:81, 82:91, 92:100, 101:108, 109:115,
    116:121, 122:126, 127:130, 131:133, 134:135, 136:136, 1:0, 1:0,
)

# The ordinary-point evaluator of the infinity endpoint values `i_values` (CoefficientTables.OrdinaryCoefficientEvaluator):
# the Taylor shift of q = (A_P, y A_Q, B_P, y^4 B_Q) (for s = 0 the A numerator and denominator negated, as in the
# former generated ordinary tables). Against the generated ordinary tables, both compared with a 256-bit evaluation on
# 2800 (s, l, m, a, omega, x0) points: maximum relative error 4.6e-14 against 7.3e-13.
function _ordinary_evaluator(s::Integer, i_values::AbstractVector)
    layout = _layout_for(s, :I)
    ap, aq, bp, bq = _split_starts(layout)
    T = eltype(i_values)
    sign = s == 0 ? -one(T) : one(T)
    polys = (sign .* _view_range(i_values, ap, layout.ap_len), vcat(zero(T), sign .* _view_range(i_values, aq, layout.aq_len)),
        _view_range(i_values, bp, layout.bp_len), vcat(zeros(T, 4), _view_range(i_values, bq, layout.bq_len)))
    ranges = s == 0 ? _ORDINARY_RANGES_S0 : _ORDINARY_RANGES
    coefficients = Vector{T}(undef, last(last(filter(!isempty, collect(ranges)))))
    slot = 0
    for q in polys, j in 0:(length(q) - 1)
        slot += 1
        block = ranges[slot]
        for i in 0:(length(block) - 1)
            coefficients[first(block) + i] = (isodd(j) ? -1 : 1) * binomial(j + i, j) * q[j + i + 1]
        end
    end
    return OrdinaryCoefficientEvaluator(:infinity_shift, :y, coefficients, length(ranges), ranges)
end

# Build the CoefficientSet from the generated endpoint (:hc, :ic) and
# ordinary tables. For large-frequency-expansion runs `horizon_shift` is set, so ordinary-point A and B
# are taken from a Taylor shift of the horizon rational coefficients when accurate.
function _construct_coefficients_loaded(params, s::Integer, lambda, m, nu, omega, controls)
    endpoint_basis = _control_endpoint_basis(controls)
    endpoint_basis.horizon == :hc ||
        throw(ArgumentError("only horizon endpoint basis :hc is migrated into the ISEM GSN package backend."))
    endpoint_basis.infinity == :ic ||
        throw(ArgumentError("only infinity endpoint basis :ic is migrated into the ISEM GSN package backend."))

    h_values = _endpoint_values(s, :H, lambda, m, nu, omega)
    i_values = _endpoint_values(s, :I, lambda, m, nu, omega)
    ordinary = _ordinary_evaluator(s, i_values)
    metadata = (
        source=:generated_default,
        horizon_basis=:hc,
        infinity_basis=:ic,
        ordinary_basis=ordinary.basis,
        horizon_shift=controls !== nothing &&
            getproperty(controls, :large_frequency_expansion),
        generated_loaded=generated_backend_loaded(),
    )
    return CoefficientSet(
        params, _split_endpoint(h_values, s, :H, :hc), _split_endpoint(i_values, s, :I, :ic), ordinary, metadata,
    )
end

# Build the CoefficientSet for `params` from the generated endpoint tables (bases :hc / :ic); the ordinary-point values
# are the Taylor shift of the infinity ones (_ordinary_evaluator).
function CoefficientTables.gsn_coefficients(params::GSNParameters; controls=nothing)
    return _construct_coefficients(
        params, params.s, params.lambda, params.m, params.nu, params.omega; controls=controls,
    )
end

end
