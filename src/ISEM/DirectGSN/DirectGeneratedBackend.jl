module DirectGeneratedBackend

using ..DirectParameters: DirectGSNParameters
using ..DirectCoefficientTables:
    DirectRationalCoefficients,
    DirectEndpointCoefficientSet,
    DirectOrdinaryCoefficientEvaluator,
    DirectCoefficientSet

export direct_generated_coefficients, generated_backend_loaded
export preload_direct_gsn_coefficient_tables!

struct _PQLayout
    ap_len::Int
    aq_len::Int
    bp_len::Int
    bq_len::Int
end

const _ALL_GENERATED_TABLES = (
    ("h_hc_sminus2", "endpoint_basis_matrix/direct_gsn_endpoint_h_hc_sminus2.jl", -2),
    ("h_hc_sminus1", "endpoint_basis_matrix/direct_gsn_endpoint_h_hc_sminus1.jl", -1),
    ("h_hc_s0", "endpoint_basis_matrix/direct_gsn_endpoint_h_hc_s0.jl", 0),
    ("h_hc_splus1", "endpoint_basis_matrix/direct_gsn_endpoint_h_hc_splus1.jl", 1),
    ("h_hc_splus2", "endpoint_basis_matrix/direct_gsn_endpoint_h_hc_splus2.jl", 2),
    ("i_ic_sminus2", "endpoint_basis_matrix/direct_gsn_endpoint_i_ic_sminus2.jl", -2),
    ("i_ic_sminus1", "endpoint_basis_matrix/direct_gsn_endpoint_i_ic_sminus1.jl", -1),
    ("i_ic_s0", "endpoint_basis_matrix/direct_gsn_endpoint_i_ic_s0.jl", 0),
    ("i_ic_splus1", "endpoint_basis_matrix/direct_gsn_endpoint_i_ic_splus1.jl", 1),
    ("i_ic_splus2", "endpoint_basis_matrix/direct_gsn_endpoint_i_ic_splus2.jl", 2),
    ("basis_yc_sminus2", "regime_basis_full_matrix/direct_gsn_basis_yc_sminus2.jl", -2),
    ("basis_yc_sminus1", "regime_basis_full_matrix/direct_gsn_basis_yc_sminus1.jl", -1),
    ("basis_yc_s0", "regime_basis_full_matrix/direct_gsn_basis_yc_s0.jl", 0),
    ("basis_yc_splus1", "regime_basis_full_matrix/direct_gsn_basis_yc_splus1.jl", 1),
    ("basis_yc_splus2", "regime_basis_full_matrix/direct_gsn_basis_yc_splus2.jl", 2),
    ("basis_ys_sminus2", "regime_basis_full_matrix/direct_gsn_basis_ys_sminus2.jl", -2),
    ("basis_ys_sminus1", "regime_basis_full_matrix/direct_gsn_basis_ys_sminus1.jl", -1),
    ("basis_ys_s0", "regime_basis_full_matrix/direct_gsn_basis_ys_s0.jl", 0),
    ("basis_ys_splus1", "regime_basis_full_matrix/direct_gsn_basis_ys_splus1.jl", 1),
    ("basis_ys_splus2", "regime_basis_full_matrix/direct_gsn_basis_ys_splus2.jl", 2),
)

function _requested_generated_spins()
    raw = strip(get(ENV, "DIRECT_GSN_GENERATED_SPINS", ""))
    isempty(raw) && return (-2, -1, 0, 1, 2)
    spins = sort!(unique!(parse.(Int, strip.(split(raw, ',')))))
    all(s -> s in -2:2, spins) ||
        throw(ArgumentError("DIRECT_GSN_GENERATED_SPINS supports only -2,-1,0,1,2."))
    return Tuple(spins)
end

function _requested_generated_bases()
    raw = strip(get(ENV, "DIRECT_GSN_GENERATED_BASES", ""))
    isempty(raw) && return (:yc, :ys)
    bases = sort!(unique!(Symbol.(strip.(split(raw, ',')))); by=string)
    all(basis -> basis in (:yc, :ys), bases) ||
        throw(ArgumentError("DIRECT_GSN_GENERATED_BASES supports only yc,ys."))
    return Tuple(bases)
end

function _generated_table_selected(file::String, s::Integer, spins, bases)
    s in spins || return false
    occursin("regime_basis_full_matrix", file) || return true
    return any(basis -> occursin("basis_$(basis)_", file), bases)
end

const _GENERATED_SPINS = _requested_generated_spins()
const _GENERATED_BASES = _requested_generated_bases()
const _GENERATED_TABLES = Tuple(
    (key, file) for (key, file, s) in _ALL_GENERATED_TABLES
    if _generated_table_selected(file, s, _GENERATED_SPINS, _GENERATED_BASES))

for (_, file) in _GENERATED_TABLES
    include(joinpath(@__DIR__, "Generated", file))
end

const _LOADED_KEYS = Set(first.(collect(_GENERATED_TABLES)))

generated_backend_loaded() = !isempty(_LOADED_KEYS)

function preload_direct_gsn_coefficient_tables!()
    for s in _GENERATED_SPINS, basis in _GENERATED_BASES
        _ensure_generated_loaded!(s, basis)
    end
    return nothing
end

function _spin_tag(s::Integer)
    s == -2 && return "sminus2"
    s == -1 && return "sminus1"
    s == 0 && return "s0"
    s == 1 && return "splus1"
    s == 2 && return "splus2"
    throw(ArgumentError("direct GSN generated coefficients support s = -2, -1, 0, 1, 2."))
end

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

_view_range(values::Vector{ComplexF64}, start::Int, len::Int) = @view values[start:(start + len - 1)]

function _split_endpoint(values::Vector{ComplexF64}, s::Integer, group::Symbol, basis::Symbol)
    layout = _layout_for(s, group)
    ap, aq, bp, bq = _split_starts(layout)
    A = DirectRationalCoefficients(
        _view_range(values, ap, layout.ap_len),
        _view_range(values, aq, layout.aq_len),
    )
    B = DirectRationalCoefficients(
        _view_range(values, bp, layout.bp_len),
        _view_range(values, bq, layout.bq_len),
    )
    return DirectEndpointCoefficientSet(A, B, basis)
end

function _ensure_generated_loaded!(s::Integer, basis::Symbol)
    tag = _spin_tag(s)
    required = ("h_hc_$tag", "i_ic_$tag", "basis_$(basis)_$tag")
    all(key -> key in _LOADED_KEYS, required) && return false
    throw(ArgumentError(
        "generated spin sector s=$s was not loaded at top level; start a fresh " *
        "Julia process with DIRECT_GSN_GENERATED_SPINS including $s."))
end

function _ordinary_basis(nu::Float64, requested::Symbol)
    requested in (:auto, :stable_default) && return abs(sin(nu)) <= 1e-3 ? :ys : :yc
    requested in (:yc, :ys) && return requested
    throw(ArgumentError("available ordinary generated bases are :auto, :yc, and :ys."))
end

function _endpoint_module_symbol(s::Integer, group::Symbol)
    tag = _spin_tag(s)
    group == :H && return Symbol("DirectGSNEndpoint_h_hc_", tag)
    group == :I && return Symbol("DirectGSNEndpoint_i_ic_", tag)
    throw(ArgumentError("unsupported endpoint group $group."))
end

function _ordinary_module_symbol(s::Integer, basis::Symbol)
    tag = _spin_tag(s)
    basis in (:yc, :ys) && return Symbol("DirectGSNBasis_", basis, "_", tag)
    throw(ArgumentError("unsupported ordinary basis $basis."))
end

function _endpoint_values(s::Integer, group::Symbol, lambda, m, nu, omega)
    if group == :H
        s == -2 && return DirectGSNEndpoint_h_hc_sminus2.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == -1 && return DirectGSNEndpoint_h_hc_sminus1.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 0 && return DirectGSNEndpoint_h_hc_s0.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 1 && return DirectGSNEndpoint_h_hc_splus1.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 2 && return DirectGSNEndpoint_h_hc_splus2.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
    elseif group == :I
        s == -2 && return DirectGSNEndpoint_i_ic_sminus2.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == -1 && return DirectGSNEndpoint_i_ic_sminus1.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 0 && return DirectGSNEndpoint_i_ic_s0.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 1 && return DirectGSNEndpoint_i_ic_splus1.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
        s == 2 && return DirectGSNEndpoint_i_ic_splus2.construct_direct_gsn_endpoint_basis(lambda, m, nu, omega)
    end
    throw(ArgumentError("unsupported endpoint spin/group s=$s group=$group."))
end

function _ordinary_coefficients_and_eval(s::Integer, basis::Symbol, lambda, m, nu, omega)
    if basis == :yc
        s == -2 && return (
            DirectGSNBasis_yc_sminus2.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_yc_sminus2.ordinary_value_count(),
            DirectGSNBasis_yc_sminus2.eval_direct_gsn_basis!,
            DirectGSNBasis_yc_sminus2.eval_direct_gsn_basis_variable!,
        )
        s == -1 && return (
            DirectGSNBasis_yc_sminus1.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_yc_sminus1.ordinary_value_count(),
            DirectGSNBasis_yc_sminus1.eval_direct_gsn_basis!,
            DirectGSNBasis_yc_sminus1.eval_direct_gsn_basis_variable!,
        )
        s == 0 && return (
            DirectGSNBasis_yc_s0.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_yc_s0.ordinary_value_count(),
            DirectGSNBasis_yc_s0.eval_direct_gsn_basis!,
            DirectGSNBasis_yc_s0.eval_direct_gsn_basis_variable!,
        )
        s == 1 && return (
            DirectGSNBasis_yc_splus1.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_yc_splus1.ordinary_value_count(),
            DirectGSNBasis_yc_splus1.eval_direct_gsn_basis!,
            DirectGSNBasis_yc_splus1.eval_direct_gsn_basis_variable!,
        )
        s == 2 && return (
            DirectGSNBasis_yc_splus2.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_yc_splus2.ordinary_value_count(),
            DirectGSNBasis_yc_splus2.eval_direct_gsn_basis!,
            DirectGSNBasis_yc_splus2.eval_direct_gsn_basis_variable!,
        )
    elseif basis == :ys
        s == -2 && return (
            DirectGSNBasis_ys_sminus2.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_ys_sminus2.ordinary_value_count(),
            DirectGSNBasis_ys_sminus2.eval_direct_gsn_basis!,
            DirectGSNBasis_ys_sminus2.eval_direct_gsn_basis_variable!,
        )
        s == -1 && return (
            DirectGSNBasis_ys_sminus1.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_ys_sminus1.ordinary_value_count(),
            DirectGSNBasis_ys_sminus1.eval_direct_gsn_basis!,
            DirectGSNBasis_ys_sminus1.eval_direct_gsn_basis_variable!,
        )
        s == 0 && return (
            DirectGSNBasis_ys_s0.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_ys_s0.ordinary_value_count(),
            DirectGSNBasis_ys_s0.eval_direct_gsn_basis!,
            DirectGSNBasis_ys_s0.eval_direct_gsn_basis_variable!,
        )
        s == 1 && return (
            DirectGSNBasis_ys_splus1.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_ys_splus1.ordinary_value_count(),
            DirectGSNBasis_ys_splus1.eval_direct_gsn_basis!,
            DirectGSNBasis_ys_splus1.eval_direct_gsn_basis_variable!,
        )
        s == 2 && return (
            DirectGSNBasis_ys_splus2.construct_direct_gsn_basis(lambda, m, nu, omega),
            DirectGSNBasis_ys_splus2.ordinary_value_count(),
            DirectGSNBasis_ys_splus2.eval_direct_gsn_basis!,
            DirectGSNBasis_ys_splus2.eval_direct_gsn_basis_variable!,
        )
    end
    throw(ArgumentError("unsupported ordinary spin/basis s=$s basis=$basis."))
end

function _ordinary_evaluator(s::Integer, lambda, m, nu::Float64, omega, basis::Symbol, requires_latest_world::Bool)
    coeffs, count, eval!, eval_variable! =
        _ordinary_coefficients_and_eval(s, basis, lambda, m, nu, omega)
    return DirectOrdinaryCoefficientEvaluator(
        basis, :y, coeffs, count, eval!, eval_variable!,
        requires_latest_world)
end

function _control_endpoint_basis(controls)
    controls === nothing && return (horizon=:hc, infinity=:ic)
    return controls.endpoint_basis
end

function _control_ordinary_basis(controls)
    controls === nothing && return :auto
    return controls.ordinary_basis
end

function _construct_coefficients(params, s::Integer, lambda, m, nu, omega; controls=nothing)
    ordinary_basis = _ordinary_basis(Float64(nu), _control_ordinary_basis(controls))
    _ensure_generated_loaded!(s, ordinary_basis)
    return _construct_coefficients_loaded(params, s, lambda, m, nu, omega, controls, ordinary_basis, false)
end

function _construct_coefficients_loaded(params, s::Integer, lambda, m, nu, omega, controls, ordinary_basis::Symbol, requires_latest_world::Bool)
    endpoint_basis = _control_endpoint_basis(controls)
    endpoint_basis.horizon == :hc ||
        throw(ArgumentError("only horizon endpoint basis :hc is migrated into the direct GSN package backend."))
    endpoint_basis.infinity == :ic ||
        throw(ArgumentError("only infinity endpoint basis :ic is migrated into the direct GSN package backend."))

    h_values = _endpoint_values(s, :H, lambda, m, nu, omega)
    i_values = _endpoint_values(s, :I, lambda, m, nu, omega)
    ordinary = _ordinary_evaluator(s, lambda, m, Float64(nu), omega, ordinary_basis, requires_latest_world)
    metadata = (
        source=:generated_default,
        horizon_basis=:hc,
        infinity_basis=:ic,
        ordinary_basis=ordinary.basis,
        horizon_shift=controls !== nothing &&
            getproperty(controls, :lfe),
        generated_loaded=generated_backend_loaded(),
        requires_latest_world=requires_latest_world,
    )
    return DirectCoefficientSet(
        params,
        _split_endpoint(h_values, s, :H, :hc),
        _split_endpoint(i_values, s, :I, :ic),
        ordinary,
        metadata,
    )
end

function direct_generated_coefficients(params::DirectGSNParameters; controls=nothing)
    return _construct_coefficients(
        params,
        params.s,
        params.lambda,
        params.m,
        params.nu,
        params.omega;
        controls=controls,
    )
end

function direct_generated_coefficients(s::Integer, lambda, m, nu, omega; controls=nothing)
    params = (
        s=Int(s),
        lambda=ComplexF64(lambda),
        m=Int(m),
        nu=Float64(nu),
        omega=ComplexF64(omega),
    )
    return _construct_coefficients(params, s, lambda, m, Float64(nu), omega; controls=controls)
end

end
