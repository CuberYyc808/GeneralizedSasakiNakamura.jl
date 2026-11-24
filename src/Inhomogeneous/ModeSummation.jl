module ModeSummation
import ..Errors: absorbable

using KerrGeodesics
using LsqFit
using HDF5
using Printf
using ..GridSampling
using ..ConvolutionIntegrals

include("SubmissionScope.jl")
include("ModeParallelism.jl")

export circular_mode_flux
export circular_mode_summation
export eccentric_mode_summation
export inclined_mode_flux
export inclined_mode_summation
export generic_mode_summation

function _fit_shell_estimate(model, idxs, vals, next_index, p0, fallback)
    isempty(vals) && return fallback
    w = min(length(vals), 6)
    xdata = idxs[end - w + 1:end]
    ydata = vals[end - w + 1:end]
    try
        fit = curve_fit(model, xdata, ydata, p0)
        est = abs(model(next_index, fit.param))
        return isfinite(est) ? est : fallback
    catch caught
        absorbable(caught) || rethrow()
        return fallback
    end
end

_use_levin_tail(index::Int; nmin::Int = 50) = index >= nmin

function _tail_policy(value)
    value === true && return true
    value === false && return false
    value === :auto && return :auto
    throw(ArgumentError("tail_levin must be true, false, or :auto"))
end

@inline _tail_enabled(policy) = policy !== false

@inline function _uniform_radial_n(N0::Int, Nmax::Int, n::Int)
    base = min(max(4N0, 256), Nmax)
    return ConvolutionIntegrals._eccentric_fourier_min_sample(base, Nmax, n)
end

# tail_levin = :auto: a mode switches to adaptive Levin only when the radial trapezoid grid its phase needs
# (_mode_radial_trapezoid_n) exceeds Nmax, where the trapezoid route cannot resolve it. Below that the trapezoid is the
# faster route: at (a, p, e, x) = (0.99, 2.162, 0.748, 0.981) its modes on 4096..8192 radial points took 20..120 ms
# and adaptive Levin 80..1100 ms for the same flux, and the sum ran about 2x faster without Levin.
@inline _levin_auto(uniform_n::Int, Nmax::Int) = uniform_n > Nmax

# Radial trapezoid intervals a mode needs: its radial phase rate (ConvolutionIntegrals._radial_phase_min_sample),
# not 4|n|. At high eccentricity the rate is far above 4|n| (about 14 n at e = 0.95 and 30 n at e = 0.99, a = 0.9)
# and grows with the counter-rotating |m|, so the tail switch below is decided per mode from it.
function _mode_radial_trapezoid_n(KG::Dict, m::Int, n::Int, k::Int, N0::Int, Nmax::Int)
    F = KG["Frequencies"]
    ω = (m * F["ϒϕ"] + n * F["ϒr"] + k * get(F, "ϒθ", 0.0)) / F["ϒt"]
    return ConvolutionIntegrals._radial_phase_min_sample(KG, ω, m, n, _uniform_radial_n(N0, Nmax, n), typemax(Int))
end

@inline _mode_cutoff_threshold(base::Float64, mode_abs_floor::Float64) = max(base, mode_abs_floor)
@inline function _adaptive_branch_mode_floor(total_branch_energy::Float64, tol::Real, base_floor::Real)
    base = Float64(base_floor)
    base > 0 || return base
    scaled = abs(total_branch_energy) * Float64(tol)
    scaled > 0 || return base
    return min(base, scaled)
end
@inline function _generic_k_shell_threshold(total_branch_energy::Float64, reference_n_shell::Float64, tol::Real, mode_abs_floor::Float64; scale::Real = 1.0)
    total_scale = abs(total_branch_energy)
    shell_scale = abs(reference_n_shell)
    # Use a branch-level scale for k tails. The current n-shell can be tiny in the tail;
    # multiplying it by tol forces pointless scans into numerical high-k noise.
    energy_scale = max(total_scale, sqrt(total_scale * shell_scale))
    return _mode_cutoff_threshold(Float64(scale) * Float64(tol) * energy_scale, mode_abs_floor)
end
@inline function _tail_levin_mode_floor(mode_abs_floor::Real, levin_mode_abs_floor::Real)
    levin_mode_abs_floor > 0 && return Float64(levin_mode_abs_floor)
    return Float64(mode_abs_floor)
end

@inline function _positive_m_tail_below(shell_m::Float64, shell_k::Float64, threshold::Float64, tol::Real, m::Int, max_m::Int, mode_abs_floor::Float64)
    abs(shell_m) <= threshold || return false
    m > max_m + 5 || return false
    threshold <= 10.0 * max(mode_abs_floor, eps(Float64)) && return true
    return abs(shell_m) < abs(shell_k) * tol
end

@inline function _negative_m_tail_below(shell_m::Float64, threshold::Float64, neg_branch_scale::Real, m::Int, max_m::Int)
    abs(shell_m) <= neg_branch_scale * threshold || return false
    return abs(m) >= max_m
end

# At a = 0 an inclined orbit is a rotated equatorial one: mode (l, m, k) carries E_eq(l, j) |d^l_{m,j}|^2 with
# |j| = |m + sign(x) k|, so the l spectrum of a k != 0 branch vanishes below and peaks at l = |j|.
@inline _branch_peak_l(lmin::Int, m::Int, k::Int, x) = max(lmin, abs(m + (signbit(x) ? -k : k)))

# For fixed k the same rotation gives |j| = |m + sign(x) k|, which falls to 0 at m* = -sign(x) k (where omega ~ 0 and
# the shells are tiny) and rises again past it. On the side of m* an m branch counts toward the m-shell streak only
# once it is past the mirror image of m = 0 about m*, |m| >= 2|k|.
@inline _before_m_peak(m::Int, k::Int, x) = m * (signbit(x) ? k : -k) > 0 && abs(m) < 2 * abs(k)

# Eccentric shells and the k = 0 shells of generic orbits: omega = (m Ups_phi + n Ups_r) / Ups_t vanishes at
# m* = -n Ups_r / Ups_phi, where the m shells are tiny, and the modes past it (the lower sideband) carry flux again:
# on the side of m* an m shell counts toward the m streak only once |m| >= |m*|.
@inline _before_omega_zero(m::Int, m_omega_zero::Float64) = m * m_omega_zero > 0 && abs(m) < abs(m_omega_zero)

# High-l low-frequency modes with negligible flux are skipped on analytic grounds: see
# ConvolutionIntegrals._negligible_low_frequency_mode.
const _negligible_low_frequency_mode = ConvolutionIntegrals._negligible_low_frequency_mode

# Orbit classes for the sum rules, chosen once from (a, p, e, x) before a sum. Rules that decide which modes are
# computed and when a walk stops may depend on the class; the accuracy of a single mode never does.
#
# high_e (e >= HIGH_E_CLASS_MIN, classes E3 and E4): the m walks of the eccentric sum
#   - take the omega = 0 guard (_before_omega_zero) only when the crossing |m*| = n Ups_r / Ups_phi is below lmax;
#     past it the guard pinned the negative walk at m = -lmax for every n >= 40;
#   - start counting a shell's stop at the last m whose shell was above the layer threshold, plus
#     minimum_consecutive, instead of the previous walk's end (the reach ratchet, kept for every other class,
#     where it was added for nearly circular orbits near the ISSO). The ratchet pinned both walks at |m| = lmax
#     from n ~ 30 on: at (0.9, 8, 0.95, 1) and n <= 400 the two changes cut the modes by 63% and moved E_inf by
#     8.6e-10.
const HIGH_E_CLASS_MIN = 0.7
orbit_rules(a, p, e, x) = (high_e = e >= HIGH_E_CLASS_MIN,)

# An eccentric m walk counts a small m shell toward its stop only once |m| is past this fraction of the |m| the previous
# n shell's walk reached. The m shells of a shell can dip and rise again: for the nearly circular (a, p, e) =
# (0.991, 1.59, 0.029) the n = 1 shell is tiny at |m| = 25..29 and carries 1.5e-8 (2e-7 of E_inf) at l = |m| = 30..38,
# which the walk, stopping in the dip, dropped. The fraction lets the reach shrink from shell to shell.
const MWALK_REACH_FRACTION = 1.0

# A mode may stop refining once two refinement levels are below its floor; the branch floor min(1e-16, tol * branch
# total) also sets the walk thresholds, but as a refinement floor it let thousands of horizon modes near 1e-18 keep
# coarse-grid values whose errors add up: at FEWv2 569994 (a = 0.44, p = 22.7, e = 0.84) E_hor moved by 4.5e-7
# (45 tol) between tol 1e-8 and 1e-10. The refinement floor is this fraction of the branch floor (E_hor at tol 1e-8 then
# within 6e-8 of tol 1e-10, same modes); the walk thresholds are unchanged. Spherical and generic modes use it too.
const MODE_REFINE_FLOOR_FRACTION = 0.01

# Length of the below-threshold run that stops an eccentric or generic n walk: minimum_consecutive + 1, or one period
# Ups_phi / Ups_r of the n shells when that is longer (zoom-whirl, see _eccentric_mode_summation_scoped).
_n_stop_run(minimum_consecutive::Int, radial_over_azimuthal::Real) =
    max(minimum_consecutive + 1, ceil(Int, abs(1 / radial_over_azimuthal)))

# Geometric extrapolation of the eccentric n tail. At high eccentricity the n shells fall off slowly and geometrically,
# and the n walk stops (minimum_consecutive + 1 shells below tol times the total) with several tol of flux still in
# the shells it did not evaluate: at e ~ 0.8, tol = 1e-8 that tail is ~ 1e-7 of E_inf. When the walk stopped by
# itself at n >= N_TAIL_MIN_N for e >= N_TAIL_MIN_E, the ratio r of the last N_TAIL_FIT shells (a least-squares fit
# of log|shell| against n) adds shell_last * r / (1 - r), provided those shells share one sign and r < N_TAIL_MAX_RATIO.
const N_TAIL_MIN_E = 0.7
const N_TAIL_MIN_N = 100
const N_TAIL_FIT = 4
const N_TAIL_MAX_RATIO = 0.98

# Returns (tail factor r / (1 - r), r), (0.0, 0.0) when no tail is added; the tail of a flux is its last shell times the factor.
# The shells must also fall steadily over the last period (period = the n stop run): where they oscillate (zoom-whirl)
# a fit over a few shells says nothing about the next peak, and no tail is added.
function _geometric_n_tail_factor(shells::AbstractVector{<:Real}, e::Real, n_reached, period::Int = N_TAIL_FIT)
    (n_reached === nothing || e < N_TAIL_MIN_E || n_reached < N_TAIL_MIN_N) && return (0.0, 0.0)
    return _geometric_shell_tail(shells, period)
end

# Geometric tail factor r / (1 - r) of a sum whose last shells fall steadily (one sign, strictly decreasing in
# magnitude over max(N_TAIL_FIT, period + 1) shells), from a log-linear fit of the last N_TAIL_FIT shells; (0.0, 0.0)
# otherwise or when r >= N_TAIL_MAX_RATIO.
function _geometric_shell_tail(shells::AbstractVector{<:Real}, period::Int = N_TAIL_FIT)
    window = max(N_TAIL_FIT, period + 1)
    length(shells) < window && return (0.0, 0.0)
    recent = Float64.(shells[(end - window + 1):end])
    sgn = sign(recent[end])
    (sgn != 0 && all(x -> sign(x) == sgn, recent) && all(abs(recent[j + 1]) < abs(recent[j]) for j in 1:(window - 1))) || return (0.0, 0.0)
    last_shells = recent[(end - N_TAIL_FIT + 1):end]
    y = log.(abs.(last_shells))
    k = collect(0.0:(N_TAIL_FIT - 1))
    slope = sum((k .- sum(k) / N_TAIL_FIT) .* (y .- sum(y) / N_TAIL_FIT)) / sum(abs2, k .- sum(k) / N_TAIL_FIT)
    r = exp(slope)
    (0 < r < N_TAIL_MAX_RATIO) || return (0.0, 0.0)
    return (r / (1 - r), r)
end

# Defaults of the generic-orbit stopping parameters `beta` (tail budget) and `deep` (deep stop).
const DEFAULT_GENERIC_BETA = 1e-4
const DEFAULT_GENERIC_DEEP = 1e-5

# The options of a flux sum, one value for every orbit class (each sum reads the fields it uses). lmax, kmax, nmax:
# integers or :auto (see _with_auto_caps); minimum_consecutive = nothing is 5 for the circular sum and 2 otherwise;
# levin_mode_abs_floor = nothing follows mode_abs_floor. Build it with flux_options, which validates it.
Base.@kwdef struct FluxOptions
    tol::Float64 = 1e-8
    lmax::Union{Int, Symbol} = :auto
    kmax::Union{Int, Symbol} = :auto
    nmax::Union{Int, Symbol} = :auto
    minimum_consecutive::Union{Nothing, Int} = nothing
    N0::Int = 64
    K0::Int = 16
    Nmax::Int = 2^14
    Kmax::Int = 2^12
    sample_tol::Float64 = 1e-3
    record::Bool = false
    record_path::Any = nothing
    mode_abs_floor::Float64 = 1e-16
    levin_mode_abs_floor::Union{Nothing, Float64} = nothing
    threaded_sampling::Bool = false
    neg_branch_scale::Float64 = 0.1
    beta::Float64 = DEFAULT_GENERIC_BETA
    deep::Float64 = DEFAULT_GENERIC_DEEP
    tail_levin::Any = :auto
    tail_levin_infinity::Any = nothing
    tail_levin_horizon::Any = nothing
    levin_nmin::Int = 50
    levin_local_n::Int = ConvolutionIntegrals.DEFAULT_ADAPTIVE_LEVIN_LOCAL_N
    levin_max_depth::Int = 8
    mode_parallelism::Int = 1
    progress::Bool = false
end

# FluxOptions from keywords, with the aliases N (N0), K (K0) and min_consecutive (minimum_consecutive) of the sums'
# earlier signatures; every option is validated here, once.
function flux_options(; N = nothing, K = nothing, min_consecutive = nothing, kwargs...)
    o = FluxOptions(; kwargs..., (N === nothing ? () : (N0 = N,))..., (K === nothing ? () : (K0 = K,))...,
        (min_consecutive === nothing ? () : (minimum_consecutive = min_consecutive,))...)
    o.tol > 0 || throw(ArgumentError("tol must be positive"))
    o.lmax === :auto || o.lmax >= 2 || throw(ArgumentError("lmax must be at least 2"))
    o.kmax === :auto || o.kmax >= 0 || throw(ArgumentError("kmax must be nonnegative"))
    o.nmax === :auto || o.nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    any(c -> c isa Symbol && c !== :auto, (o.lmax, o.kmax, o.nmax)) && throw(ArgumentError("caps are integers or :auto"))
    something(o.minimum_consecutive, 1) >= 1 || throw(ArgumentError("minimum_consecutive must be positive"))
    all(ispow2, (o.N0, o.Nmax, o.K0, o.Kmax)) || throw(ArgumentError("N0, Nmax, K0 and Kmax must be powers of 2"))
    o.N0 <= o.Nmax || throw(ArgumentError("N0 must not exceed Nmax"))
    o.K0 <= o.Kmax || throw(ArgumentError("K0 must not exceed Kmax"))
    o.levin_local_n >= 1 || throw(ArgumentError("levin_local_n must be positive"))
    o.levin_max_depth >= 0 || throw(ArgumentError("levin_max_depth must be nonnegative"))
    o.mode_parallelism >= 1 || throw(ArgumentError("mode_parallelism must be a positive integer"))
    return o
end

# `o` with the given fields replaced.
_with(o::FluxOptions; kwargs...) =
    FluxOptions(; merge(NamedTuple{fieldnames(FluxOptions)}(map(f -> getfield(o, f), fieldnames(FluxOptions))), values(kwargs))...)

# The options of a public sum: `options` as given, or built from its keywords with the grid caps of `submission`.
_sum_options(options, submission, kwargs) = options !== nothing ? options :
    flux_options(; merge((Nmax = submission === nothing ? 2^14 : submission.Nmax,
        Kmax = submission === nothing ? 2^12 : submission.Kmax), values(kwargs))...)

# Walk ls, calling eval_l(l) (which returns the mode's energy flux), until minimum_consecutive consecutive modes have
# |flux| <= threshold, or (deep > 0) one mode has |flux| <= deep * threshold. Returns the last l walked, whether a
# stop rule ended the walk (false when it ran off the end of ls), the below-threshold streak at the end, the
# first mode's |flux| (0.0 when ls is empty), the trail and the largest |flux| before the stop. below_count, walk_max
# and trail continue a walk that ran off the end of an earlier, shorter ls (see _WalkLedgerEntry).
function _l_walk(eval_l, ls, threshold, minimum_consecutive::Int; deep = 0.0, below_count::Int = 0, walk_max::Float64 = 0.0,
        trail::Vector{Float64} = Float64[])
    last_l = first(ls) - step(ls)
    stopped = false
    first_value = 0.0
    n_walked = 0
    for l in ls
        last_l = l
        value = abs(eval_l(l))
        n_walked += 1
        n_walked == 1 && (first_value = value)
        _push_trail!(trail, value)
        below_count = value <= threshold ? below_count + 1 : 0
        if below_count >= minimum_consecutive || _deep_stop(value, deep, threshold, walk_max)
            stopped = true
            break
        end
        walk_max = max(walk_max, value)
    end
    return (last_l = last_l, stopped = stopped, below_count = below_count, first_value = first_value, trail = trail, walk_max = walk_max)
end

# Keeps the last three |values| of a walk, oldest first.
_push_trail!(trail, value) = (push!(trail, abs(value)); length(trail) > 3 && popfirst!(trail); trail)

# Tail beyond the last of the trailing |values|: v_last r / (1 - r), with r the mean ratio over the trail. When the
# values do not decrease (rounding-level wiggles far below the cutoff, or a sum still rising at the cap), the tail
# is bounded by `cap` more terms of the largest trailing value; a single value counts as one more such term.
function _geometric_tail(trail, cap)
    n = length(trail)
    n == 0 && return 0.0
    last_value = trail[end]
    n == 1 && return last_value
    first_value = trail[1]
    r = first_value > 0 ? (last_value / first_value)^(1 / (n - 1)) : Inf
    return r < 1 ? last_value * r / (1 - r) : cap * maximum(trail)
end

# A sum that reached its cap is truncated when its extrapolated tail exceeds tol times the running boundary total.
_tail_hit(tail, tol, total) = tail > tol * abs(total)

# Tail of one mode (both signs of m) beyond an l walk that ran off the end of ls without its stop rule; 0 otherwise.
@inline _walk_tail(walk, ls) = !walk.stopped && !isempty(ls) && walk.last_l == last(ls) ? 2 * _geometric_tail(walk.trail, last(ls)) : 0.0

# Walk ledger for automatic lmax and kmax raises (see _with_auto_caps). A pass records, per l walk (s, m, n, k), the
# sums of its modes and the state at which it ended; a later pass with a larger cap takes the walk's earlier modes from
# that entry instead of solving them again and only continues a walk that ran off the earlier lmax. One entry per
# walk, not per mode: no mode value is kept. How the walk ended (`code`):
const WALK_DONE = Int8(0)       # stopped by its rule: the entry is the whole walk
const WALK_UP = Int8(1)         # its last segment ran off lmax: continue upward with the walk's threshold (and deep stop)
const WALK_RECOMPUTE = Int8(2)  # its peak l lies beyond lmax: the walk's shape depends on lmax, walk it again
const WALK_BOTTOM = Int8(3)     # its bottom segment (horizon branch walks) ran off lmax: continue with search_threshold
struct _WalkLedgerEntry
    energy::Float64        # sums of 2 x the mode fluxes, in walk order
    angular::Float64
    carter::Float64
    max_abs::Float64       # largest mode |energy flux|
    tail::Float64          # the walk's extrapolated l tail (inclined and generic branch walks)
    count::Int32           # modes
    last_l::Int32          # highest l walked
    failed::Int32          # modes flagged FailedMode / NearStaticOverflow
    near_static::Int32
    code::Int8
    below::Int32           # stop state of the segment to continue: below-threshold streak,
    walk_max::Float64      # largest |flux| (deep stop),
    running::Float64       # running energy sum (eccentric m = 0 cutoff),
    trail::NTuple{3, Float64}   # last |fluxes|, oldest first (ntrail of them)
    ntrail::Int8
end
const _WalkLedger = Dict{NTuple{4, Int}, _WalkLedgerEntry}

# The stop state of a walk's segment to continue: (code, below, walk_max, running, trail).
_walk_state(code::Int8; below::Int = 0, walk_max::Float64 = 0.0, running::Float64 = 0.0, trail = ()) =
    (code = code, below = below, walk_max = walk_max, running = running, trail = trail)
const _WALK_STATE_DONE = _walk_state(WALK_DONE)
_ledger_trail(entry::_WalkLedgerEntry) = Float64[entry.trail[j] for j in 1:entry.ntrail]
_entry_state(entry::_WalkLedgerEntry) = _walk_state(entry.code; below = Int(entry.below), walk_max = entry.walk_max,
    running = entry.running, trail = _ledger_trail(entry))
# Whether a walk with ledger entry `prior` (from a pass with lmax_prior) is continued at lmax.
_continues(prior::_WalkLedgerEntry, lmax::Int, lmax_prior::Int) = lmax > lmax_prior && (prior.code == WALK_UP || prior.code == WALK_BOTTOM)

# The ledger entry of a walk: the earlier entry `prior` (or nothing), the modes this pass walked, its end state and tail.
function _ledger_entry(prior, modes, state, tail::Float64)
    energy, angular, carter = prior === nothing ? (0.0, 0.0, 0.0) : (prior.energy, prior.angular, prior.carter)
    max_abs = prior === nothing ? 0.0 : prior.max_abs
    count = prior === nothing ? 0 : Int(prior.count)
    last_l = prior === nothing ? 0 : Int(prior.last_l)
    failed = prior === nothing ? 0 : Int(prior.failed)
    near_static = prior === nothing ? 0 : Int(prior.near_static)
    for (l, mode) in modes
        E = mode["EnergyFlux"]
        energy += 2 * E
        angular += 2 * mode["AngularMomentumFlux"]
        carter += 2 * mode["CarterConstantFlux"]
        max_abs = max(max_abs, abs(E))
        count += 1
        last_l = max(last_l, l)
        get(mode, "FailedMode", false) && (failed += 1)
        get(mode, "NearStaticOverflow", false) && (near_static += 1)
    end
    tr = state.trail
    ntr = min(length(tr), 3)
    trail = ntuple(j -> j <= ntr ? Float64(tr[length(tr) - ntr + j]) : 0.0, 3)
    return _WalkLedgerEntry(energy, angular, carter, max_abs, tail, count, last_l, failed, near_static, state.code,
        state.below, state.walk_max, state.running, trail, ntr)
end

# Sums of a walk's earlier modes (its ledger entry; zeros without one): energy, angular momentum, Carter, 2 x modes.
_prior_sums(prior) = prior === nothing ? (0.0, 0.0, 0.0, 0) : (prior.energy, prior.angular, prior.carter, Int(prior.count))

# The ledger entry of walk `key` from an earlier pass with lmax `lmax_prior`, or nothing when the walk must be solved
# from its start (no entry, or its shape depended on the earlier lmax).
function _prior_walk(ledger, key, lmax::Int, lmax_prior::Int)
    ledger === nothing && return nothing
    entry = get(ledger, key, nothing)
    entry === nothing && return nothing
    entry.code == WALK_RECOMPUTE && lmax > lmax_prior && return nothing
    return entry
end

# The segment state to continue: WALK_UP / WALK_BOTTOM when the segment ran off the end of ls (lpeak beyond lmax:
# WALK_RECOMPUTE), else done.
function _segment_state(seg, ls, lpeak::Int, lmax::Int, code::Int8)
    lpeak > lmax && return _walk_state(WALK_RECOMPUTE)
    !seg.stopped && !isempty(ls) && seg.last_l == last(ls) || return _WALK_STATE_DONE
    return _walk_state(code; below = seg.below_count, walk_max = seg.walk_max, trail = copy(seg.trail))
end

# Modes within two steps of a cap, for the tail-based cap-hit test of the eccentric and generic routes, which walk l,
# m and k in many nested loops: (n, m, k) => (l, E) of an l walk, (n, k, m) => m-shell E, (n, k) => k-shell E.
struct CapTracker
    lmax::Int
    kmax::Int
    l_walks::Dict{NTuple{3, Int}, Vector{Tuple{Int, Float64}}}
    m_shells::Dict{NTuple{3, Int}, Float64}
    k_shells::Dict{NTuple{2, Int}, Float64}
end
CapTracker(lmax::Int, kmax::Int) = CapTracker(lmax, kmax, Dict{NTuple{3, Int}, Vector{Tuple{Int, Float64}}}(),
    Dict{NTuple{3, Int}, Float64}(), Dict{NTuple{2, Int}, Float64}())

function _note_cap_mode!(t::CapTracker, l::Int, m::Int, n::Int, k::Int, E)
    isfinite(E) || return t
    l >= t.lmax - 2 && push!(get!(Vector{Tuple{Int, Float64}}, t.l_walks, (n, m, k)), (l, Float64(E)))
    abs(m) >= t.lmax - 2 && (t.m_shells[(n, k, m)] = get(t.m_shells, (n, k, m), 0.0) + 2E)
    t.kmax > 0 && abs(k) >= t.kmax - 2 && (t.k_shells[(n, k)] = get(t.k_shells, (n, k), 0.0) + 2E)
    return t
end

# Tail-based hit of the walks that reached a cap, sorted by their walk index (last three values, oldest first).
function _capped_walk_hit(groups, cap, tol, total, factor)
    for v in values(groups)
        any(e -> e[1] == cap, v) || continue
        sort!(v)
        trail = [factor * abs(e[2]) for e in v[max(1, end - 2):end]]
        _tail_hit(_geometric_tail(trail, cap), tol, total) && return true
    end
    return false
end

# (lmax_hit, kmax_hit) of one boundary from its tracker and its total.
function _cap_hits(t::CapTracker, tol, total)
    lhit = _capped_walk_hit(t.l_walks, t.lmax, tol, total, 2.0)
    if !lhit
        groups = Dict{NTuple{3, Int}, Vector{Tuple{Int, Float64}}}()
        for ((n, k, m), E) in t.m_shells
            push!(get!(Vector{Tuple{Int, Float64}}, groups, (n, k, sign(m))), (abs(m), E))
        end
        lhit = _capped_walk_hit(groups, t.lmax, tol, total, 1.0)
    end
    khit = false
    if t.kmax > 0
        groups = Dict{NTuple{3, Int}, Vector{Tuple{Int, Float64}}}()
        for ((n, k), E) in t.k_shells
            push!(get!(Vector{Tuple{Int, Float64}}, groups, (n, sign(k), 0)), (abs(k), E))
        end
        khit = _capped_walk_hit(groups, t.kmax, tol, total, 1.0)
    end
    return lhit, khit
end

# An n sum that reached nmax without its stop rule is truncated when the geometric tail of its last three shells matters.
_nmax_hit(n_reached, shells, nmax, tol, total) =
    n_reached === nothing && !isempty(shells) && _tail_hit(_geometric_tail(abs.(shells[max(1, end - 2):end]), nmax), tol, total)

# Modes more than this factor below the largest mode of their l walk are taken as zeros by symmetry
# (e.g. every other l at a = 0, |flux| ~ 1e-35 next to neighbours of ~1e-7), not as a decaying tail.
const _STRUCTURAL_ZERO_RATIO = 1e-20

# Deep stop of an l walk: one mode with |flux| <= deep * threshold ends the walk, unless it is a zero by
# symmetry relative to the largest earlier mode of the walk (`walk_max`; none yet means no deep stop), in
# which case only the consecutive-mode rule can stop the walk.
@inline _deep_stop(value, deep, threshold, walk_max) =
    deep > 0 && walk_max > 0 && value <= deep * threshold && value > _STRUCTURAL_ZERO_RATIO * walk_max

# l walk of one k != 0 branch; each mode is evaluated at most once, and every segment stops after minimum_consecutive
# modes below threshold. Infinity (from_bottom = false): up from the peak, then down from the peak towards lmin.
# Walks that search for the peak use search_threshold (the plain shell cutoff) without the deep stop, since a
# larger tail budget cannot bound an unvisited interior peak between the stopped bottom and descent walks.
# Horizon (from_bottom = true): Kerr horizon modes below the peak are not suppressed and often peak at lmin, so first
# walk up from lmin as for k = 0; if that stops below the peak, walk down from the peak to where it stopped, then up
# from the peak.
# Returns the extrapolated l tail (both signs of m) beyond lmax: the geometric tail of an ascending segment that ran
# off the end of lmax, lmax times the mode at lmax when the peak lies beyond lmax and that mode is above threshold,
# 0 otherwise.
function _branch_l_walk(eval_l, lmin::Int, lpeak::Int, lmax::Int, threshold, minimum_consecutive::Int, from_bottom::Bool; deep = 0.0, search_threshold = threshold)
    top = min(lpeak, lmax + 1) - 1
    if from_bottom
        bottom = _l_walk(eval_l, lmin:lmax, search_threshold, minimum_consecutive)
        bottom.stopped || return (tail = _walk_tail(bottom, lmin:lmax), state = _segment_state(bottom, lmin:lmax, lpeak, lmax, WALK_BOTTOM))
        lpeak > bottom.last_l || return (tail = 0.0, state = _WALK_STATE_DONE)
        down = _l_walk(eval_l, top:-1:(bottom.last_l + 1), search_threshold, minimum_consecutive)
        up = _l_walk(eval_l, lpeak:lmax, threshold, minimum_consecutive; deep=deep)
        tail = lpeak > lmax && down.first_value > search_threshold ? 2 * lmax * down.first_value : _walk_tail(up, lpeak:lmax)
    else
        up = _l_walk(eval_l, lpeak:lmax, threshold, minimum_consecutive; deep=deep)
        down = _l_walk(eval_l, top:-1:lmin, search_threshold, minimum_consecutive)
        tail = lpeak > lmax && down.first_value > search_threshold ? 2 * lmax * down.first_value : _walk_tail(up, lpeak:lmax)
    end
    return (tail = tail, state = _segment_state(up, lpeak:lmax, lpeak, lmax, WALK_UP))
end

# Continuation of a branch walk (_branch_l_walk) from its ledger entry `prior`, whose segment ran off lmax_prior: that
# segment walked on to lmax with its stop state. Once it stops, the walk is complete (its peak lies at or below
# lmax_prior). Returns (tail, state) as _branch_l_walk.
function _continue_branch_l_walk(eval_l, prior::_WalkLedgerEntry, lmax_prior::Int, lmax::Int, threshold, minimum_consecutive::Int;
        deep = 0.0, search_threshold = threshold)
    bottom = prior.code == WALK_BOTTOM
    ls = (lmax_prior + 1):lmax
    seg = _l_walk(eval_l, ls, bottom ? search_threshold : threshold, minimum_consecutive; deep = bottom ? 0.0 : deep,
        below_count = Int(prior.below), walk_max = prior.walk_max, trail = _ledger_trail(prior))
    return (tail = _walk_tail(seg, ls), state = _segment_state(seg, ls, 0, lmax, prior.code))
end

# Adaptive Levin modes of the eccentric sum must also change by less than LEVIN_ABS_FRACTION tol times the
# branch total between refinement levels. Without it the relative test (1e-6 relaxed to 1e-2 with depth) stopped
# the dominant mode of a shell early: (l, m, n) = (2, 2, 400) at (0.9, 8, 0.95, 1) was off by 1.7e-5.
const LEVIN_ABS_FRACTION = 0.01

# Automatic l and k caps (lmax = :auto, kmax = :auto).
const AUTO_CAP_MIN = 30
const AUTO_CAP_MAX = 150
const AUTO_CAP_MARGIN = 3
const AUTO_CAP_MAX_RAISES = 2

# Energy flux |F_inf| + |F_hor| of the (l, m = l) modes, l = 2..10, of the equatorial circular orbit at radius r, or
# nothing when that orbit cannot be evaluated.
function _auto_cap_probe_fluxes(a, r, x_probe)
    try
        scope, owned = ensure_public_submission(nothing, a, r, 0.0, x_probe)
        return _with_public_scope(scope, owned) do
            with_submission_operation(scope) do
                map(2:10) do l
                    mode = _circular_mode_flux_scoped(scope, a, r, l, l; x = x_probe)
                    abs(mode.infinity.energy_flux) + abs(mode.horizon.energy_flux)
                end
            end
        end
    catch caught
        absorbable(caught) || rethrow()
        return nothing
    end
end

# Upper estimate of the l needed for tol, from the l falloff of the equatorial circular orbit at the same a and
# radius r = p / (1 + e) (prograde for x >= 0, retrograde otherwise, prograde if that fails). A linear fit of
# log F_l over l = 6..10 gives the slope s < 0; the tail beyond L is F_10 exp(s (L - 10)) / (1 - e^s). The cap is
# the smallest L with that tail below tol * sum(F_2..F_10), plus AUTO_CAP_MARGIN, clamped to
# [AUTO_CAP_MIN, AUTO_CAP_MAX]. Returns (cap, status).
function auto_lmax_estimate(a, p, e, x; tol = 1e-8)
    r = p / (1 + e)
    probes = x >= 0 ? (1.0,) : (-1.0, 1.0)
    F = nothing
    for x_probe in probes
        F = _auto_cap_probe_fluxes(a, r, x_probe)
        F !== nothing && all(f -> isfinite(f) && f > 0, F) && break
        F = nothing
    end
    F === nothing && return (cap = AUTO_CAP_MIN, status = :probe_failed)
    ls = 6:10
    ys = log.(F[ls .- 1])
    lbar = sum(ls) / length(ls)
    ybar = sum(ys) / length(ys)
    slope = sum((ls .- lbar) .* (ys .- ybar)) / sum((ls .- lbar) .^ 2)
    (isfinite(slope) && slope < 0) || return (cap = AUTO_CAP_MAX, status = :no_falloff)
    budget = tol * sum(F)
    tail_scale = F[end] / (1 - exp(slope))
    L = tail_scale <= budget ? 10 : 10 + ceil(Int, log(budget / tail_scale) / slope)
    return (cap = clamp(L + AUTO_CAP_MARGIN, AUTO_CAP_MIN, AUTO_CAP_MAX), status = :ok)
end

# Initial caps: integers pass through; :auto takes the estimate (kmax = lmax estimate).
const AUTO_NMAX_MIN = 100
const AUTO_NMAX_MAX = 20000
const AUTO_NMAX_COEFF = 0.75

# Radial harmonic content: the azimuthal rate at periapsis r_p = p / (1 + e) (at the equatorial crossing for generic
# orbits) over the radial frequency. It grows without bound towards the separatrix (Omega_r -> 0).
function _radial_harmonic_scale(a, p, e, x)
    KG = kerr_geo_orbit(a, p, e, x)
    E = KG["Energy"]
    Lz = KG["AngularMomentum"]
    F = KG["Frequencies"]
    r = p / (1 + e)
    Δ = r^2 - 2r + a^2
    P = E * (r^2 + a^2) - a * Lz
    dφ = a * P / Δ - a * E + Lz
    dt = (r^2 + a^2) * P / Δ - a * (a * E - Lz)
    return abs(dφ / dt) / abs(F["ϒr"] / F["ϒt"])
end

# nmax = c_n (lmax + log(1/tol)) w_p / Omega_r, clamped to [AUTO_NMAX_MIN, AUTO_NMAX_MAX]. c_n = 0.75 gives at least
# 1.5 times the n reached by (0.9, 8, 0.99, 1) at tol 1e-6 (8436), (0.9, 8, 0.95, 1) (1005), (0.9, 6, 0.7, 1) at
# 1e-8 (124) and (0.5, p_sep + 0.05, 0.8, 1) (388).
function auto_nmax_estimate(a, p, e, x, tol, lmax)
    scale = try
        _radial_harmonic_scale(a, p, e, x)
    catch caught
        absorbable(caught) || rethrow()
        return (cap = AUTO_NMAX_MAX, status = :scale_failed)
    end
    isfinite(scale) || return (cap = AUTO_NMAX_MAX, status = :scale_failed)
    return (cap = clamp(ceil(Int, AUTO_NMAX_COEFF * (lmax + log(1 / tol)) * scale), AUTO_NMAX_MIN, AUTO_NMAX_MAX), status = :ok)
end

# Initial caps: integers pass through; :auto takes the estimates (kmax = lmax estimate; nmax from the radial
# harmonic scale with the lmax used). kmax, nmax = nothing for routes without a k or n sum. Validates tol and the caps
# for every sum.
function _auto_caps_start(a, p, e, x, tol, lmax, kmax, nmax = nothing)
    auto_l = lmax === :auto
    auto_k = kmax === :auto
    auto_n = nmax === :auto
    tol > 0 || throw(ArgumentError("tol must be positive"))
    auto_l || lmax >= 2 || throw(ArgumentError("lmax must be at least 2"))
    auto_k || kmax === nothing || kmax >= 0 || throw(ArgumentError("kmax must be nonnegative"))
    auto_n || nmax === nothing || nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    est = (cap = nothing, status = :fixed)
    if auto_l || auto_k
        est = auto_lmax_estimate(a, p, e, x; tol)
        if est.status == :probe_failed
            @warn "Automatic lmax/kmax: the equatorial circular probe at r = p / (1 + e) failed; using $(AUTO_CAP_MIN)." a p e x
        elseif est.status == :no_falloff
            @warn "Automatic lmax/kmax: the probe shows no l falloff by l = 10; using $(AUTO_CAP_MAX)." a p e x
        end
    end
    L = auto_l ? est.cap : lmax
    nest = (cap = nothing, status = :fixed)
    if auto_n
        nest = auto_nmax_estimate(a, p, e, x, tol, L)
        nest.status == :scale_failed && @warn "Automatic nmax: the radial harmonic scale could not be evaluated; using $(AUTO_NMAX_MAX)." a p e x
    end
    return (lmax = L, kmax = auto_k ? est.cap : kmax, nmax = auto_n ? nest.cap : nmax, auto_l, auto_k, auto_n,
        estimate = est.cap, n_estimate = nest.cap, status = est.status)
end

_result_lmax_hit(result, lmax) = get(result, :lmax_hit_inf, false) || get(result, :lmax_hit_hor, false) ||
    (haskey(result, :l_reached_inf) && !haskey(result, :lmax_hit_inf) && (something(result.l_reached_inf, 0) >= lmax || something(get(result, :l_reached_hor, 0), 0) >= lmax))
_result_kmax_hit(result) = get(result, :kmax_hit_inf, false) || get(result, :kmax_hit_hor, false)
_result_nmax_hit(result) = get(result, :nmax_hit_inf, false) || get(result, :nmax_hit_hor, false)

# Run run(lmax, kmax, nmax, resume) with the start caps; when a cap came from :auto and the sum hit it, raise
# that cap by half (lmax and kmax at most AUTO_CAP_MAX, nmax at most AUTO_NMAX_MAX) and sum again, at most
# AUTO_CAP_MAX_RAISES times. Adds the caps to the result.
#
# A raise does not sum again from the start. resume = (prev, track): prev is the previous pass's `_resume` (nothing on
# the first pass), track whether a later raise is still possible, so that the pass keeps what a raise continues from.
#   - When the raised cap only extends the outermost shell loop (nmax of the eccentric and generic sums, kmax of the
#     inclined sum, lmax of the circular sum), the pass continues that loop from where the previous one stopped, with
#     the same result as a sum from the start.
#   - When lmax (or the generic kmax) is raised, the pass walks the shells again, but takes each l walk of the previous
#     pass from its walk ledger (sums and stop state per walk, see _WalkLedgerEntry) and solves only the modes past the
#     previous lmax of the walks that ran off it, and the walks that pass did not reach. The m, k and n stop rules are
#     applied to the combined sums. The result agrees with a sum from the start within the stop tolerance (a walk
#     stopped below the previous lmax is not walked again under the new pass's slightly different cutoffs).
# modes_solved: the (mode, boundary) amplitudes computed over all passes (as total_modes).
function _with_auto_caps(run, caps)
    L, Kc, Nc = caps.lmax, caps.kmax, caps.nmax
    auto = caps.auto_l || caps.auto_k || caps.auto_n
    raises = 0
    can_raise = (L, Kc, Nc, raises) -> auto && raises < AUTO_CAP_MAX_RAISES && ((caps.auto_l && L < AUTO_CAP_MAX) ||
        (caps.auto_k && Kc isa Integer && Kc < AUTO_CAP_MAX) || (caps.auto_n && Nc isa Integer && Nc < AUTO_NMAX_MAX))
    result = run(L, Kc, Nc, (prev = nothing, track = can_raise(L, Kc, Nc, raises)))
    modes_solved = get(result, :modes_new, result.total_modes)
    while auto && raises < AUTO_CAP_MAX_RAISES
        raise_l = caps.auto_l && L < AUTO_CAP_MAX && _result_lmax_hit(result, L)
        raise_k = caps.auto_k && Kc isa Integer && Kc < AUTO_CAP_MAX && _result_kmax_hit(result)
        raise_n = caps.auto_n && Nc isa Integer && Nc < AUTO_NMAX_MAX && _result_nmax_hit(result)
        (raise_l || raise_k || raise_n) || break
        raise_l && (L = min(AUTO_CAP_MAX, ceil(Int, 1.5L)))
        raise_k && (Kc = min(AUTO_CAP_MAX, ceil(Int, 1.5Kc)))
        raise_n && (Nc = min(AUTO_NMAX_MAX, ceil(Int, 1.5Nc)))
        raises += 1
        result = run(L, Kc, Nc, (prev = get(result, :_resume, nothing), track = can_raise(L, Kc, Nc, raises)))
        modes_solved += get(result, :modes_new, result.total_modes)
    end
    result = merge(Base.structdiff(result, NamedTuple{(:_resume, :modes_new)}), (modes_solved = modes_solved,))
    # Every orbit class, fixed or automatic caps.
    how = auto ? " after $(raises) automatic raise(s)" : "; pass a larger cap"
    _result_lmax_hit(result, L) && @warn "Mode summation hit lmax = $(L) with modes still above the cutoff$(how): the l sum is truncated." lmax = L
    _result_kmax_hit(result) && @warn "Mode summation ran to kmax = $(Kc) without converged k shells$(how): the k sum may be truncated." kmax = Kc
    _result_nmax_hit(result) && @warn "Mode summation ran to nmax = $(Nc) without converged n shells$(how): the n sum may be truncated." nmax = Nc
    return merge(result, (lmax_used = L, kmax_used = Kc, nmax_used = Nc, lmax_auto = caps.auto_l, kmax_auto = caps.auto_k,
        nmax_auto = caps.auto_n, cap_estimate = caps.estimate, nmax_estimate = caps.n_estimate, cap_status = caps.status,
        cap_raises = raises))
end

# Mode metadata written when present: the adaptive Levin leaves and the Y interpolation.
const _RECORDED_MODE_KEYS = ("AdaptiveLevin", "AdaptiveLevinMaxDepth", "AdaptiveLevinMaxLeafDepth",
    "AdaptiveLevinEffectiveIntervals", "AdaptiveLevinEffectiveRIntervals", "AdaptiveLevinEffectiveThetaIntervals",
    "AdaptiveLevinAcceptedSegments", "AdaptiveLevinSegmentEvaluations", "AdaptiveLevinSplitCount",
    "AdaptiveLevinStopReason", "AdaptiveLevinUnconverged", "YInterpolationPoints", "YInterpolationDoublings",
    "YInterpolationFallback", "YInterpolationSelfCheckError")

_record_group(parent, key) = haskey(parent, key) ? parent[key] : create_group(parent, key)

# One mode of a recorded sum, at [n_<n>/][k_<k>/]m_<m>/l_<l>/<boundary>: n for the eccentric and generic sums, k for
# the generic (signed format) and inclined (unsigned format) sums. N_sample / K_sample: the grid written when the mode
# does not carry its own. Counted only when recording: with mode_parallelism the two boundaries run as tasks and would
# otherwise race on the shared counter.
function _record_current_mode!(h5, record::Bool, boundary::AbstractString, mode_index::Base.RefValue{Int}, mode;
        l::Int, m::Int, n = nothing, k = nothing, signed_k::Bool = true, N_sample = nothing, K_sample = nothing)
    record || return nothing
    mode_index[] += 1
    group = h5
    n === nothing || (group = _record_group(group, @sprintf("n_%06d", n)))
    k === nothing || (group = _record_group(group, signed_k ? @sprintf("k_%+06d", k) : @sprintf("k_%06d", k)))
    bgrp = _record_group(_record_group(_record_group(group, @sprintf("m_%+06d", m)), @sprintf("l_%04d", l)), boundary)
    bgrp["source"] = "julia"
    bgrp["mode_index"] = mode_index[]
    bgrp["boundary"] = boundary
    bgrp["s"] = boundary == "I" ? -2 : 2
    bgrp["l"] = l
    bgrp["m"] = m
    n === nothing || (bgrp["n"] = n)
    k === nothing || (bgrp["k"] = k)
    N_sample === nothing || (bgrp["N_sample"] = get(mode, "N_sample", N_sample))
    K_sample === nothing || (bgrp["K_sample"] = get(mode, "K_sample", K_sample))
    bgrp["Amplitude_re"] = real(get(mode, "Amplitude", 0.0 + 0.0im))
    bgrp["Amplitude_im"] = imag(get(mode, "Amplitude", 0.0 + 0.0im))
    bgrp["EnergyFlux"] = get(mode, "EnergyFlux", NaN)
    bgrp["AngularMomentumFlux"] = get(mode, "AngularMomentumFlux", NaN)
    bgrp["CarterConstantFlux"] = get(mode, "CarterConstantFlux", NaN)
    for key in _RECORDED_MODE_KEYS
        haskey(mode, key) && (bgrp[key] = mode[key])
    end
    return nothing
end

# Classify the orbit as the public ISEM route does (1e-12 tolerance), before allocating a mode cache.
_mode_flux_cache_family(e, x) = kerr_geo_orbit_shape(e, x)

# Snapshot the parent under its existing lock; invoke user code outside the lock.
function _mode_flux_parent_stop_requested(scope)
    _public_cancelled(scope) && return true
    callback = lock(() -> scope.stop_requested, scope.lock)
    return callback !== nothing && callback() === true
end

function _circular_mode_flux_scoped(scope, a, p, l, m; x=1.0)
    family = _mode_flux_cache_family(0.0, x)
    share_geometry = isequal(scope.parameters[3], 0.0)
    flux_inf = with_submission_mode(scope; family = family, share_geometry = share_geometry, mode = (-2, l, m, 0, 0)) do mode_cache
        ConvolutionIntegrals.convolution_integral_trapezoidal_isem(a, p, 0.0, x, -2, l, m, 0, 0; cache = mode_cache, Nmax = scope.Nmax, Kmax = scope.Kmax, trajectory = share_geometry ? submission_orbit!(scope) : nothing)
    end
    flux_hor = with_submission_mode(scope; family = family, share_geometry = share_geometry, mode = (2, l, m, 0, 0)) do mode_cache
        ConvolutionIntegrals.convolution_integral_trapezoidal_isem(a, p, 0.0, x, 2, l, m, 0, 0; cache = mode_cache, Nmax = scope.Nmax, Kmax = scope.Kmax, trajectory = share_geometry ? submission_orbit!(scope) : nothing)
    end
    return (
        l = l,
        m = m,
        infinity = (
            energy_flux = flux_inf["EnergyFlux"],
            angular_momentum_flux = flux_inf["AngularMomentumFlux"],
            carter_constant_flux = flux_inf["CarterConstantFlux"],
        ),
        horizon = (
            energy_flux = flux_hor["EnergyFlux"],
            angular_momentum_flux = flux_hor["AngularMomentumFlux"],
            carter_constant_flux = flux_hor["CarterConstantFlux"],
        ),
    )
end

# Infinity and horizon fluxes of the circular-orbit mode (l, m) (s = -2 and s = +2), in a submission scope.
function circular_mode_flux(a, p, l, m; submission = nothing, x=1.0, stop_requested = nothing, Nmax = submission === nothing ? 2^14 : submission.Nmax, Kmax = submission === nothing ? 2^12 : submission.Kmax)
    scope, owned = ensure_public_submission(submission, a, p, 0.0, x; Nmax, Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        with_submission_operation(scope) do
            _circular_mode_flux_scoped(scope, a, p, l, m; x = x)
        end
    end
end

# The l shells of one boundary of the circular sum (`flux(l, m)` its mode flux), from `state` (the totals before the l
# tail, the shell list, the last shell, the stop counter and the last l summed; nothing yet: from lmin) up to lmax or
# until min_consecutive shells are below tol times the running total. Returns the new state.
function _circular_branch_shells(flux, state, lmin, lmax, tol, min_consecutive)
    (; energy, angular, carter, shells, last, stopped, below, l_reached) = state
    stopped && return state
    for l in (l_reached === nothing ? lmin : l_reached + 1):lmax
        shell_energy = 0.0
        shell_angular = 0.0
        shell_carter = 0.0
        for m in 1:l
            f = flux(l, m)
            shell_energy += 2 * f.energy_flux
            shell_angular += 2 * f.angular_momentum_flux
            shell_carter += 2 * f.carter_constant_flux
        end
        l_reached = l
        energy += shell_energy
        angular += shell_angular
        carter += shell_carter

        push!(shells, shell_energy)
        last = (shell_energy, shell_angular, shell_carter)
        threshold = tol * abs(energy)
        below = abs(shell_energy) <= threshold ? below + 1 : 0
        if below >= min_consecutive
            stopped = true
            break
        end
    end
    return (; energy, angular, carter, shells, last, stopped, below, l_reached)
end

_circular_branch_start() = (energy = 0.0, angular = 0.0, carter = 0.0, shells = Float64[], last = (0.0, 0.0, 0.0),
    stopped = false, below = 0, l_reached = nothing)

# `resume`: the `_resume` of a pass with a smaller lmax (see _with_auto_caps); each boundary that had not stopped
# continues from its last l shell.
function _circular_mode_summation_scoped(scope, a, p; x=1.0, tol = 1e-8, lmax = 30, min_consecutive = 5, resume = nothing)
    lmin = 2

    mode_cache = Dict{Tuple{Int, Int}, Any}()
    get_mode(l, m) = get!(mode_cache, (l, m)) do
        circular_mode_flux(a, p, l, m; x, submission = scope)
    end
    prev = resume === nothing ? nothing : resume.prev
    resumed = prev isa NamedTuple && get(prev, :kind, nothing) === :circular
    inf = _circular_branch_shells((l, m) -> get_mode(l, m).infinity, resumed ? prev.infinity : _circular_branch_start(),
        lmin, lmax, tol, min_consecutive)
    hor = _circular_branch_shells((l, m) -> get_mode(l, m).horizon, resumed ? prev.horizon : _circular_branch_start(),
        lmin, lmax, tol, min_consecutive)
    l_reached_inf = inf.l_reached
    l_reached_hor = hor.l_reached

    # Geometric l tail beyond the stop (as the n tail of eccentric orbits): the l shells of a circular orbit fall off
    # geometrically, and min_consecutive shells below tol can still leave a few tol in the shells not evaluated.
    l_tail_factor_inf, l_tail_ratio_inf = inf.stopped ? _geometric_shell_tail(inf.shells, min_consecutive) : (0.0, 0.0)
    l_tail_factor_hor, l_tail_ratio_hor = hor.stopped ? _geometric_shell_tail(hor.shells, min_consecutive) : (0.0, 0.0)
    l_tail_inf = l_tail_factor_inf .* inf.last
    l_tail_hor = l_tail_factor_hor .* hor.last
    total_infinity_energy = inf.energy + l_tail_inf[1]
    total_infinity_angular = inf.angular + l_tail_inf[2]
    total_infinity_carter = inf.carter + l_tail_inf[3]
    total_horizon_energy = hor.energy + l_tail_hor[1]
    total_horizon_angular = hor.angular + l_tail_hor[2]
    total_horizon_carter = hor.carter + l_tail_hor[3]

    total_modes = 2 * sum(lmin:max(something(l_reached_inf, lmin - 1), something(l_reached_hor, lmin - 1)))

    return (
        infinity_energy_flux = total_infinity_energy, infinity_angular_momentum_flux = total_infinity_angular,
        infinity_carter_constant_flux = total_infinity_carter, horizon_energy_flux = total_horizon_energy,
        horizon_angular_momentum_flux = total_horizon_angular, horizon_carter_constant_flux = total_horizon_carter,
        total_modes = total_modes, tolerance = tol, lmin = lmin, lmax = lmax, min_consecutive = min_consecutive,
        l_reached = max(something(l_reached_inf, lmin - 1), something(l_reached_hor, lmin - 1)),
        l_reached_inf = l_reached_inf, l_reached_hor = l_reached_hor,
        l_tail_infinity = (energy = l_tail_inf[1], angular_momentum = l_tail_inf[2], carter_constant = l_tail_inf[3], ratio = l_tail_ratio_inf),
        l_tail_horizon = (energy = l_tail_hor[1], angular_momentum = l_tail_hor[2], carter_constant = l_tail_hor[3], ratio = l_tail_ratio_hor),
        _resume = (kind = :circular, infinity = inf, horizon = hor, total_modes = total_modes),
        modes_new = total_modes - (resumed ? prev.total_modes : 0),
    )
end

# Total circular-orbit fluxes: sum m = 1..l (doubled for -m) over l shells until min_consecutive shells are
# below tol times the running total, separately at infinity and at the horizon.
# Total circular equatorial (or polar circular, x = -1) fluxes, summed over l shells. Options: FluxOptions keywords
# (tol, lmax, minimum_consecutive = 5), or `options`.
function circular_mode_summation(a, p; submission = nothing, x = 1.0, stop_requested = nothing, options = nothing, kwargs...)
    o = _sum_options(options, submission, kwargs)
    caps = _auto_caps_start(a, p, 0.0, x, o.tol, o.lmax, nothing)
    scope, owned = ensure_public_submission(submission, a, p, 0.0, x; Nmax = o.Nmax, Kmax = o.Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        _with_auto_caps(caps) do L, _, _, resume
            with_submission_operation(scope) do
                _circular_mode_summation_scoped(scope, a, p; x, tol = o.tol, lmax = L,
                    min_consecutive = something(o.minimum_consecutive, 5), resume)
            end
        end
    end
end

_record_tag(x) = replace(replace(@sprintf("%.16g", x), "." => "p"), "-" => "m")

const _PROGRESS_LOCK = ReentrantLock()

# One l walk of the eccentric sum: the modes (s, l, m, n), l = max(2, |m|)..lmax, until minimum_consecutive modes are
# below the shell cutoff, from the shell's parameters q only (max_flux, use_levin, floor, levin_floor, levin_abs of
# the modes; branch_floor and threshold of the cutoff: tol times the previous n shell, or, for m = 0, tol times the
# running m shell when threshold === nothing). Computed ahead (`strip`, which also drops the large members of the mode
# Dicts) with q.l_pred, the last l of this m in the previous shell, the modes up to l_pred are computed on tasks of
# their own at once (the walk is otherwise the critical path of a shell: at e = 0.7 the m = 0 walk alone took a third
# of the shell's time on 8 threads), the stop rule applied to them in order; a mode beyond the stop is waited for and
# dropped (counted in `unused`). Returns (modes = the (l, mode) in walk order, unused, prior, state, tail). A mode's
# non-finite flux is the error of the walk. With `prior`, the walk's ledger entry from a pass with lmax_prior, only
# a walk that ran off lmax_prior is continued (modes = its new modes).
function _eccentric_l_walk(compute, s::Int, m::Int, n::Int, q, lmax::Int, minimum_consecutive::Int, tol::Float64, strip::Bool;
        prior = nothing, lmax_prior::Int = 0, arena = ConvolutionIntegrals._series_arena())
    lmin = max(2, abs(m))
    running = 0.0
    below = 0
    if prior !== nothing
        _continues(prior, lmax, lmax_prior) || return (modes = Tuple{Int, Any}[], unused = 0, prior = prior,
            state = _entry_state(prior), tail = 0.0)
        lmin = lmax_prior + 1
        running = prior.running
        below = Int(prior.below)
    end
    l_pred = strip && prior === nothing ? q.l_pred : nothing
    speculate = l_pred !== nothing && l_pred > lmin
    factors = speculate ? ConvolutionIntegrals._SharedFactors() : Dict{Any, Any}()
    ahead = Dict{Int, Task}()
    # `arena`: the work arena of the closed-form series for the modes this walk computes itself (a task computing ahead
    # gets its own)
    if speculate
        for l in lmin:min(lmax, l_pred)
            ahead[l] = Threads.@spawn compute(s, l, m, n, q, factors, ConvolutionIntegrals._series_arena())
        end
    end
    modes = Tuple{Int, Any}[]
    stopped = false
    for l in lmin:lmax
        task = pop!(ahead, l, nothing)
        mode = task === nothing ? compute(s, l, m, n, q, factors, arena) : _fetch_mode(task)
        ConvolutionIntegrals._check_finite_mode_flux(mode, s, l, m, n, 0)
        strip && _strip_mode!(mode)
        push!(modes, (l, mode))
        E = mode["EnergyFlux"]
        running += 2 * E
        base = q.threshold === nothing ? tol * abs(running) : q.threshold
        below = abs(E) <= _mode_cutoff_threshold(base, q.branch_floor) ? below + 1 : 0
        below >= minimum_consecutive && (stopped = true; break)
    end
    unused = 0
    for (_, task) in ahead
        # beyond the stop: the serial walk never computes these, so their errors do not count
        try
            wait(task)
        catch caught
            absorbable(caught) || rethrow()
        end
        unused += 1
    end
    state = stopped ? _WALK_STATE_DONE : _walk_state(WALK_UP; below = below, running = running)
    return (modes = modes, unused = unused, prior = prior, state = state, tail = 0.0)
end

function _fetch_mode(task::Task)
    try
        return fetch(task)
    catch error
        error isa TaskFailedException ? throw(error.task.result) : rethrow()
    end
end

# The n >= 1 shells of one boundary of the eccentric sum (s = -2: infinity, 2: horizon), after its n = 0 shell: the
# m = 0, m = 1, 2, .. and m = -1, -2, .. l walks of each shell, the m and n stop rules, the shell lists and the totals.
# The l walks depend on the shell's parameters only: with `prefetch` they are computed ahead, planned from the m reach
# of the previous shell. The two boundaries may run on different tasks: everything mutable here is local or owned by
# this boundary (ctx). Returns (total energy, angular momentum, Carter constant, n_reached, last shell, modes, state);
# state is what the next shell needs, from which a pass with a larger nmax continues (ctx.resume).
function _eccentric_branch_shells!(s::Int, ctx, prefetch)
    (; lmax, nmax, minimum_consecutive, n_stop_run, radial_over_azimuthal, high_e, e, tol, model, p0, mode_abs_floor,
        n0_reach, energy_list, angular_list, carter_list, n_list, floor_ref, levin_abs_ref, run) = ctx
    r = ctx.resume
    r === nothing || r.n_reached === nothing || return (r.totals..., r.n_reached, r.last_shell, r.total_modes, r)
    total_energy, total_angular, total_carter = r === nothing ? ctx.totals : r.totals
    shell_n_energy_last = r === nothing ? ctx.shell_n_energy_last : r.shell_n_energy_last
    boundary = s == -2 ? "I" : "H"
    total_modes = r === nothing ? 0 : r.total_modes
    n_reached = nothing
    last_shell = r === nothing ? (0.0, 0.0, 0.0) : r.last_shell
    below_count_n = r === nothing ? 0 : r.below_count_n
    prev_peak_m_pos = r === nothing ? 1 : r.prev_peak_m_pos
    prev_peak_m_neg = r === nothing ? 1 : r.prev_peak_m_neg
    # |m| reached by the previous n shell's walks; an m shell counts toward the stop only past MWALK_REACH_FRACTION of it
    prev_reach_pos = r === nothing ? n0_reach : r.prev_reach_pos
    prev_reach_neg = r === nothing ? n0_reach : r.prev_reach_neg
    # last l of each m walk of the previous shell (the l the walks computed ahead speculate up to)
    prev_l_reach = r === nothing ? Dict{Int, Int}() : r.prev_l_reach
    n_first = r === nothing ? 1 : r.n_next
    l_reach = Dict{Int, Int}()
    params_for = nothing
    # the walk of (m, n), computed ahead or now; a walk beyond the plan starts the next two of its direction
    take = m -> begin
        key = (s, m, n_current[], 0)
        walk = if prefetch === nothing
            run(key, params_for(key))
        else
            step = m >= 0 ? 1 : -1
            prefetch(key; ahead = [(s, mm, n_current[], 0) for mm in (m + step):step:(m + 2step) if abs(mm) <= lmax])
        end
        p = walk.prior
        if !isempty(walk.modes)
            l_reach[m] = walk.modes[end][1]
        elseif p !== nothing
            l_reach[m] = Int(p.last_l)
        end
        p === nothing || ctx.prior_counts!(s, p)
        ctx.ledger_out === nothing || (ctx.ledger_out[key] = _ledger_entry(p, walk.modes, walk.state, walk.tail))
        return walk
    end
    n_current = Ref(0)
    n_next = n_first
    for n in n_first:nmax
        n_current[] = n
        n_next = n + 1
        m_omega_zero = -n * radial_over_azimuthal
        omega_zero_gate = !high_e || abs(m_omega_zero) < lmax
        E_estimate = _fit_shell_estimate(model, n_list, energy_list, n, p0, abs(shell_n_energy_last))
        mode_floor, levin_floor, levin_abs = ctx.shell_parameters(s)
        branch_floor = floor_ref[]
        base_threshold = tol * abs(shell_n_energy_last)
        layer_threshold = _mode_cutoff_threshold(base_threshold, branch_floor)
        shell_q = (max_flux = 2 * E_estimate, floor = mode_floor, levin_floor = levin_floor, levin_abs = levin_abs, branch_floor = branch_floor)
        # the Levin choice of (s, m, n) carries the serial bookkeeping: asked in the serial m order (m = 0, 1, .., -1, ..)
        params_for = key -> (; shell_q..., use_levin = ctx.use_levin(s, key[2], n), threshold = key[2] == 0 ? nothing : base_threshold,
            l_pred = get(prev_l_reach, key[2], nothing))
        if prefetch !== nothing
            prefetch.params_for = params_for
            _walk_plan!(prefetch, [(s, 0, n, 0); [(s, m, n, 0) for m in 1:min(lmax, prev_reach_pos + 1)];
                [(s, -m, n, 0) for m in 1:min(lmax, prev_reach_neg + 1)]])
        end
        shell_n_energy = 0.0
        shell_n_angular = 0.0
        shell_n_carter = 0.0
        below_count_m = 0

        walk = take(0)
        shell_m_energy, shell_m_angular, shell_m_carter, prior_modes = _prior_sums(walk.prior)
        total_modes += prior_modes
        for (l, mode) in walk.modes
            mode = ctx.accept!(s, l, 0, n, mode)
            ctx.record!(boundary, n, 0, l, mode)
            shell_m_energy += 2 * mode["EnergyFlux"]
            shell_m_angular += 2 * mode["AngularMomentumFlux"]
            shell_m_carter += 2 * mode["CarterConstantFlux"]
            total_modes += 1
        end

        shell_n_energy += shell_m_energy
        shell_n_angular += shell_m_angular
        shell_n_carter += shell_m_carter

        Max_flux = 0.0
        Max_m = 1
        reach_pos = 0
        reach_neg = 0
        last_big_pos = 0
        last_big_neg = 0
        for m in 1:lmax
            reach_pos = m
            walk = take(m)
            shell_m_energy, shell_m_angular, shell_m_carter, prior_modes = _prior_sums(walk.prior)
            total_modes += prior_modes
            if walk.prior !== nothing && walk.prior.max_abs > Max_flux
                Max_flux = walk.prior.max_abs
                Max_m = m
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, n, mode)
                ctx.record!(boundary, n, m, l, mode)
                shell_m_energy += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
                if abs(mode["EnergyFlux"]) > Max_flux
                    Max_flux = abs(mode["EnergyFlux"])
                    Max_m = m
                end
            end

            shell_n_energy += shell_m_energy
            shell_n_angular += shell_m_angular
            shell_n_carter += shell_m_carter

            abs(shell_m_energy) > layer_threshold && (last_big_pos = abs(m))
            # Near the ISSO Ups_r << Ups_phi and the n shells of even a nearly circular orbit peak at large m (at
            # (a, p, e) = (0.932, 2.095, 0.009) the n = 6 shell sits at m = l = 15..27 while m <= 2 are tiny, and the walk
            # dropped 4e-6 of E_inf there); a shell counts toward the stop only past the peak m of the previous n shell.
            below_count_m = !(omega_zero_gate && _before_omega_zero(m, m_omega_zero)) && (abs(shell_m_energy) <= layer_threshold) && (abs(shell_n_energy) > min(0.1, exp(1 - 1 / e)) * abs(shell_n_energy_last)) &&
                (m >= max(prev_peak_m_pos, floor(Int, MWALK_REACH_FRACTION * prev_reach_pos))) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end

        below_count_m = 0
        Max_flux_neg = 0.0
        Max_m_neg = 1
        for m in -1:-1:-lmax
            reach_neg = -m
            walk = take(m)
            shell_m_energy, shell_m_angular, shell_m_carter, prior_modes = _prior_sums(walk.prior)
            total_modes += prior_modes
            if walk.prior !== nothing && walk.prior.max_abs > Max_flux_neg
                Max_flux_neg = walk.prior.max_abs
                Max_m_neg = abs(m)
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, n, mode)
                ctx.record!(boundary, n, m, l, mode)
                shell_m_energy += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
                if abs(mode["EnergyFlux"]) > Max_flux_neg
                    Max_flux_neg = abs(mode["EnergyFlux"])
                    Max_m_neg = abs(m)
                end
            end

            shell_n_energy += shell_m_energy
            shell_n_angular += shell_m_angular
            shell_n_carter += shell_m_carter

            abs(shell_m_energy) > layer_threshold && (last_big_neg = abs(m))
            # As in the positive branch, the streak only counts once the n shell holds a fair part of the previous one:
            # for retrograde orbits (x = -1) the negative m carry the flux, and at high n it sits at |m| ~ 5..10, past
            # m = -1, -2, which are tiny (at (a, p, e) = (0.525, 9.52, 0.531) the walk stopped there for n = 22, 25,
            # 27.., dropping 7.6e-4 of E_inf).
            below_count_m = !(omega_zero_gate && _before_omega_zero(m, m_omega_zero)) && (abs(shell_m_energy) <= layer_threshold) && (abs(m) >= max(Max_m, prev_peak_m_neg, floor(Int, MWALK_REACH_FRACTION * prev_reach_neg))) &&
                (abs(shell_n_energy) > min(0.1, exp(1 - 1 / e)) * abs(shell_n_energy_last)) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end
        prefetch === nothing || _walk_discard!(prefetch)
        prev_l_reach = l_reach
        l_reach = Dict{Int, Int}()
        shell_n_energy_last = shell_n_energy
        prev_peak_m_pos = Max_m
        prev_peak_m_neg = Max_m_neg
        prev_reach_pos = high_e ? min(lmax, last_big_pos + minimum_consecutive) : reach_pos
        prev_reach_neg = high_e ? min(lmax, last_big_neg + minimum_consecutive) : reach_neg

        push!(energy_list, shell_n_energy)
        push!(n_list, n)
        push!(angular_list, shell_n_angular); push!(carter_list, shell_n_carter)
        ctx.log_shell(s == -2 ? "infinity" : "horizon", n, shell_n_energy, shell_n_angular, shell_n_carter, total_modes)
        last_shell = (shell_n_energy, shell_n_angular, shell_n_carter)
        total_energy += shell_n_energy
        total_angular += shell_n_angular
        total_carter += shell_n_carter
        floor_ref[] = _adaptive_branch_mode_floor(total_energy, tol, mode_abs_floor)
        levin_abs_ref[] = LEVIN_ABS_FRACTION * tol * abs(total_energy)
        n_threshold = _mode_cutoff_threshold(tol * abs(total_energy), floor_ref[])
        below_count_n = abs(shell_n_energy) <= n_threshold ? below_count_n + 1 : 0
        if below_count_n >= n_stop_run
            n_reached = n
            break
        end
    end
    state = (totals = (total_energy, total_angular, total_carter), n_reached = n_reached, last_shell = last_shell,
        total_modes = total_modes, below_count_n = below_count_n, prev_peak_m_pos = prev_peak_m_pos,
        prev_peak_m_neg = prev_peak_m_neg, prev_reach_pos = prev_reach_pos, prev_reach_neg = prev_reach_neg,
        prev_l_reach = prev_l_reach, shell_n_energy_last = shell_n_energy_last, n_next = n_next)
    return (total_energy, total_angular, total_carter, n_reached, last_shell, total_modes, state)
end

# The two boundaries' shell walks on two tasks (mode_parallelism > 1 without a record), else one after the other;
# returns (infinity result, horizon result). An error of either is raised as such once both are finished.
function _run_boundaries(f_inf, f_hor, concurrent::Bool)
    concurrent || return (f_inf(), f_hor())
    inf_task = Threads.@spawn f_inf()
    hor = try
        f_hor()
    catch caught
        absorbable(caught) || rethrow()
        try; wait(inf_task); catch caught; absorbable(caught) || rethrow(); end   # the other boundary's walks cannot be cancelled: let it finish
        rethrow()
    end
    inf = try
        fetch(inf_task)
    catch error
        error isa TaskFailedException ? throw(error.task.result) : rethrow()
    end
    return (inf, hor)
end

function _eccentric_mode_summation_scoped(scope, a, p, e, o::FluxOptions; x = 1.0, lmax = 30, nmax = 500, resume = nothing)
    (; tol, N0, Nmax, sample_tol, record, mode_abs_floor, threaded_sampling, tail_levin, tail_levin_infinity,
        tail_levin_horizon, levin_nmin, levin_local_n, levin_max_depth, mode_parallelism, progress) = o
    record_path = o.record_path
    minimum_consecutive = something(o.minimum_consecutive, 2)
    levin_mode_abs_floor = something(o.levin_mode_abs_floor, mode_abs_floor)
    rules = orbit_rules(a, p, e, x)
    if e == 0.0
        return circular_mode_summation(a, p; x, submission = scope, options = _with(o; lmax, minimum_consecutive))
    end
    N = N0
    record_path = record_path === nothing ? "eccentric_mode_data_a_$(_record_tag(a))_p_$(_record_tag(p))_e_$(_record_tag(e))$(x == 1 ? "" : "_x_$(_record_tag(x))").h5" : record_path

    KG = submission_orbit!(scope)
    radial_over_azimuthal = KG["Frequencies"]["ϒr"] / KG["Frequencies"]["ϒϕ"]
    # The n shells repeat about every Ups_phi / Ups_r shells (lowering m by one and raising n by Ups_phi / Ups_r gives the
    # same omega). Near the separatrix (zoom-whirl) that is ~ 30 shells, with deep dips between the peaks: at
    # (a, p, e, x) = (0.254, 8.61, 0.821, -1) the walk stopped in the dip at n = 250 and missed the next peaks (1e-6 of
    # E_inf at tol = 1e-8). The n stop needs a run of below-threshold shells spanning one such period.
    n_stop_run = _n_stop_run(minimum_consecutive, radial_over_azimuthal)
    F_orbit = KG["Frequencies"]
    KG_sample = submission_master!(scope)
    # resume (see _with_auto_caps): the state of a pass with the same lmax and a smaller nmax that stopped at its nmax;
    # this pass continues its n loops from there (not with a record, whose file is written anew)
    prev = resume === nothing ? nothing : resume.prev
    track = resume !== nothing && resume.track && !record
    rs = prev isa NamedTuple && get(prev, :kind, nothing) === :eccentric && prev.lmax == lmax && !record ? prev : nothing
    # replay (see _with_auto_caps): a pass with a smaller lmax; its l walks come from its walk ledgers
    rl = rs === nothing && prev isa NamedTuple && get(prev, :kind, nothing) === :eccentric && prev.lmax < lmax &&
        prev.ledgers !== nothing && !record ? prev : nothing
    ledger_in = rl === nothing ? (nothing, nothing) : rl.ledgers
    lmax_prior = rl === nothing ? lmax : rl.lmax
    ledger_out = !track ? (nothing, nothing) : rs !== nothing ? something(rs.ledgers, (nothing, nothing)) : (_WalkLedger(), _WalkLedger())
    reused_modes = [0, 0]
    Energy_flux_inf = rs === nothing ? Vector{Float64}() : rs.Energy_flux_inf
    Energy_flux_hor = rs === nothing ? Vector{Float64}() : rs.Energy_flux_hor
    # angular-momentum and Carter-constant shells, alongside the energy shells (the n of each shell in n_list_*)
    Angular_flux_inf = rs === nothing ? Vector{Float64}() : rs.Angular_flux_inf
    Angular_flux_hor = rs === nothing ? Vector{Float64}() : rs.Angular_flux_hor
    Carter_flux_inf = rs === nothing ? Vector{Float64}() : rs.Carter_flux_inf
    Carter_flux_hor = rs === nothing ? Vector{Float64}() : rs.Carter_flux_hor
    n_list_inf = rs === nothing ? Vector{Int64}() : rs.n_list_inf
    n_list_hor = rs === nothing ? Vector{Int64}() : rs.n_list_hor
    model(x, p) = p[1] .* (x .+ 1) .^p[2] .* exp.(-p[3] .* x)
    p0 = [1e-3, 0.0, 0.1]

    total_infinity_energy = 0.0
    total_infinity_angular = 0.0
    total_infinity_carter = 0.0
    total_horizon_energy = 0.0
    total_horizon_angular = 0.0
    total_horizon_carter = 0.0
    total_modes = 0
    n_reached_inf = nothing
    n_reached_hor = nothing
    last_shell_inf = (0.0, 0.0, 0.0)
    last_shell_hor = (0.0, 0.0, 0.0)

    shell_n_infinity_energy = 0.0
    shell_n_infinity_energy_last = 0.0
    shell_n_infinity_angular = 0.0
    shell_n_infinity_carter = 0.0
    shell_n_horizon_energy = 0.0
    shell_n_horizon_energy_last = 0.0
    shell_n_horizon_angular = 0.0
    shell_n_horizon_carter = 0.0
    mode_index = Ref(0)
    # progress = true: one line per finished n shell on stdout (the number of shells is not known in advance)
    t_start = rs === nothing ? time() : rs.t_start
    shells_done = rs === nothing ? Ref(0) : rs.shells_done
    log_shell = (branch, n, E, L, Q, modes) -> begin
        progress || return nothing
        lock(_PROGRESS_LOCK) do
            shells_done[] += 1
            @printf("GSN shell (a, p, e, x) = (%s, %s, %s, %s) lmax=%d nmax=%d branch=%s n=%d E=%.6e L=%.6e Q=%.6e modes=%d shells_done=%d elapsed=%.1fs\n",
                a, p, e, x, lmax, nmax, branch, n, E, L, Q, modes, shells_done[], time() - t_start)
            flush(stdout)
        end
        return nothing
    end
    adaptive_local_n = levin_local_n
    tail_policy = _tail_policy(tail_levin)
    tail_policy_inf = _tail_policy(tail_levin_infinity === nothing ? tail_policy : tail_levin_infinity)
    tail_policy_hor = _tail_policy(tail_levin_horizon === nothing ? tail_policy : tail_levin_horizon)
    branch_mode_floor_inf = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_inf
    branch_mode_floor_hor = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_hor
    # Absolute energy tolerance of an adaptive Levin mode: LEVIN_ABS_FRACTION tol times the branch total so far.
    branch_levin_abs_inf = rs === nothing ? Ref(0.0) : rs.branch_levin_abs_inf
    branch_levin_abs_hor = rs === nothing ? Ref(0.0) : rs.branch_levin_abs_hor
    levin_floor_tracks_mode_floor = levin_mode_abs_floor == mode_abs_floor
    branch_mode_floor = s -> s == -2 ? branch_mode_floor_inf[] : branch_mode_floor_hor[]
    branch_levin_floor = s -> levin_floor_tracks_mode_floor ? branch_mode_floor(s) : Float64(levin_mode_abs_floor)
    tail_levin_infinity_active = rs === nothing ? Ref(false) : rs.tail_levin_infinity_active
    tail_levin_horizon_active = rs === nothing ? Ref(false) : rs.tail_levin_horizon_active
    tail_levin_infinity_start_n = rs === nothing ? Ref{Union{Nothing, Int}}(nothing) : rs.tail_levin_infinity_start_n
    tail_levin_horizon_start_n = rs === nothing ? Ref{Union{Nothing, Int}}(nothing) : rs.tail_levin_horizon_start_n
    # tail_levin = true: every mode from the first |n| >= nmin on. :auto: each mode whose trapezoid grid,
    # estimated from its radial phase rate, exceeds Nmax (_levin_auto).
    use_eccentric_tail_levin = (s, m, n) -> begin
        branch_policy = s == -2 ? tail_policy_inf : tail_policy_hor
        _tail_enabled(branch_policy) || return false
        active = s == -2 ? tail_levin_infinity_active : tail_levin_horizon_active
        start_n = s == -2 ? tail_levin_infinity_start_n : tail_levin_horizon_start_n
        branch_policy === true && active[] && return true
        _use_levin_tail(n; nmin = levin_nmin) || return false
        if branch_policy === true || _levin_auto(_mode_radial_trapezoid_n(KG_sample, m, n, 0, N, Nmax), Nmax)
            active[] = true
            start_n[] === nothing && (start_n[] = n)
            return true
        end
        return false
    end
    cap_trackers = rs === nothing ? (CapTracker(lmax, 0), CapTracker(lmax, 0)) : rs.cap_trackers
    failed_modes = rs === nothing ? [0, 0] : rs.failed_modes
    near_static_zero = rs === nothing ? [0, 0] : rs.near_static_zero
    # One mode from the shell's parameters (no summation state is read or written here, so modes of one shell can be
    # computed concurrently) and `factors`, the phase factorizations shared by its l walk (see _eccentric_l_walk), and
    # its acceptance (the counters), which the summation applies in its own order.
    ecc_compute = (s, l, m, n, max_flux, use_levin, floor, levin_floor, levin_abs, factors, arena) -> begin
        ω_mode = (m * F_orbit["ϒϕ"] + n * F_orbit["ϒr"]) / F_orbit["ϒt"]
        mode = if _negligible_low_frequency_mode(s, l, ω_mode, F_orbit["ϒϕ"] / F_orbit["ϒt"], a, p, e)
            res = ConvolutionIntegrals._zero_radiative_mode(ω_mode, nothing; reason = "negligible_low_frequency")
            res["NearStaticOverflow"] = true
            res
        else
        let; ConvolutionIntegrals._with_coverage_retry(s, l, m, n, 0; omega = ω_mode, Ωφ = F_orbit["ϒϕ"] / F_orbit["ϒt"], rmax = p / (1 - e), negligible_if_failed = _negligible_low_frequency_mode(s, l, ω_mode, F_orbit["ϒϕ"] / F_orbit["ϒt"], a, p, e; bound = ConvolutionIntegrals.NEGLIGIBLE_FAILED_MODE_BOUND)) do; if use_levin
            with_submission_mode(scope; mode = (s, l, m, n, 0)) do mode_cache
                mode_cache.walk_factors = factors
                ConvolutionIntegrals.convolution_integral_eccentric_adaptive_levin_isem(KG_sample, s, l, m, n; local_n = adaptive_local_n, max_depth = levin_max_depth, tol = tol, mode_abs_floor = _tail_levin_mode_floor(floor, levin_floor), energy_abs_tol = levin_abs, threaded_sampling = threaded_sampling, cache = mode_cache, arena)
            end
        else
            with_submission_mode(scope; mode = (s, l, m, n, 0)) do mode_cache
                ConvolutionIntegrals.convolution_integral_eccentric_trapezoidal_isem(KG_sample, s, l, m, n, N; tol = tol, sample_tol = sample_tol, max_flux = max_flux, Nmax = Nmax, mode_abs_floor = floor, threaded_sampling = threaded_sampling, cache = mode_cache, arena)
            end
        end; end; end
        end
        mode
    end
    ecc_accept! = (s, l, m, n, mode) -> begin
        get(mode, "FailedMode", false) && (failed_modes[s == -2 ? 1 : 2] += 1)
        get(mode, "NearStaticOverflow", false) && (near_static_zero[s == -2 ? 1 : 2] += 1)
        ConvolutionIntegrals._check_finite_mode_flux(mode, s, l, m, n, 0)
        _note_cap_mode!(cap_trackers[s == -2 ? 1 : 2], l, m, n, 0, mode["EnergyFlux"])
        mode
    end
    shell_parameters = s -> (MODE_REFINE_FLOOR_FRACTION * branch_mode_floor(s), branch_levin_floor(s),
        s == -2 ? branch_levin_abs_inf[] : branch_levin_abs_hor[])
    ecc_compute_q = (s, l, m, n, q, factors, arena) -> ecc_compute(s, l, m, n, q.max_flux, q.use_levin, q.floor, q.levin_floor, q.levin_abs, factors, arena)
    pf_inf = nothing   # the walk prefetches of the two boundaries (mode_parallelism > 1), set below, drained in finally
    pf_hor = nothing

    record_h5 = record ? h5open(record_path, "w") : nothing
    try
        if record
            record_h5["meta/a"] = a
            record_h5["meta/p"] = p
            record_h5["meta/e"] = e
            record_h5["meta/x"] = x
            record_h5["meta/N_requested"] = N
            record_h5["meta/Nmax"] = Nmax
            record_h5["meta/tol"] = tol
            record_h5["meta/lmax"] = lmax
            record_h5["meta/nmax"] = nmax
            record_h5["meta/minimum_consecutive"] = minimum_consecutive
            record_h5["meta/sample_tol"] = sample_tol
            record_h5["meta/levin_max_depth"] = levin_max_depth
        end

    n0_reach_inf = 0
    n0_reach_hor = 0
    rs === nothing || (total_modes = rs.n0_modes)
    if rs === nothing
    # The n = 0 shell of each boundary: m = 1, 2, ... l walks, every mode through ecc_compute / ecc_accept! (the
    # negligible-mode skip, coverage retry, counters, finite check and cap tracking of the n >= 1 shells) refined to
    # MODE_REFINE_FLOOR_FRACTION of the running floor; an l walk stops after minimum_consecutive modes below tol times
    # its m shell, the m walk after minimum_consecutive m shells below tol times the n shell. Returns the shell sums,
    # the last m reached and the modes computed.
    n0_shell = s -> begin
        boundary = s == -2 ? "I" : "H"
        energy, angular, carter = 0.0, 0.0, 0.0
        below_m, reach, modes = 0, 0, 0
        arena = ConvolutionIntegrals._series_arena()   # work arena of the closed-form series for this shell's modes
        for m in 1:lmax
            reach = m
            m_energy, m_angular, m_carter = 0.0, 0.0, 0.0
            below_l = 0
            for l in max(2, m):lmax
                floor = _adaptive_branch_mode_floor(energy + m_energy, tol, mode_abs_floor)
                mode = ecc_accept!(s, l, m, 0,
                    ecc_compute(s, l, m, 0, 1.0, false, MODE_REFINE_FLOOR_FRACTION * floor, floor, 0.0, nothing, arena))
                _record_current_mode!(record_h5, record, boundary, mode_index, mode; l, m, n = 0, N_sample = N)
                m_energy += 2 * mode["EnergyFlux"]
                m_angular += 2 * mode["AngularMomentumFlux"]
                m_carter += 2 * mode["CarterConstantFlux"]
                modes += 1
                below_l = abs(mode["EnergyFlux"]) <= _mode_cutoff_threshold(tol * abs(m_energy), floor) ? below_l + 1 : 0
                below_l >= minimum_consecutive && break
            end
            energy += m_energy
            angular += m_angular
            carter += m_carter
            threshold = _mode_cutoff_threshold(tol * abs(energy), _adaptive_branch_mode_floor(energy, tol, mode_abs_floor))
            below_m = abs(m_energy) <= threshold ? below_m + 1 : 0
            below_m >= minimum_consecutive && break
        end
        return energy, angular, carter, reach, modes
    end
    shell_n_infinity_energy, shell_n_infinity_angular, shell_n_infinity_carter, n0_reach_inf, modes_n0_inf = n0_shell(-2)
    shell_n_horizon_energy, shell_n_horizon_angular, shell_n_horizon_carter, n0_reach_hor, modes_n0_hor = n0_shell(2)
    total_modes += modes_n0_inf + modes_n0_hor
    # the reference of the n = 1 shell, whether the m walk stopped by its rule or ran off lmax
    shell_n_infinity_energy_last = shell_n_infinity_energy
    shell_n_horizon_energy_last = shell_n_horizon_energy

    push!(Energy_flux_inf, shell_n_infinity_energy)
    push!(Energy_flux_hor, shell_n_horizon_energy)
    push!(n_list_inf, 0)
    push!(n_list_hor, 0)
    push!(Angular_flux_inf, shell_n_infinity_angular); push!(Carter_flux_inf, shell_n_infinity_carter)
    push!(Angular_flux_hor, shell_n_horizon_angular); push!(Carter_flux_hor, shell_n_horizon_carter)
    log_shell("infinity", 0, shell_n_infinity_energy, shell_n_infinity_angular, shell_n_infinity_carter, total_modes)
    log_shell("horizon", 0, shell_n_horizon_energy, shell_n_horizon_angular, shell_n_horizon_carter, total_modes)
    total_infinity_energy += shell_n_infinity_energy
    total_infinity_angular += shell_n_infinity_angular
    total_infinity_carter += shell_n_infinity_carter
    total_horizon_energy += shell_n_horizon_energy
    total_horizon_angular += shell_n_horizon_angular
    total_horizon_carter += shell_n_horizon_carter
    branch_mode_floor_inf[] = _adaptive_branch_mode_floor(total_infinity_energy, tol, mode_abs_floor)
    branch_mode_floor_hor[] = _adaptive_branch_mode_floor(total_horizon_energy, tol, mode_abs_floor)
    branch_levin_abs_inf[] = LEVIN_ABS_FRACTION * tol * abs(total_infinity_energy)
    branch_levin_abs_hor[] = LEVIN_ABS_FRACTION * tol * abs(total_horizon_energy)
    end
    n0_modes = total_modes

    # n >= 1 shells of both boundaries (see _eccentric_branch_shells!), each from the state after its n = 0 shell;
    # with mode_parallelism > 1 their l walks are computed ahead and, without a record, the boundaries run on two tasks
    walk_run = (key, q; arena = ConvolutionIntegrals._series_arena()) -> _eccentric_l_walk(ecc_compute_q, key[1], key[2], key[3], q, lmax, minimum_consecutive, Float64(tol), mode_parallelism > 1;
        prior = _prior_walk(ledger_in[key[1] == -2 ? 1 : 2], key, lmax, lmax_prior), lmax_prior = lmax_prior, arena)
    prior_counts! = (s, p) -> begin
        i = s == -2 ? 1 : 2
        failed_modes[i] += p.failed
        near_static_zero[i] += p.near_static
        reused_modes[i] += Int(p.count)
    end
    branch_ctx = s -> (lmax = lmax, nmax = nmax, minimum_consecutive = minimum_consecutive, n_stop_run = n_stop_run,
        radial_over_azimuthal = radial_over_azimuthal, high_e = rules.high_e, e = e, tol = Float64(tol), model = model, p0 = p0,
        mode_abs_floor = mode_abs_floor, n0_reach = s == -2 ? n0_reach_inf : n0_reach_hor,
        energy_list = s == -2 ? Energy_flux_inf : Energy_flux_hor, angular_list = s == -2 ? Angular_flux_inf : Angular_flux_hor,
        carter_list = s == -2 ? Carter_flux_inf : Carter_flux_hor, n_list = s == -2 ? n_list_inf : n_list_hor,
        floor_ref = s == -2 ? branch_mode_floor_inf : branch_mode_floor_hor,
        levin_abs_ref = s == -2 ? branch_levin_abs_inf : branch_levin_abs_hor,
        totals = s == -2 ? (total_infinity_energy, total_infinity_angular, total_infinity_carter) :
            (total_horizon_energy, total_horizon_angular, total_horizon_carter),
        shell_n_energy_last = s == -2 ? shell_n_infinity_energy_last : shell_n_horizon_energy_last,
        shell_parameters = shell_parameters, use_levin = use_eccentric_tail_levin,
        # the serial walks of one boundary share one work arena (walks computed ahead make their own)
        run = let arena = ConvolutionIntegrals._series_arena(); (key, q) -> walk_run(key, q; arena) end, accept! = ecc_accept!,
        record! = (boundary, n, m, l, mode) -> _record_current_mode!(record_h5, record, boundary, mode_index, mode; l, m, n, N_sample = N),
        log_shell = log_shell, resume = rs === nothing ? nothing : s == -2 ? rs.infinity : rs.horizon,
        ledger_out = ledger_out[s == -2 ? 1 : 2], prior_counts! = prior_counts!)
    pf_inf = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_inf) : nothing
    pf_hor = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_hor) : nothing
    inf, hor = _run_boundaries(() -> _eccentric_branch_shells!(-2, branch_ctx(-2), pf_inf),
        () -> _eccentric_branch_shells!(2, branch_ctx(2), pf_hor), mode_parallelism > 1 && !record)
    total_infinity_energy, total_infinity_angular, total_infinity_carter, n_reached_inf, last_shell_inf, modes_inf, state_inf = inf
    total_horizon_energy, total_horizon_angular, total_horizon_carter, n_reached_hor, last_shell_hor, modes_hor, state_hor = hor
    total_modes += modes_inf + modes_hor

    tail_factor_inf, tail_ratio_inf = _geometric_n_tail_factor(Energy_flux_inf, e, n_reached_inf, n_stop_run)
    tail_factor_hor, tail_ratio_hor = _geometric_n_tail_factor(Energy_flux_hor, e, n_reached_hor, n_stop_run)
    n_tail_inf = tail_factor_inf .* last_shell_inf
    n_tail_hor = tail_factor_hor .* last_shell_hor
    total_infinity_energy += n_tail_inf[1]
    total_infinity_angular += n_tail_inf[2]
    total_infinity_carter += n_tail_inf[3]
    total_horizon_energy += n_tail_hor[1]
    total_horizon_angular += n_tail_hor[2]
    total_horizon_carter += n_tail_hor[3]

    if record
        record_h5["meta/infinity_energy_flux"] = total_infinity_energy
        record_h5["meta/infinity_angular_momentum_flux"] = total_infinity_angular
        record_h5["meta/infinity_carter_constant_flux"] = total_infinity_carter
        record_h5["meta/horizon_energy_flux"] = total_horizon_energy
        record_h5["meta/horizon_angular_momentum_flux"] = total_horizon_angular
        record_h5["meta/horizon_carter_constant_flux"] = total_horizon_carter
        record_h5["meta/n_reached_inf"] = something(n_reached_inf, -1)
        record_h5["meta/n_reached_hor"] = something(n_reached_hor, -1)
        record_h5["meta/tail_levin_infinity_enabled"] = _tail_enabled(tail_policy_inf)
        record_h5["meta/tail_levin_horizon_enabled"] = _tail_enabled(tail_policy_hor)
        record_h5["meta/tail_levin_infinity_policy"] = string(tail_policy_inf)
        record_h5["meta/tail_levin_horizon_policy"] = string(tail_policy_hor)
        record_h5["meta/tail_levin_start_inf"] = something(tail_levin_infinity_start_n[], -1)
        record_h5["meta/tail_levin_start_hor"] = something(tail_levin_horizon_start_n[], -1)
        record_h5["meta/tail_levin_nmin"] = levin_nmin
        record_h5["meta/tail_levin_local_n"] = levin_local_n
        record_h5["meta/infinity_truncation_floor"] = branch_mode_floor_inf[]
        record_h5["meta/horizon_truncation_floor"] = branch_mode_floor_hor[]
        record_h5["meta/total_modes"] = total_modes
        record_h5["meta/n_tail_infinity"] = collect(n_tail_inf)
        record_h5["meta/n_tail_horizon"] = collect(n_tail_hor)
        record_h5["meta/n_tail_ratio_infinity"] = tail_ratio_inf
        record_h5["meta/n_tail_ratio_horizon"] = tail_ratio_hor
        record_h5["meta/infinity_energy_flux_list"] = Energy_flux_inf
        record_h5["meta/horizon_energy_flux_list"] = Energy_flux_hor
    end

    return (
        infinity_energy_flux = total_infinity_energy,
        infinity_angular_momentum_flux = total_infinity_angular,
        infinity_carter_constant_flux = total_infinity_carter,
        horizon_energy_flux = total_horizon_energy,
        horizon_angular_momentum_flux = total_horizon_angular,
        horizon_carter_constant_flux = total_horizon_carter,
        total_modes = total_modes,
        tolerance = tol,
        minimum_consecutive = minimum_consecutive,
        l_max = lmax,
        n_max = nmax,
        N_sample = N,
        N_max = Nmax,
        n_reached_inf = n_reached_inf,
        n_reached_hor = n_reached_hor,
        mode_parallelism = mode_parallelism,
        # with mode_parallelism > 1: modes computed in all, and those computed ahead but never asked for
        modes_computed = pf_inf === nothing ? nothing : pf_inf.modes + pf_hor.modes,
        modes_computed_unused = pf_inf === nothing ? nothing : (pf_inf.modes - pf_inf.modes_used) + (pf_hor.modes - pf_hor.modes_used),
        lmax_hit_inf = _cap_hits(cap_trackers[1], tol, total_infinity_energy)[1],
        lmax_hit_hor = _cap_hits(cap_trackers[2], tol, total_horizon_energy)[1],
        nmax_hit_inf = _nmax_hit(n_reached_inf, Energy_flux_inf, nmax, tol, total_infinity_energy),
        nmax_hit_hor = _nmax_hit(n_reached_hor, Energy_flux_hor, nmax, tol, total_horizon_energy),
        failed_modes_inf = failed_modes[1],
        failed_modes_hor = failed_modes[2],
        near_static_zero_inf = near_static_zero[1],
        near_static_zero_hor = near_static_zero[2],
        tail_levin_infinity_enabled = _tail_enabled(tail_policy_inf),
        tail_levin_horizon_enabled = _tail_enabled(tail_policy_hor),
        tail_levin_infinity_policy = tail_policy_inf,
        tail_levin_horizon_policy = tail_policy_hor,
        tail_levin_start_inf = tail_levin_infinity_start_n[],
        tail_levin_start_hor = tail_levin_horizon_start_n[],
        tail_levin_nmin = levin_nmin,
        tail_levin_local_n = levin_local_n,
        tail_levin_max_depth = levin_max_depth,
        infinity_truncation_floor = branch_mode_floor_inf[],
        horizon_truncation_floor = branch_mode_floor_hor[],
        n_tail_infinity = (energy = n_tail_inf[1], angular_momentum = n_tail_inf[2], carter_constant = n_tail_inf[3], ratio = tail_ratio_inf),
        n_tail_horizon = (energy = n_tail_hor[1], angular_momentum = n_tail_hor[2], carter_constant = n_tail_hor[3], ratio = tail_ratio_hor),
        infinity_energy_flux_list = Energy_flux_inf,
        horizon_energy_flux_list = Energy_flux_hor,
        # every n shell of the run (symmetry weights included; the totals are their sums plus n_tail_*)
        infinity_shells = (n = n_list_inf, energy = Energy_flux_inf, angular_momentum = Angular_flux_inf, carter_constant = Carter_flux_inf),
        horizon_shells = (n = n_list_hor, energy = Energy_flux_hor, angular_momentum = Angular_flux_hor, carter_constant = Carter_flux_hor),
        record_file = record ? record_path : nothing,
        modes_new = total_modes - (rs !== nothing ? rs.total_modes : reused_modes[1] + reused_modes[2]),
        _resume = record ? nothing : (; kind = :eccentric, lmax = lmax, infinity = state_inf, horizon = state_hor, n0_modes = n0_modes,
            total_modes, ledgers = track ? ledger_out : nothing,
            Energy_flux_inf, Energy_flux_hor, Angular_flux_inf, Angular_flux_hor, Carter_flux_inf, Carter_flux_hor,
            n_list_inf, n_list_hor, branch_mode_floor_inf, branch_mode_floor_hor, branch_levin_abs_inf, branch_levin_abs_hor,
            tail_levin_infinity_active, tail_levin_horizon_active, tail_levin_infinity_start_n, tail_levin_horizon_start_n,
            cap_trackers, failed_modes, near_static_zero, t_start, shells_done, pf_inf, pf_hor),
    )
    finally
        pf_inf === nothing || _walk_discard!(pf_inf)
        pf_hor === nothing || _walk_discard!(pf_hor)
        record && close(record_h5)
    end
end

# Total eccentric equatorial fluxes: n = 0 shell, then n = 1, 2, ... shells (each over m >= 1, m <= -1 and l),
# until minimum_consecutive + 1 n shells are below tol times the branch total. See Teukolsky_pointparticle_flux.
function eccentric_mode_summation(a, p, e; submission = nothing, x = 1.0, stop_requested = nothing, options = nothing, kwargs...)
    o = _sum_options(options, submission, kwargs)
    caps = _auto_caps_start(a, p, e, x, o.tol, o.lmax, nothing, o.nmax)
    scope, owned = ensure_public_submission(submission, a, p, e, x; Nmax = o.Nmax, Kmax = o.Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        _with_auto_caps(caps) do L, Kc, Nc, resume
            with_submission_operation(scope) do
                _eccentric_mode_summation_scoped(scope, a, p, e, o; x, lmax = L, nmax = Nc, resume)
            end
        end
    end
end



function _inclined_mode_flux_scoped(scope, a, p, x, l, m, k; K = 16, Kmax = 2^12, sample_tol = 1e-3)
    if x == 1.0 || x == -1.0
        if k != 0
            zero_flux = (energy_flux = 0.0, angular_momentum_flux = 0.0, carter_constant_flux = 0.0)
            return (l = l, m = m, k = k, infinity = zero_flux, horizon = zero_flux)
        end
        mode = circular_mode_flux(a, p, l, m; x = x, submission = scope)
        return (l = l, m = m, k = k, infinity = mode.infinity, horizon = mode.horizon)
    end

    KG = submission_orbit!(scope)
    KG_sample = submission_master!(scope)
    flux_inf = let; with_submission_mode(scope; mode = (-2, l, m, 0, k)) do mode_cache
        ConvolutionIntegrals.convolution_integral_inclined_trapezoidal_isem(KG_sample, -2, l, m, k, K; Kmax = Kmax, sample_tol = sample_tol, cache = mode_cache)
    end; end
    flux_hor = let; with_submission_mode(scope; mode = (2, l, m, 0, k)) do mode_cache
        ConvolutionIntegrals.convolution_integral_inclined_trapezoidal_isem(KG_sample, 2, l, m, k, K; Kmax = Kmax, sample_tol = sample_tol, cache = mode_cache)
    end; end
    return (
        l = l,
        m = m,
        k = k,
        infinity = (
            energy_flux = flux_inf["EnergyFlux"],
            angular_momentum_flux = flux_inf["AngularMomentumFlux"],
            carter_constant_flux = flux_inf["CarterConstantFlux"],
        ),
        horizon = (
            energy_flux = flux_hor["EnergyFlux"],
            angular_momentum_flux = flux_hor["AngularMomentumFlux"],
            carter_constant_flux = flux_hor["CarterConstantFlux"],
        ),
    )
end

# Infinity and horizon fluxes of the inclined circular mode (l, m, k), in a submission scope.
function inclined_mode_flux(a, p, x, l, m, k; submission = nothing, K = 16, Kmax = submission === nothing ? 2^12 : submission.Kmax, sample_tol = 1e-3, stop_requested = nothing, Nmax = submission === nothing ? 2^14 : submission.Nmax)
    scope, owned = ensure_public_submission(submission, a, p, 0.0, x; Nmax, Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        with_submission_operation(scope) do
            _inclined_mode_flux_scoped(scope, a, p, x, l, m, k; K = K, Kmax = Kmax, sample_tol = sample_tol)
        end
    end
end

# One l walk of the inclined sum, (s, m, k) with k >= 1, up from the peak l (infinity) or from lmin (horizon) as in
# _branch_l_walk, from the shell's parameters q only (max_flux and floor of the modes, threshold of the cutoff): pure.
# Returns the (l, mode) in walk order and the extrapolated l tail beyond lmax; `strip` drops the large members of the
# mode Dicts.
function _inclined_l_walk(compute, s::Int, m::Int, k::Int, q, lmax::Int, minimum_consecutive::Int, x, strip::Bool;
        prior = nothing, lmax_prior::Int = 0, arena = ConvolutionIntegrals._series_arena())
    modes = Tuple{Int, Any}[]
    lmin = max(2, abs(m))
    eval_l = l -> begin
        mode = compute(s, l, m, k, q, arena)
        strip && _strip_mode!(mode)
        push!(modes, (l, mode))
        mode["EnergyFlux"]
    end
    walk = if prior === nothing
        _branch_l_walk(eval_l, lmin, _branch_peak_l(lmin, m, k, x), lmax, q.threshold, minimum_consecutive, s == 2)
    elseif _continues(prior, lmax, lmax_prior)
        _continue_branch_l_walk(eval_l, prior, lmax_prior, lmax, q.threshold, minimum_consecutive)
    else
        (tail = prior.tail, state = _entry_state(prior))
    end
    return (modes = modes, tail = walk.tail, prior = prior, state = walk.state)
end

# The k >= 1 shells of one boundary of the inclined sum (s = -2: infinity, 2: horizon), after its k = 0 shell: the
# m = 0, m = 1, 2, .. and m = -1, -2, .. l walks of each shell, the m and k stop rules, the shell lists and the totals.
# The l walks depend on the shell's parameters only: with `prefetch` they are computed ahead, planned from the m reach
# of the previous shell. The two boundaries may run on different tasks: everything mutable here is local or owned by
# this boundary (ctx). Returns (total energy, angular momentum, Carter constant, k_reached, l_reached, lmax_hit,
# near-static zeros, failed modes, modes, state); state is what the next shell needs, from which a pass with a larger
# kmax continues (ctx.resume).
function _inclined_branch_shells!(s::Int, ctx, prefetch)
    (; lmax, kmax, minimum_consecutive, tol, x, model, p0, mode_abs_floor, energy_list, angular_list, carter_list,
        k_list, floor_ref, run) = ctx
    r = ctx.resume
    r === nothing || r.k_reached === nothing ||
        return (r.totals..., r.k_reached, r.l_reached, r.lmax_hit, r.near_static_zero, r.failed_modes, r.total_modes, r)
    total_energy, total_angular, total_carter = r === nothing ? ctx.totals : r.totals
    shell_k_energy_last = r === nothing ? ctx.shell_k_energy_last : r.shell_k_energy_last
    l_reached = r === nothing ? ctx.l_reached : r.l_reached
    lmax_hit = r === nothing ? ctx.lmax_hit : r.lmax_hit
    boundary = s == -2 ? "I" : "H"
    near_static_zero = r === nothing ? 0 : r.near_static_zero
    failed_modes = r === nothing ? 0 : r.failed_modes
    total_modes = r === nothing ? 0 : r.total_modes
    reused_modes = 0   # modes taken from the walk ledger of an earlier pass
    k_reached = nothing
    below_count_n = r === nothing ? 0 : r.below_count_n
    prev_reach_pos = r === nothing ? ctx.k0_reach : r.prev_reach_pos
    prev_reach_neg = r === nothing ? ctx.k0_reach : r.prev_reach_neg
    k_first = r === nothing ? 1 : r.k_next
    k_next = k_first
    params_for = nothing
    k_current = Ref(0)
    # the walk of (m, k), computed ahead or now; a walk beyond the plan starts the next two of its direction
    take = m -> begin
        key = (s, m, 0, k_current[])
        walk = if prefetch === nothing
            run(key, params_for(key))
        else
            step = m >= 0 ? 1 : -1
            prefetch(key; ahead = [(s, mm, 0, k_current[]) for mm in (m + step):step:(m + 2step) if abs(mm) <= lmax])
        end
        ctx.ledger_out === nothing || (ctx.ledger_out[key] = _ledger_entry(walk.prior, walk.modes, walk.state, walk.tail))
        return walk
    end
    # the replay of one walk: the serial bookkeeping of its modes, in order (those of an earlier pass from its ledger entry)
    replay! = (walk, m, k, shell_k_energy, track_max) -> begin
        p = walk.prior
        shell_m_energy, shell_m_angular, shell_m_carter, prior_modes = _prior_sums(p)
        Max_flux = 0.0
        if p !== nothing
            near_static_zero += p.near_static
            failed_modes += p.failed
            reused_modes += prior_modes
            total_modes += prior_modes
            p.count > 0 && (l_reached = max(l_reached, Int(p.last_l)))
            track_max && p.max_abs > Max_flux && (Max_flux = p.max_abs)
        end
        for (l, mode) in walk.modes
            ctx.record!(boundary, k, m, l, mode)
            get(mode, "NearStaticOverflow", false) && (near_static_zero += 1)
            get(mode, "FailedMode", false) && (failed_modes += 1)
            l_reached = max(l_reached, l)
            shell_m_energy += 2 * mode["EnergyFlux"]
            shell_m_angular += 2 * mode["AngularMomentumFlux"]
            shell_m_carter += 2 * mode["CarterConstantFlux"]
            total_modes += 1
            track_max && abs(mode["EnergyFlux"]) > Max_flux && (Max_flux = abs(mode["EnergyFlux"]))
        end
        lmax_hit = lmax_hit || _tail_hit(walk.tail, tol, total_energy + shell_k_energy + shell_m_energy)
        return (shell_m_energy, shell_m_angular, shell_m_carter, Max_flux)
    end
    for k in k_first:kmax
        k_current[] = k
        k_next = k + 1
        E_estimate = _fit_shell_estimate(model, k_list, energy_list, k, p0, abs(shell_k_energy_last))
        layer_threshold = _mode_cutoff_threshold(tol * abs(shell_k_energy_last), floor_ref[])
        shell_q = (max_flux = 2 * E_estimate, floor = MODE_REFINE_FLOOR_FRACTION * floor_ref[], threshold = layer_threshold)
        params_for = key -> shell_q
        if prefetch !== nothing
            prefetch.params_for = params_for
            _walk_plan!(prefetch, [(s, 0, 0, k); [(s, m, 0, k) for m in 1:min(lmax, prev_reach_pos + 1)];
                [(s, -m, 0, k) for m in 1:min(lmax, prev_reach_neg + 1)]])
        end
        shell_k_energy = 0.0
        shell_k_angular = 0.0
        shell_k_carter = 0.0
        below_count_m = 0

        shell_m_energy, shell_m_angular, shell_m_carter, _ = replay!(take(0), 0, k, shell_k_energy, false)

        shell_k_energy += shell_m_energy
        shell_k_angular += shell_m_angular
        shell_k_carter += shell_m_carter

        Max_flux = 0.0
        Max_m = 1
        m_trail = Float64[]
        # an m shell counts towards the stop only past the largest m shell of this k walk
        peak_m = 0.0
        reach_pos = 0
        reach_neg = 0
        for m in 1:lmax
            reach_pos = m
            shell_m_energy, shell_m_angular, shell_m_carter, walk_max = replay!(take(m), m, k, shell_k_energy, true)
            if walk_max > Max_flux
                Max_flux = walk_max
                Max_m = m
            end

            shell_k_energy += shell_m_energy
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            _push_trail!(m_trail, shell_m_energy)

            # Horizon shells are negative in the superradiant regime, so the shell ratio compares magnitudes.
            below_count_m = !_before_m_peak(m, k, x) && (abs(shell_m_energy) <= layer_threshold) && (abs(shell_m_energy) < peak_m) ? below_count_m + 1 : 0
            peak_m = max(peak_m, abs(shell_m_energy))
            if below_count_m >= minimum_consecutive
                break
            end
        end
        below_count_m < minimum_consecutive && _tail_hit(_geometric_tail(m_trail, lmax), tol, total_energy + shell_k_energy) && (lmax_hit = true)

        below_count_m = 0
        m_trail = Float64[]
        for m in -1:-1:-lmax
            reach_neg = -m
            shell_m_energy, shell_m_angular, shell_m_carter, _ = replay!(take(m), m, k, shell_k_energy, false)

            shell_k_energy += shell_m_energy
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            _push_trail!(m_trail, shell_m_energy)

            below_count_m = !_before_m_peak(m, k, x) && (abs(shell_m_energy) <= layer_threshold) && (abs(m) >= Max_m) ? below_count_m + 1 : 0
            below_count_m >= minimum_consecutive && break
        end
        below_count_m < minimum_consecutive && _tail_hit(_geometric_tail(m_trail, lmax), tol, total_energy + shell_k_energy) && (lmax_hit = true)
        # the next shell's reference, whether the m walk stopped by its rule or ran off lmax (as in the other sums)
        shell_k_energy_last = shell_k_energy
        prefetch === nothing || _walk_discard!(prefetch)
        prev_reach_pos = reach_pos
        prev_reach_neg = reach_neg

        push!(energy_list, shell_k_energy)
        push!(k_list, k)
        push!(angular_list, shell_k_angular); push!(carter_list, shell_k_carter)
        ctx.log_shell(s == -2 ? "infinity" : "horizon", "k", k, shell_k_energy, shell_k_angular, shell_k_carter, total_modes)
        total_energy += shell_k_energy
        total_angular += shell_k_angular
        total_carter += shell_k_carter
        floor_ref[] = _adaptive_branch_mode_floor(total_energy, tol, mode_abs_floor)
        k_threshold = _mode_cutoff_threshold(tol * abs(total_energy), floor_ref[])
        below_count_n = abs(shell_k_energy) <= k_threshold ? below_count_n + 1 : 0
        if below_count_n >= minimum_consecutive + 1
            k_reached = k
            break
        end
    end
    state = (totals = (total_energy, total_angular, total_carter), k_reached = k_reached, l_reached = l_reached,
        lmax_hit = lmax_hit, near_static_zero = near_static_zero, failed_modes = failed_modes, total_modes = total_modes,
        below_count_n = below_count_n, prev_reach_pos = prev_reach_pos, prev_reach_neg = prev_reach_neg,
        shell_k_energy_last = shell_k_energy_last, k_next = k_next, reused_modes = reused_modes)
    return (total_energy, total_angular, total_carter, k_reached, l_reached, lmax_hit, near_static_zero, failed_modes, total_modes, state)
end

function _inclined_mode_summation_scoped(scope, a, p, x, o::FluxOptions; lmax = 30, kmax = 20, resume = nothing)
    (; tol, K0, Kmax, sample_tol, record, mode_abs_floor, threaded_sampling, mode_parallelism, progress) = o
    record_path = o.record_path
    minimum_consecutive = something(o.minimum_consecutive, 2)

    if x == 1.0 || x == -1.0
        return circular_mode_summation(a, p; x, submission = scope, options = _with(o; lmax, minimum_consecutive))
    end

    K = K0
    record_path = record_path === nothing ? "inclined_mode_data_a_$(_record_tag(a))_p_$(_record_tag(p))_x_$(_record_tag(x)).h5" : record_path

    KG = submission_orbit!(scope)
    KG_sample = submission_master!(scope)
    # resume (see _with_auto_caps): the state of a pass with the same lmax and a smaller kmax that stopped at its kmax;
    # this pass continues its k loops from there (not with a record, whose file is written anew)
    prev = resume === nothing ? nothing : resume.prev
    track = resume !== nothing && resume.track && !record
    rs = prev isa NamedTuple && get(prev, :kind, nothing) === :inclined && prev.lmax == lmax && !record ? prev : nothing
    # replay (see _with_auto_caps): a pass with a smaller lmax; its l walks come from its walk ledgers
    rl = rs === nothing && prev isa NamedTuple && get(prev, :kind, nothing) === :inclined && prev.lmax < lmax &&
        prev.ledgers !== nothing && !record ? prev : nothing
    ledger_in = rl === nothing ? (nothing, nothing) : rl.ledgers
    lmax_prior = rl === nothing ? lmax : rl.lmax
    ledger_out = !track ? (nothing, nothing) : rs !== nothing ? something(rs.ledgers, (nothing, nothing)) : (_WalkLedger(), _WalkLedger())
    Energy_flux_inf = rs === nothing ? Float64[] : rs.Energy_flux_inf
    Energy_flux_hor = rs === nothing ? Float64[] : rs.Energy_flux_hor
    k_list_inf = rs === nothing ? Int64[] : rs.k_list_inf
    k_list_hor = rs === nothing ? Int64[] : rs.k_list_hor
    # angular-momentum and Carter-constant k shells alongside the energy ones (the k of each in k_list_*)
    Angular_flux_inf = rs === nothing ? Float64[] : rs.Angular_flux_inf
    Carter_flux_inf = rs === nothing ? Float64[] : rs.Carter_flux_inf
    Angular_flux_hor = rs === nothing ? Float64[] : rs.Angular_flux_hor
    Carter_flux_hor = rs === nothing ? Float64[] : rs.Carter_flux_hor
    # progress = true: one line per finished k shell (the shell index of inclined orbits is k; there is no n)
    t_start = rs === nothing ? time() : rs.t_start
    shells_done = rs === nothing ? Ref(0) : rs.shells_done
    log_shell = (branch, label, idx, E, L, Q, modes) -> begin
        progress || return nothing
        lock(_PROGRESS_LOCK) do
            shells_done[] += 1
            @printf("GSN shell (a, p, e, x) = (%s, %s, %s, %s) lmax=%d branch=%s %s=%d E=%.6e L=%.6e Q=%.6e modes=%d shells_done=%d elapsed=%.1fs\n",
                a, p, 0.0, x, lmax, branch, label, idx, E, L, Q, modes, shells_done[], time() - t_start)
            flush(stdout)
        end
        return nothing
    end
    model(x, p) = p[1] .* (x .+ 1) .^ p[2] .* exp.(-p[3] .* x)
    p0 = [1e-3, 0.0, 0.1]

    total_infinity_energy = 0.0
    total_infinity_angular = 0.0
    total_infinity_carter = 0.0
    total_horizon_energy = 0.0
    total_horizon_angular = 0.0
    total_horizon_carter = 0.0
    total_modes = 0
    k_reached_inf = nothing
    k_reached_hor = nothing
    below_count_n = 0
    # Highest l evaluated per boundary, and whether an l (or m) walk hit lmax with modes still above the cutoff.
    l_reached_inf = 0
    l_reached_hor = 0
    lmax_hit_inf = false
    lmax_hit_hor = false
    # Near-static modes recorded as zero after a matching overflow (see ConvolutionIntegrals.NEAR_STATIC_RATIO).
    near_static_zero_inf = 0
    near_static_zero_hor = 0
    failed_modes_inf = 0
    failed_modes_hor = 0

    mode_index = Ref(0)
    ci_kwargs = (Kmax = Kmax, threaded_sampling = threaded_sampling)
    branch_mode_floor_inf = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_inf
    branch_mode_floor_hor = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_hor
    # One mode from its parameters (max_flux, floor; the k = 0 walks use the default max_flux 1.0); the walks do the
    # bookkeeping. The k >= 1 shells are walked by _inclined_branch_shells! (computed ahead with mode_parallelism > 1).
    inc_compute = (s, l, m, k, max_flux, floor, arena) -> with_submission_mode(scope; mode = (s, l, m, 0, k)) do mode_cache
        ConvolutionIntegrals.convolution_integral_inclined_trapezoidal_isem(KG_sample, s, l, m, k, K; tol = tol, sample_tol = sample_tol, max_flux = max_flux, mode_abs_floor = floor, ci_kwargs..., cache = mode_cache, arena)
    end
    inc_mode = inc_compute
    inc_compute_q = (s, l, m, k, q, arena) -> inc_compute(s, l, m, k, q.max_flux, q.floor, arena)
    pf_inf = nothing   # the walk prefetches of the two boundaries (mode_parallelism > 1), set below, drained in finally
    pf_hor = nothing

    record_h5 = record ? h5open(record_path, "w") : nothing
    try
        if record
            record_h5["meta/a"] = a
            record_h5["meta/p"] = p
            record_h5["meta/x"] = x
            record_h5["meta/K_requested"] = K
            record_h5["meta/Kmax"] = Kmax
            record_h5["meta/tol"] = tol
            record_h5["meta/lmax"] = lmax
            record_h5["meta/kmax"] = kmax
            record_h5["meta/minimum_consecutive"] = minimum_consecutive
            record_h5["meta/sample_tol"] = sample_tol
        end

        shell_k_infinity_energy = 0.0
        shell_k_infinity_angular = 0.0
        shell_k_infinity_carter = 0.0
        shell_k_horizon_energy = 0.0
        shell_k_horizon_angular = 0.0
        shell_k_horizon_carter = 0.0

        # The k = 0 m walks stop per boundary once minimum_consecutive m shells past the largest one are below tol
        # times the running k = 0 total, so modes far below the cutoff (which can underflow to NaN, l = m ~ 20 at
        # p ~ 2e5) are never evaluated.
        done_inf = false
        done_hor = false
        below_m_inf = 0
        below_m_hor = 0
        peak_m_inf = 0.0
        peak_m_hor = 0.0
        m_trail_inf = Float64[]
        m_trail_hor = Float64[]
        k0_reach = 0   # the m reached by the k = 0 walks, the first plan of the k = 1 walks
        shell_k_infinity_energy_last = 0.0
        shell_k_horizon_energy_last = 0.0
        if rs !== nothing
            # the k = 0 shell's counts; its sums are in the boundary states
            total_modes, near_static_zero_inf, near_static_zero_hor, failed_modes_inf, failed_modes_hor = rs.k0_counts
        else
        for m in 1:lmax
            k0_reach = m
            k0_arena = ConvolutionIntegrals._series_arena()   # work arena of the closed-form series for this m's modes
            lmin = max(2, abs(m))
            shell_m_infinity_energy = 0.0
            shell_m_infinity_angular = 0.0
            shell_m_infinity_carter = 0.0
            shell_m_horizon_energy = 0.0
            shell_m_horizon_angular = 0.0
            shell_m_horizon_carter = 0.0
            below_count_l = 0
            if !done_inf
                l_trail = Float64[]

                for l in lmin:lmax
                    mode_inf = inc_mode(-2, l, m, 0, 1.0, MODE_REFINE_FLOOR_FRACTION *
                        _adaptive_branch_mode_floor(shell_k_infinity_energy + shell_m_infinity_energy, tol, mode_abs_floor), k0_arena)
                    _record_current_mode!(record_h5, record, "I", mode_index, mode_inf; l, m, k = 0, signed_k = false, K_sample = K)
                    get(mode_inf, "NearStaticOverflow", false) && (near_static_zero_inf += 1)
                    get(mode_inf, "FailedMode", false) && (failed_modes_inf += 1)
                    shell_m_infinity_energy += 2 * mode_inf["EnergyFlux"]
                    shell_m_infinity_angular += 2 * mode_inf["AngularMomentumFlux"]
                    shell_m_infinity_carter += 2 * mode_inf["CarterConstantFlux"]
                    total_modes += 1
                    l_reached_inf = max(l_reached_inf, l)
                    _push_trail!(l_trail, mode_inf["EnergyFlux"])
                    layer_threshold = _mode_cutoff_threshold(tol * abs(shell_m_infinity_energy), _adaptive_branch_mode_floor(shell_k_infinity_energy + shell_m_infinity_energy, tol, mode_abs_floor))
                    below_count_l = abs(mode_inf["EnergyFlux"]) <= layer_threshold ? below_count_l + 1 : 0
                    if below_count_l >= minimum_consecutive
                        break
                    end
                end
                # A walk that ran to lmax is truncated when its extrapolated tail matters for the boundary total; the cutoff
                # inside the walk is relative to this m shell, so short walks at large m always end above it.
                below_count_l < minimum_consecutive && _tail_hit(2 * _geometric_tail(l_trail, lmax), tol, shell_k_infinity_energy + shell_m_infinity_energy) && (lmax_hit_inf = true)

                shell_k_infinity_energy += shell_m_infinity_energy
                shell_k_infinity_angular += shell_m_infinity_angular
                shell_k_infinity_carter += shell_m_infinity_carter
                _push_trail!(m_trail_inf, shell_m_infinity_energy)
                peak_m_inf = max(peak_m_inf, abs(shell_m_infinity_energy))
                below_m_inf = abs(shell_m_infinity_energy) < peak_m_inf && abs(shell_m_infinity_energy) <= tol * abs(shell_k_infinity_energy) ? below_m_inf + 1 : 0
                done_inf = below_m_inf >= minimum_consecutive
            end

            if !done_hor
                below_count_l = 0
                l_trail = Float64[]
                for l in lmin:lmax
                    mode_hor = inc_mode(2, l, m, 0, 1.0, MODE_REFINE_FLOOR_FRACTION *
                        _adaptive_branch_mode_floor(shell_k_horizon_energy + shell_m_horizon_energy, tol, mode_abs_floor), k0_arena)
                    _record_current_mode!(record_h5, record, "H", mode_index, mode_hor; l, m, k = 0, signed_k = false, K_sample = K)
                    get(mode_hor, "NearStaticOverflow", false) && (near_static_zero_hor += 1)
                    get(mode_hor, "FailedMode", false) && (failed_modes_hor += 1)
                    shell_m_horizon_energy += 2 * mode_hor["EnergyFlux"]
                    shell_m_horizon_angular += 2 * mode_hor["AngularMomentumFlux"]
                    shell_m_horizon_carter += 2 * mode_hor["CarterConstantFlux"]
                    total_modes += 1
                    l_reached_hor = max(l_reached_hor, l)
                    _push_trail!(l_trail, mode_hor["EnergyFlux"])
                    layer_threshold = _mode_cutoff_threshold(tol * abs(shell_m_horizon_energy), _adaptive_branch_mode_floor(shell_k_horizon_energy + shell_m_horizon_energy, tol, mode_abs_floor))
                    below_count_l = abs(mode_hor["EnergyFlux"]) <= layer_threshold ? below_count_l + 1 : 0
                    if below_count_l >= minimum_consecutive
                        break
                    end
                end
                # A walk that ran to lmax is truncated when its extrapolated tail matters for the boundary total; the cutoff
                # inside the walk is relative to this m shell, so short walks at large m always end above it.
                below_count_l < minimum_consecutive && _tail_hit(2 * _geometric_tail(l_trail, lmax), tol, shell_k_horizon_energy + shell_m_horizon_energy) && (lmax_hit_hor = true)

                shell_k_horizon_energy += shell_m_horizon_energy
                shell_k_horizon_angular += shell_m_horizon_angular
                shell_k_horizon_carter += shell_m_horizon_carter
                _push_trail!(m_trail_hor, shell_m_horizon_energy)
                peak_m_hor = max(peak_m_hor, abs(shell_m_horizon_energy))
                below_m_hor = abs(shell_m_horizon_energy) < peak_m_hor && abs(shell_m_horizon_energy) <= tol * abs(shell_k_horizon_energy) ? below_m_hor + 1 : 0
                done_hor = below_m_hor >= minimum_consecutive
            end
            done_inf && done_hor && break
        end

        done_inf || _tail_hit(_geometric_tail(m_trail_inf, lmax), tol, shell_k_infinity_energy) && (lmax_hit_inf = true)
        done_hor || _tail_hit(_geometric_tail(m_trail_hor, lmax), tol, shell_k_horizon_energy) && (lmax_hit_hor = true)

        push!(Energy_flux_inf, shell_k_infinity_energy)
        push!(Energy_flux_hor, shell_k_horizon_energy)
        push!(k_list_inf, 0)
        push!(k_list_hor, 0)
        push!(Angular_flux_inf, shell_k_infinity_angular); push!(Carter_flux_inf, shell_k_infinity_carter)
        push!(Angular_flux_hor, shell_k_horizon_angular); push!(Carter_flux_hor, shell_k_horizon_carter)
        log_shell("infinity", "k", 0, shell_k_infinity_energy, shell_k_infinity_angular, shell_k_infinity_carter, total_modes)
        log_shell("horizon", "k", 0, shell_k_horizon_energy, shell_k_horizon_angular, shell_k_horizon_carter, total_modes)
        total_infinity_energy += shell_k_infinity_energy
        total_infinity_angular += shell_k_infinity_angular
        total_infinity_carter += shell_k_infinity_carter
        total_horizon_energy += shell_k_horizon_energy
        total_horizon_angular += shell_k_horizon_angular
        total_horizon_carter += shell_k_horizon_carter

        shell_k_infinity_energy_last = shell_k_infinity_energy
        shell_k_horizon_energy_last = shell_k_horizon_energy
        branch_mode_floor_inf[] = _adaptive_branch_mode_floor(total_infinity_energy, tol, mode_abs_floor)
        branch_mode_floor_hor[] = _adaptive_branch_mode_floor(total_horizon_energy, tol, mode_abs_floor)
        end
        k0_counts = (total_modes, near_static_zero_inf, near_static_zero_hor, failed_modes_inf, failed_modes_hor)

        # k >= 1 shells of both boundaries (see _inclined_branch_shells!), each from the state after the k = 0 shell;
        # with mode_parallelism > 1 their l walks are computed ahead and, without a record, the boundaries run on two tasks
        walk_run = (key, q; arena = ConvolutionIntegrals._series_arena()) -> _inclined_l_walk(inc_compute_q, key[1], key[2], key[4], q, lmax, minimum_consecutive, x, mode_parallelism > 1;
            prior = _prior_walk(ledger_in[key[1] == -2 ? 1 : 2], key, lmax, lmax_prior), lmax_prior = lmax_prior, arena)
        branch_ctx = s -> (lmax = lmax, kmax = kmax, minimum_consecutive = minimum_consecutive, tol = tol, x = x, model = model,
            p0 = p0, mode_abs_floor = mode_abs_floor, k0_reach = k0_reach,
            energy_list = s == -2 ? Energy_flux_inf : Energy_flux_hor, angular_list = s == -2 ? Angular_flux_inf : Angular_flux_hor,
            carter_list = s == -2 ? Carter_flux_inf : Carter_flux_hor, k_list = s == -2 ? k_list_inf : k_list_hor,
            floor_ref = s == -2 ? branch_mode_floor_inf : branch_mode_floor_hor,
            totals = s == -2 ? (total_infinity_energy, total_infinity_angular, total_infinity_carter) :
                (total_horizon_energy, total_horizon_angular, total_horizon_carter),
            shell_k_energy_last = s == -2 ? shell_k_infinity_energy_last : shell_k_horizon_energy_last,
            l_reached = s == -2 ? l_reached_inf : l_reached_hor, lmax_hit = s == -2 ? lmax_hit_inf : lmax_hit_hor,
            run = let arena = ConvolutionIntegrals._series_arena(); (key, q) -> walk_run(key, q; arena) end, log_shell = log_shell,
            record! = (boundary, k, m, l, mode) -> _record_current_mode!(record_h5, record, boundary, mode_index, mode; l, m, k, signed_k = false, K_sample = K),
            resume = rs === nothing ? nothing : s == -2 ? rs.infinity : rs.horizon, ledger_out = ledger_out[s == -2 ? 1 : 2])
        pf_inf = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_inf) : nothing
        pf_hor = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_hor) : nothing
        inf, hor = _run_boundaries(() -> _inclined_branch_shells!(-2, branch_ctx(-2), pf_inf),
            () -> _inclined_branch_shells!(2, branch_ctx(2), pf_hor), mode_parallelism > 1 && !record)
        total_infinity_energy, total_infinity_angular, total_infinity_carter, k_reached_inf, l_reached_inf, lmax_hit_inf, nsz_inf, fm_inf, modes_inf, state_inf = inf
        total_horizon_energy, total_horizon_angular, total_horizon_carter, k_reached_hor, l_reached_hor, lmax_hit_hor, nsz_hor, fm_hor, modes_hor, state_hor = hor
        near_static_zero_inf += nsz_inf
        near_static_zero_hor += nsz_hor
        failed_modes_inf += fm_inf
        failed_modes_hor += fm_hor
        total_modes += modes_inf + modes_hor

        # A k sum that ran to kmax is truncated when the geometric tail of its last three k shells matters.
        kmax_hit_inf = k_reached_inf === nothing && _tail_hit(_geometric_tail(abs.(Energy_flux_inf[max(1, end - 2):end]), kmax), tol, total_infinity_energy)
        kmax_hit_hor = k_reached_hor === nothing && _tail_hit(_geometric_tail(abs.(Energy_flux_hor[max(1, end - 2):end]), kmax), tol, total_horizon_energy)

        if record
            record_h5["meta/infinity_energy_flux"] = total_infinity_energy
            record_h5["meta/infinity_angular_momentum_flux"] = total_infinity_angular
            record_h5["meta/infinity_carter_constant_flux"] = total_infinity_carter
            record_h5["meta/horizon_energy_flux"] = total_horizon_energy
            record_h5["meta/horizon_angular_momentum_flux"] = total_horizon_angular
            record_h5["meta/horizon_carter_constant_flux"] = total_horizon_carter
            record_h5["meta/k_reached_inf"] = something(k_reached_inf, -1)
            record_h5["meta/k_reached_hor"] = something(k_reached_hor, -1)
            record_h5["meta/l_reached_inf"] = l_reached_inf
            record_h5["meta/l_reached_hor"] = l_reached_hor
            record_h5["meta/lmax_hit_inf"] = lmax_hit_inf
            record_h5["meta/lmax_hit_hor"] = lmax_hit_hor
            record_h5["meta/near_static_zero_inf"] = near_static_zero_inf
            record_h5["meta/near_static_zero_hor"] = near_static_zero_hor
            record_h5["meta/total_modes"] = total_modes
            record_h5["meta/infinity_energy_flux_list"] = Energy_flux_inf
            record_h5["meta/horizon_energy_flux_list"] = Energy_flux_hor
        end

        return (
            infinity_energy_flux = total_infinity_energy,
            infinity_angular_momentum_flux = total_infinity_angular,
            infinity_carter_constant_flux = total_infinity_carter,
            horizon_energy_flux = total_horizon_energy,
            horizon_angular_momentum_flux = total_horizon_angular,
            horizon_carter_constant_flux = total_horizon_carter,
            total_modes = total_modes,
            tolerance = tol,
            minimum_consecutive = minimum_consecutive,
            l_max = lmax,
            k_max = kmax,
            K_sample = K,
            K_max = Kmax,
            k_reached_inf = k_reached_inf,
            k_reached_hor = k_reached_hor,
            l_reached_inf = l_reached_inf,
            l_reached_hor = l_reached_hor,
            lmax_hit_inf = lmax_hit_inf,
            lmax_hit_hor = lmax_hit_hor,
            kmax_hit_inf = kmax_hit_inf,
            kmax_hit_hor = kmax_hit_hor,
            near_static_zero_inf = near_static_zero_inf,
            near_static_zero_hor = near_static_zero_hor,
            failed_modes_inf = failed_modes_inf,
            failed_modes_hor = failed_modes_hor,
            infinity_truncation_floor = branch_mode_floor_inf[],
            horizon_truncation_floor = branch_mode_floor_hor[],
            mode_parallelism = mode_parallelism,
            modes_computed = pf_inf === nothing ? nothing : pf_inf.modes + pf_hor.modes,
            modes_computed_unused = pf_inf === nothing ? nothing : (pf_inf.modes - pf_inf.modes_used) + (pf_hor.modes - pf_hor.modes_used),
            # every k shell (symmetry weights included)
            infinity_shells = (k = k_list_inf, energy = Energy_flux_inf, angular_momentum = Angular_flux_inf, carter_constant = Carter_flux_inf),
            horizon_shells = (k = k_list_hor, energy = Energy_flux_hor, angular_momentum = Angular_flux_hor, carter_constant = Carter_flux_hor),
            infinity_energy_flux_list = Energy_flux_inf,
            horizon_energy_flux_list = Energy_flux_hor,
            record_file = record ? record_path : nothing,
            modes_new = total_modes - (rs !== nothing ? rs.total_modes : state_inf.reused_modes + state_hor.reused_modes),
            _resume = record ? nothing : (; kind = :inclined, lmax = lmax, infinity = state_inf, horizon = state_hor, k0_counts,
                total_modes, ledgers = track ? ledger_out : nothing,
                Energy_flux_inf, Energy_flux_hor, k_list_inf, k_list_hor, Angular_flux_inf, Carter_flux_inf, Angular_flux_hor,
                Carter_flux_hor, branch_mode_floor_inf, branch_mode_floor_hor, t_start, shells_done, pf_inf, pf_hor),
        )
    finally
        pf_inf === nothing || _walk_discard!(pf_inf)
        pf_hor === nothing || _walk_discard!(pf_hor)
        record && close(record_h5)
    end
end

# Total inclined circular fluxes, summed over k shells as the eccentric summation sums n shells.
function inclined_mode_summation(a, p, x; submission = nothing, stop_requested = nothing, options = nothing, kwargs...)
    o = _sum_options(options, submission, kwargs)
    caps = _auto_caps_start(a, p, 0.0, x, o.tol, o.lmax, o.kmax)
    scope, owned = ensure_public_submission(submission, a, p, 0.0, x; Nmax = o.Nmax, Kmax = o.Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        _with_auto_caps(caps) do L, Kc, Nc, resume
            with_submission_operation(scope) do
                _inclined_mode_summation_scoped(scope, a, p, x, o; lmax = L, kmax = Kc, resume)
            end
        end
    end
end

# One l walk of the generic sum, (s, m, n, k): for k = 0 up from lmin with the consecutive and deep stops, for k != 0
# as in _branch_l_walk (search for the peak l), from the shell's parameters q only (max_flux, floor, levin_floor and
# levin_flux_scale of the modes; the Levin choice of the shell, completed by the auto rule of (s, m, n, k) here;
# threshold, search_threshold and deep of the stop rules): pure. Returns the (l, mode) in walk order; `strip` drops
# the large members of the mode Dicts. A mode's non-finite flux is the error of the walk.
function _generic_l_walk(compute, s::Int, m::Int, n::Int, k::Int, q, lmax::Int, minimum_consecutive::Int, x, strip::Bool;
        prior = nothing, lmax_prior::Int = 0, arena = ConvolutionIntegrals._series_arena())
    factors = Dict{Any, Any}()
    modes = Tuple{Int, Any}[]
    prior === nothing || _continues(prior, lmax, lmax_prior) ||
        return (modes = modes, tail = prior.tail, prior = prior, state = _entry_state(prior))
    use_levin = q.use_levin_shell || q.levin_auto(s, m, n, k)
    mq = (max_flux = q.max_flux, use_levin = use_levin, floor = q.floor, levin_floor = q.levin_floor,
        levin_flux_scale = use_levin ? q.levin_flux_scale : 0.0)
    eval_l = l -> begin
        mode = compute(s, l, m, n, k, mq, factors, arena)
        ConvolutionIntegrals._check_finite_mode_flux(mode, s, l, m, n, k)
        strip && _strip_mode!(mode)
        push!(modes, (l, mode))
        mode["EnergyFlux"]
    end
    lmin = max(2, abs(m))
    if k == 0
        below = prior === nothing ? 0 : Int(prior.below)
        walk_max = prior === nothing ? 0.0 : prior.walk_max
        stopped = false
        for l in (prior === nothing ? lmin : lmax_prior + 1):lmax
            E = abs(eval_l(l))
            below = E <= q.threshold ? below + 1 : 0
            (below >= minimum_consecutive || _deep_stop(E, q.deep, q.threshold, walk_max)) && (stopped = true; break)
            walk_max = max(walk_max, E)
        end
        state = stopped ? _WALK_STATE_DONE : _walk_state(WALK_UP; below = below, walk_max = walk_max)
        return (modes = modes, tail = 0.0, prior = prior, state = state)
    end
    walk = prior === nothing ?
        _branch_l_walk(eval_l, lmin, _branch_peak_l(lmin, m, k, x), lmax, q.threshold, minimum_consecutive, s == 2;
            deep = q.deep, search_threshold = q.search_threshold) :
        _continue_branch_l_walk(eval_l, prior, lmax_prior, lmax, q.threshold, minimum_consecutive;
            deep = q.deep, search_threshold = q.search_threshold)
    return (modes = modes, tail = walk.tail, prior = prior, state = walk.state)
end

# The m reach (m > 0 walks, m < 0 walks) planned for row k from the rows of the previous shell: that row, else the
# nearest row towards k = 0.
function _generic_row_reach(prev_rows, k::Int)
    kk = k
    while !haskey(prev_rows, kk) && kk != 0
        kk -= sign(kk)
    end
    return get(prev_rows, kk, (1, 1))
end

# One boundary of the generic sum (s = -2: infinity, 2: horizon): the n = 0 shell (its k = 0 m walks serial, with the
# running cutoff), then the n >= 1 shells, each over the k = 0 row and the k = 1, 2, .. and k = -1, -2, .. rows of m = 0,
# m > 0 and m < 0 l walks, with the m, k and n stop rules. The l walks of a row depend on the row's parameters only:
# with `prefetch` they are computed ahead, a whole shell planned from the reach of the previous one. The two boundaries
# may run on different tasks: everything mutable here is local or owned by this boundary (ctx). The result's `state`
# is what the next n shell needs, from which a pass with a larger nmax continues (ctx.resume).
function _generic_branch_summation!(s::Int, ctx, prefetch)
    (; e, x, tol, lmax, kmax, nmax, minimum_consecutive, mode_abs_floor, beta, deep, neg_branch_scale,
        radial_over_azimuthal, polar_over_azimuthal, floor_ref, run) = ctx
    # the n stop spans one period of the n shells near the separatrix (zoom-whirl), as for eccentric orbits
    n_stop_run = _n_stop_run(minimum_consecutive, radial_over_azimuthal)
    params_for = nothing
    # the walk key, computed ahead or now; a walk beyond the plan starts the next two of its direction
    take = key -> begin
        walk = if prefetch === nothing
            run(key, params_for(key))
        else
            m, n, k = key[2], key[3], key[4]
            step = m >= 0 ? 1 : -1
            prefetch(key; ahead = [(s, mm, n, k) for mm in (m + step):step:(m + 2step) if abs(mm) <= lmax])
        end
        walk.prior === nothing || ctx.prior_counts!(s, walk.prior)
        ctx.ledger_out === nothing || (ctx.ledger_out[key] = _ledger_entry(walk.prior, walk.modes, walk.state, walk.tail))
        return walk
    end
    plan_rows! = (n, ks, prev_rows) -> begin
        prefetch === nothing && return nothing
        prefetch.params_for = params_for
        plan = NTuple{4, Int}[]
        for k in ks
            rp, rn = _generic_row_reach(prev_rows, k)
            push!(plan, (s, 0, n, k))
            for m in 1:min(lmax, rp)
                push!(plan, (s, m, n, k))
            end
            for m in 1:min(lmax, rn)
                push!(plan, (s, -m, n, k))
            end
        end
        _walk_plan!(prefetch, plan)
        return nothing
    end
    r = ctx.resume
    r === nothing || r.n_reached === nothing || return merge(r.result, (state = r,))
    shell_list = r === nothing ? Float64[] : r.shell_list
    angular_list = r === nothing ? Float64[] : r.angular_list   # L and Q of each n shell, alongside shell_list
    carter_list = r === nothing ? Float64[] : r.carter_list
    last_shell = r === nothing ? (0.0, 0.0, 0.0) : r.last_shell
    n_list = r === nothing ? Int[] : r.n_list
    total_energy, total_angular, total_carter = r === nothing ? (0.0, 0.0, 0.0) : r.totals
    total_modes = r === nothing ? 0 : r.total_modes
    n_reached = nothing
    shell_last = r === nothing ? 0.0 : r.shell_last
    row_reach = r === nothing ? Dict{Int, Tuple{Int, Int}}() : r.row_reach   # k => (m reach of the m > 0 walks, of the m < 0 walks) of the last shell
    below_count_n = r === nothing ? 0 : r.below_count_n
    n_first = r === nothing ? 1 : r.n_next

    if r === nothing
    shell_n = 0.0
    shell_n_angular = 0.0
    shell_n_carter = 0.0
    below_count_m = 0
    reach_pos = 0
    reach_neg = 0
    for m in 1:lmax
        reach_pos = m
        lmin = max(2, abs(m))
        shell_m = 0.0
        shell_m_angular = 0.0
        shell_m_carter = 0.0
        below_count_l = 0
        max_flux = 1e-2
        for l in lmin:lmax
            mode = ctx.eval_mode(s, l, m, 0, 0, max_flux)
            ctx.record_mode(s, l, m, 0, 0, mode)
            shell_m += 2 * mode["EnergyFlux"]
            shell_m_angular += 2 * mode["AngularMomentumFlux"]
            shell_m_carter += 2 * mode["CarterConstantFlux"]
            shell_n += 2 * mode["EnergyFlux"]
            shell_n_angular += 2 * mode["AngularMomentumFlux"]
            shell_n_carter += 2 * mode["CarterConstantFlux"]
            total_modes += 1
            below_count_l = abs(mode["EnergyFlux"]) <= tol * abs(shell_n) ? below_count_l + 1 : 0
            floor_ref[] = _adaptive_branch_mode_floor(shell_n, tol, mode_abs_floor)
            if below_count_l >= minimum_consecutive
                break
            end
        end
        below_count_m = abs(shell_m) <= tol * abs(shell_n) ? below_count_m + 1 : 0
        if below_count_m >= minimum_consecutive
            break
        end
    end

    initial_floor = _adaptive_branch_mode_floor(total_energy + shell_n, tol, mode_abs_floor)
    initial_mode_threshold = _mode_cutoff_threshold(tol * max(abs(total_energy + shell_n), abs(shell_n)), initial_floor)
    initial_k_shell_threshold = _generic_k_shell_threshold(total_energy + shell_n, shell_n, tol, initial_floor)
    below_count_k = 0
    peak_k_shell = 0.0
    row_reach[0] = (reach_pos, reach_pos)
    k_reach_pos = 0
    k_reach_neg = 0
    for k in 1:kmax
        k_reach_pos = k
        shell_k = 0.0
        shell_k_angular = 0.0
        shell_k_carter = 0.0
        max_flux = abs(shell_n)
        # the modes of this row from its parameters (as gen_parameters for the serial walk)
        row_floor = MODE_REFINE_FLOOR_FRACTION * floor_ref[]
        shell_q = (max_flux = max_flux, floor = row_floor, levin_floor = _tail_levin_mode_floor(row_floor, ctx.levin_floor(s)),
            threshold = initial_mode_threshold, search_threshold = initial_mode_threshold, deep = deep,
            use_levin_shell = ctx.use_shell_levin(s, 0), levin_auto = ctx.levin_auto, levin_flux_scale = max(abs(sum(shell_list)), abs(max_flux)))
        params_for = key -> shell_q
        plan_rows!(0, [k], row_reach)   # the next row's max_flux is known at its start only
        reach_pos = 0
        reach_neg = 0

        shell_m = 0.0
        shell_m_angular = 0.0
        shell_m_carter = 0.0
        below_count_l = 0
        walk = take((s, 0, 0, k))
        if walk.prior !== nothing
            pe, pa, pc, pn = _prior_sums(walk.prior)
            shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
            shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
            total_modes += pn
        end
        for (l, mode) in walk.modes
            mode = ctx.accept!(s, l, 0, 0, k, mode)
            ctx.record_mode(s, l, 0, 0, k, mode)
            shell_m += 2 * mode["EnergyFlux"]
            shell_m_angular += 2 * mode["AngularMomentumFlux"]
            shell_m_carter += 2 * mode["CarterConstantFlux"]
            shell_n += 2 * mode["EnergyFlux"]
            shell_n_angular += 2 * mode["AngularMomentumFlux"]
            shell_n_carter += 2 * mode["CarterConstantFlux"]
            total_modes += 1
        end
        shell_k += shell_m
        shell_k_angular += shell_m_angular
        shell_k_carter += shell_m_carter

        below_count_m = 0
        previous_shell_m = Inf
        Max_flux = 0.0
        Max_m = 1
        for m in 1:lmax
            reach_pos = m
            lmin = max(2, abs(m))
            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk = take((s, m, 0, k))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, 0, k, mode)
                ctx.record_mode(s, l, m, 0, k, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            if abs(shell_m) > Max_flux
                Max_flux = abs(shell_m)
                Max_m = m
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
            growing_m = abs(shell_m) > abs(previous_shell_m)
            previous_shell_m = shell_m
            below_count_m = !growing_m && !_before_m_peak(m, k, x) && _positive_m_tail_below(shell_m, shell_k, initial_mode_threshold, tol, m, Max_m, mode_abs_floor) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end

        below_count_m = 0
        previous_shell_m = Inf
        for m in -1:-1:-lmax
            reach_neg = -m
            lmin = max(2, abs(m))
            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk = take((s, m, 0, k))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, 0, k, mode)
                ctx.record_mode(s, l, m, 0, k, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
            growing_m = abs(shell_m) > abs(previous_shell_m)
            previous_shell_m = shell_m
            below_count_m = !growing_m && !_before_m_peak(m, k, x) && _negative_m_tail_below(shell_m, initial_mode_threshold, neg_branch_scale, m, Max_m) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end

        row_reach[k] = (reach_pos, reach_neg)
        prefetch === nothing || _walk_discard!(prefetch)
        # a k shell counts towards the stop only past the largest k shell (shells can rise before they fall)
        below_count_k = abs(shell_k) <= initial_k_shell_threshold && abs(shell_k) < peak_k_shell ? below_count_k + 1 : 0
        peak_k_shell = max(peak_k_shell, abs(shell_k))
        if below_count_k >= minimum_consecutive
            break
        end
    end

    push!(shell_list, shell_n)
    push!(n_list, 0)
    push!(angular_list, shell_n_angular); push!(carter_list, shell_n_carter)
    ctx.log_shell(s == -2 ? "infinity" : "horizon", "n", 0, shell_n, shell_n_angular, shell_n_carter, total_modes)
    total_energy += shell_n
    total_angular += shell_n_angular
    total_carter += shell_n_carter
    floor_ref[] = _adaptive_branch_mode_floor(total_energy, tol, mode_abs_floor)
    shell_last = shell_n
    end

    n_next = n_first
    for n in n_first:nmax
        n_next = n + 1
        m_omega_zero = -n * radial_over_azimuthal
        max_flux = 2 * abs(ctx.fit_estimate(n_list, shell_list, n, abs(shell_last)))
        shell_n = 0.0
        shell_n_angular = 0.0
        shell_n_carter = 0.0
        floor = floor_ref[]
        search_threshold = _mode_cutoff_threshold(tol * abs(shell_last), floor)
        threshold = max(search_threshold, beta * tol * abs(total_energy))
        k_shell_scale = sqrt(abs(total_energy) * abs(shell_last))
        # k-shell stops are relative to the branch total; the absolute per-mode floor would cut whole
        # k shells when the branch total is small (horizon totals of 1e-8..1e-7).
        k_shell_threshold = tol * k_shell_scale
        negative_k_shell_threshold = neg_branch_scale * tol * abs(total_energy)
        use_levin_shell = ctx.use_shell_levin(s, n)
        # the modes of this shell from its parameters (as gen_parameters for the serial walk); the walks of the shell
        # are planned from the reach of the previous shell, rows (k) and m walks
        mode_floor = MODE_REFINE_FLOOR_FRACTION * floor
        shell_q = (max_flux = max_flux, floor = mode_floor, levin_floor = _tail_levin_mode_floor(mode_floor, ctx.levin_floor(s)),
            threshold = threshold, search_threshold = search_threshold, deep = deep, use_levin_shell = use_levin_shell,
            levin_auto = ctx.levin_auto, levin_flux_scale = max(abs(sum(shell_list)), abs(max_flux)))
        params_for = key -> shell_q
        prev_row_reach = row_reach
        row_reach = Dict{Int, Tuple{Int, Int}}()
        plan_rows!(n, [0; 1:min(kmax, 1); -1:-1:-min(kmax, 1)], prev_row_reach)
        reach_pos = 0
        reach_neg = 0

        shell_k = 0.0
        shell_k_angular = 0.0
        shell_k_carter = 0.0
        shell_m = 0.0
        shell_m_angular = 0.0
        shell_m_carter = 0.0
        below_count_l = 0
        walk_max_l = 0.0
        walk = take((s, 0, n, 0))
        if walk.prior !== nothing
            pe, pa, pc, pn = _prior_sums(walk.prior)
            shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
            shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
            total_modes += pn
        end
        for (l, mode) in walk.modes
            mode = ctx.accept!(s, l, 0, n, 0, mode)
            ctx.record_mode(s, l, 0, n, 0, mode)
            shell_m += 2 * mode["EnergyFlux"]
            shell_m_angular += 2 * mode["AngularMomentumFlux"]
            shell_m_carter += 2 * mode["CarterConstantFlux"]
            shell_n += 2 * mode["EnergyFlux"]
            shell_n_angular += 2 * mode["AngularMomentumFlux"]
            shell_n_carter += 2 * mode["CarterConstantFlux"]
            total_modes += 1
        end

        shell_k += shell_m
        shell_k_angular += shell_m_angular
        shell_k_carter += shell_m_carter

        below_count_m = 0
        previous_shell_m = Inf
        Max_flux = 0.0
        Max_m = 1
        for m in 1:lmax
            reach_pos = m
            lmin = max(2, abs(m))
            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk_max_l = 0.0
            walk = take((s, m, n, 0))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, n, 0, mode)
                ctx.record_mode(s, l, m, n, 0, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            if abs(shell_m) > Max_flux
                Max_flux = abs(shell_m)
                Max_m = m
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
            growing_m = abs(shell_m) > abs(previous_shell_m)
            previous_shell_m = shell_m
            below_count_m = !growing_m && !_before_omega_zero(m, m_omega_zero) && _positive_m_tail_below(shell_m, shell_k, threshold, tol, m, Max_m, mode_abs_floor) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end

        below_count_m = 0
        previous_shell_m = Inf
        for m in -1:-1:-lmax
            reach_neg = -m
            lmin = max(2, abs(m))
            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk_max_l = 0.0
            walk = take((s, m, n, 0))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, m, n, 0, mode)
                ctx.record_mode(s, l, m, n, 0, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter
            # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
            growing_m = abs(shell_m) > abs(previous_shell_m)
            previous_shell_m = shell_m
            below_count_m = !growing_m && !_before_omega_zero(m, m_omega_zero) && _negative_m_tail_below(shell_m, threshold, neg_branch_scale, m, Max_m) ? below_count_m + 1 : 0
            if below_count_m >= minimum_consecutive
                break
            end
        end

        row_reach[0] = (reach_pos, reach_neg)
        k_reach_pos = 0
        k_reach_neg = 0

        below_count_k = 0
        previous_shell_k = Inf
        for k in 1:kmax
            k_reach_pos = k
            k < kmax && plan_rows!(n, [k + 1], prev_row_reach)
            reach_pos = 0
            reach_neg = 0
            shell_k = 0.0
            shell_k_angular = 0.0
            shell_k_carter = 0.0

            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk = take((s, 0, n, k))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, 0, n, k, mode)
                ctx.record_mode(s, l, 0, n, k, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter

            below_count_m = 0
            previous_shell_m = Inf
            Max_flux = 0.0
            Max_m = 1
            for m in 1:lmax
                reach_pos = m
                lmin = max(2, abs(m))
                shell_m = 0.0
                shell_m_angular = 0.0
                shell_m_carter = 0.0
                below_count_l = 0
                walk = take((s, m, n, k))
                if walk.prior !== nothing
                    pe, pa, pc, pn = _prior_sums(walk.prior)
                    shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                    shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                    total_modes += pn
                end
                for (l, mode) in walk.modes
                    mode = ctx.accept!(s, l, m, n, k, mode)
                    ctx.record_mode(s, l, m, n, k, mode)
                    shell_m += 2 * mode["EnergyFlux"]
                    shell_m_angular += 2 * mode["AngularMomentumFlux"]
                    shell_m_carter += 2 * mode["CarterConstantFlux"]
                    shell_n += 2 * mode["EnergyFlux"]
                    shell_n_angular += 2 * mode["AngularMomentumFlux"]
                    shell_n_carter += 2 * mode["CarterConstantFlux"]
                    total_modes += 1
                end
                if abs(shell_m) > Max_flux
                    Max_flux = abs(shell_m)
                    Max_m = m
                end
                shell_k += shell_m
                shell_k_angular += shell_m_angular
                shell_k_carter += shell_m_carter
                # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
                growing_m = abs(shell_m) > abs(previous_shell_m)
                previous_shell_m = shell_m
                below_count_m = !growing_m && !_before_m_peak(m, k, x) && !_before_omega_zero(m, -(n * radial_over_azimuthal + k * polar_over_azimuthal)) && _positive_m_tail_below(shell_m, shell_k, threshold, tol, m, Max_m, mode_abs_floor) ? below_count_m + 1 : 0
                if below_count_m >= minimum_consecutive
                    break
                end
            end

            below_count_m = 0
            previous_shell_m = Inf
            for m in -1:-1:-lmax
                reach_neg = -m
                lmin = max(2, abs(m))
                shell_m = 0.0
                shell_m_angular = 0.0
                shell_m_carter = 0.0
                below_count_l = 0
                walk = take((s, m, n, k))
                if walk.prior !== nothing
                    pe, pa, pc, pn = _prior_sums(walk.prior)
                    shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                    shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                    total_modes += pn
                end
                for (l, mode) in walk.modes
                    mode = ctx.accept!(s, l, m, n, k, mode)
                    ctx.record_mode(s, l, m, n, k, mode)
                    shell_m += 2 * mode["EnergyFlux"]
                    shell_m_angular += 2 * mode["AngularMomentumFlux"]
                    shell_m_carter += 2 * mode["CarterConstantFlux"]
                    shell_n += 2 * mode["EnergyFlux"]
                    shell_n_angular += 2 * mode["AngularMomentumFlux"]
                    shell_n_carter += 2 * mode["CarterConstantFlux"]
                    total_modes += 1
                end
                shell_k += shell_m
                shell_k_angular += shell_m_angular
                shell_k_carter += shell_m_carter
                # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
                growing_m = abs(shell_m) > abs(previous_shell_m)
                previous_shell_m = shell_m
                below_count_m = !growing_m && !_before_m_peak(m, k, x) && !_before_omega_zero(m, -(n * radial_over_azimuthal + k * polar_over_azimuthal)) && _negative_m_tail_below(shell_m, threshold, neg_branch_scale, m, Max_m) ? below_count_m + 1 : 0
                if below_count_m >= minimum_consecutive
                    break
                end
            end

            row_reach[k] = (reach_pos, reach_neg)
            # a k shell that is still growing does not count toward the stop: shells can dip, then rise again
            growing = abs(shell_k) > abs(previous_shell_k)
            previous_shell_k = shell_k
            below_count_k = !growing && abs(shell_k) <= k_shell_threshold ? below_count_k + 1 : 0
            if below_count_k >= minimum_consecutive
                break
            end
        end

        below_count_k = 0
        previous_shell_k = Inf
        for k in -1:-1:-kmax
            k_reach_neg = -k
            -k < kmax && plan_rows!(n, [k - 1], prev_row_reach)
            reach_pos = 0
            reach_neg = 0
            shell_k = 0.0
            shell_k_angular = 0.0
            shell_k_carter = 0.0

            shell_m = 0.0
            shell_m_angular = 0.0
            shell_m_carter = 0.0
            below_count_l = 0
            walk = take((s, 0, n, k))
            if walk.prior !== nothing
                pe, pa, pc, pn = _prior_sums(walk.prior)
                shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                total_modes += pn
            end
            for (l, mode) in walk.modes
                mode = ctx.accept!(s, l, 0, n, k, mode)
                ctx.record_mode(s, l, 0, n, k, mode)
                shell_m += 2 * mode["EnergyFlux"]
                shell_m_angular += 2 * mode["AngularMomentumFlux"]
                shell_m_carter += 2 * mode["CarterConstantFlux"]
                shell_n += 2 * mode["EnergyFlux"]
                shell_n_angular += 2 * mode["AngularMomentumFlux"]
                shell_n_carter += 2 * mode["CarterConstantFlux"]
                total_modes += 1
            end
            shell_k += shell_m
            shell_k_angular += shell_m_angular
            shell_k_carter += shell_m_carter

            below_count_m = 0
            previous_shell_m = Inf
            Max_flux = 0.0
            Max_m = 1
            for m in 1:lmax
                reach_pos = m
                lmin = max(2, abs(m))
                shell_m = 0.0
                shell_m_angular = 0.0
                shell_m_carter = 0.0
                below_count_l = 0
                walk = take((s, m, n, k))
                if walk.prior !== nothing
                    pe, pa, pc, pn = _prior_sums(walk.prior)
                    shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                    shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                    total_modes += pn
                end
                for (l, mode) in walk.modes
                    mode = ctx.accept!(s, l, m, n, k, mode)
                    ctx.record_mode(s, l, m, n, k, mode)
                    shell_m += 2 * mode["EnergyFlux"]
                    shell_m_angular += 2 * mode["AngularMomentumFlux"]
                    shell_m_carter += 2 * mode["CarterConstantFlux"]
                    shell_n += 2 * mode["EnergyFlux"]
                    shell_n_angular += 2 * mode["AngularMomentumFlux"]
                    shell_n_carter += 2 * mode["CarterConstantFlux"]
                    total_modes += 1
                end
                if abs(shell_m) > Max_flux
                    Max_flux = abs(shell_m)
                    Max_m = m
                end
                shell_k += shell_m
                shell_k_angular += shell_m_angular
                shell_k_carter += shell_m_carter
                # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
                growing_m = abs(shell_m) > abs(previous_shell_m)
                previous_shell_m = shell_m
                below_count_m = !growing_m && !_before_m_peak(m, k, x) && !_before_omega_zero(m, -(n * radial_over_azimuthal + k * polar_over_azimuthal)) && _positive_m_tail_below(shell_m, shell_k, threshold, tol, m, Max_m, mode_abs_floor) ? below_count_m + 1 : 0
                if below_count_m >= minimum_consecutive
                    break
                end
            end

            below_count_m = 0
            previous_shell_m = Inf
            for m in -1:-1:-lmax
                reach_neg = -m
                lmin = max(2, abs(m))
                shell_m = 0.0
                shell_m_angular = 0.0
                shell_m_carter = 0.0
                below_count_l = 0
                walk = take((s, m, n, k))
                if walk.prior !== nothing
                    pe, pa, pc, pn = _prior_sums(walk.prior)
                    shell_m += pe; shell_m_angular += pa; shell_m_carter += pc
                    shell_n += pe; shell_n_angular += pa; shell_n_carter += pc
                    total_modes += pn
                end
                for (l, mode) in walk.modes
                    mode = ctx.accept!(s, l, m, n, k, mode)
                    ctx.record_mode(s, l, m, n, k, mode)
                    shell_m += 2 * mode["EnergyFlux"]
                    shell_m_angular += 2 * mode["AngularMomentumFlux"]
                    shell_m_carter += 2 * mode["CarterConstantFlux"]
                    shell_n += 2 * mode["EnergyFlux"]
                    shell_n_angular += 2 * mode["AngularMomentumFlux"]
                    shell_n_carter += 2 * mode["CarterConstantFlux"]
                    total_modes += 1
                end
                shell_k += shell_m
                shell_k_angular += shell_m_angular
                shell_k_carter += shell_m_carter
                # an m shell that is still growing does not count toward the stop (retrograde orbits rise toward m < 0)
                growing_m = abs(shell_m) > abs(previous_shell_m)
                previous_shell_m = shell_m
                below_count_m = !growing_m && !_before_m_peak(m, k, x) && !_before_omega_zero(m, -(n * radial_over_azimuthal + k * polar_over_azimuthal)) && _negative_m_tail_below(shell_m, threshold, neg_branch_scale, m, Max_m) ? below_count_m + 1 : 0
                if below_count_m >= minimum_consecutive
                    break
                end
            end

            row_reach[k] = (reach_pos, reach_neg)
            # a k shell that is still growing does not count toward the stop: shells can dip, then rise again
            growing = abs(shell_k) > abs(previous_shell_k)
            previous_shell_k = shell_k
            below_count_k = !growing && abs(shell_k) <= negative_k_shell_threshold ? below_count_k + 1 : 0
            if below_count_k >= minimum_consecutive
                break
            end
        end

        prefetch === nothing || _walk_discard!(prefetch)
        push!(shell_list, shell_n)
        push!(n_list, n)
        push!(angular_list, shell_n_angular); push!(carter_list, shell_n_carter)
        ctx.log_shell(s == -2 ? "infinity" : "horizon", "n", n, shell_n, shell_n_angular, shell_n_carter, total_modes)
        last_shell = (shell_n, shell_n_angular, shell_n_carter)
        total_energy += shell_n
        total_angular += shell_n_angular
        total_carter += shell_n_carter
        floor_ref[] = _adaptive_branch_mode_floor(total_energy, tol, mode_abs_floor)
        n_threshold = _mode_cutoff_threshold(tol * abs(total_energy), floor_ref[])
        below_count_n = abs(shell_n) <= n_threshold ? below_count_n + 1 : 0
        shell_last = shell_n

        if below_count_n >= n_stop_run
            n_reached = n
            break
        end
    end

    state = (; totals = (total_energy, total_angular, total_carter), n_reached, shell_list, angular_list, carter_list, n_list,
        last_shell, total_modes, shell_last, row_reach, below_count_n, n_next)
    # Geometric n tail at high eccentricity, as for eccentric equatorial orbits.
    tail_factor, tail_ratio = _geometric_n_tail_factor(shell_list, e, n_reached, n_stop_run)
    n_tail = tail_factor .* last_shell
    total_energy += n_tail[1]
    total_angular += n_tail[2]
    total_carter += n_tail[3]

    result = (
        energy_flux = total_energy, angular_momentum_flux = total_angular, carter_constant_flux = total_carter,
        total_modes = total_modes, n_reached = n_reached, shell_list = shell_list,
        shells = (n = n_list, energy = shell_list, angular_momentum = angular_list, carter_constant = carter_list),
        truncation_floor = floor_ref[],
        n_tail = (energy = n_tail[1], angular_momentum = n_tail[2], carter_constant = n_tail[3], ratio = tail_ratio),
    )
    return merge(result, (state = merge(state, (result = result,)),))
end

function _generic_mode_summation_scoped(scope, a, p, e, x, o::FluxOptions; lmax = 30, kmax = 20, nmax = 500, resume = nothing)
    (; tol, N0, K0, Nmax, Kmax, sample_tol, record, mode_abs_floor, threaded_sampling, neg_branch_scale, beta, deep,
        tail_levin, tail_levin_infinity, tail_levin_horizon, levin_nmin, levin_local_n, levin_max_depth, mode_parallelism,
        progress) = o
    record_path = o.record_path
    minimum_consecutive = something(o.minimum_consecutive, 2)
    levin_mode_abs_floor = something(o.levin_mode_abs_floor, mode_abs_floor)

    if x == 1.0 || x == -1.0
        return eccentric_mode_summation(a, p, e; x, submission = scope, options = _with(o; lmax, nmax, minimum_consecutive))
    end
    if e == 0.0
        return inclined_mode_summation(a, p, x; options = _with(o; lmax, kmax, minimum_consecutive),
            submission = isequal(e, 0.0) ? scope : nothing,
            stop_requested = isequal(e, 0.0) ? nothing : (() -> _mode_flux_parent_stop_requested(scope)))
    end

    record_path = record_path === nothing ? "generic_mode_data_a_$(_record_tag(a))_p_$(_record_tag(p))_e_$(_record_tag(e))_x_$(_record_tag(x)).h5" : record_path

    KG = submission_orbit!(scope)
    radial_over_azimuthal = KG["Frequencies"]["ϒr"] / KG["Frequencies"]["ϒϕ"]
    F_orbit = KG["Frequencies"]
    polar_over_azimuthal = KG["Frequencies"]["ϒθ"] / KG["Frequencies"]["ϒϕ"]

    KG_master = submission_master!(scope)
    tail_policy = _tail_policy(tail_levin)
    tail_policy_inf = _tail_policy(tail_levin_infinity === nothing ? tail_policy : tail_levin_infinity)
    tail_policy_hor = _tail_policy(tail_levin_horizon === nothing ? tail_policy : tail_levin_horizon)
    model(x, p) = p[1] .* (x .+ 1) .^ p[2] .* exp.(-p[3] .* x)
    p0 = [1e-3, 0.0, 0.1]
    mode_index = Ref(0)

    fit_estimate = (idxs, vals, n, fallback) -> begin
        length(vals) < 3 && return fallback
        w = min(length(vals), 6)
        xdata = idxs[end - w + 1:end]
        ydata = vals[end - w + 1:end]
        try
            fit = curve_fit(model, xdata, ydata, p0)
            est = abs(model(n, fit.param))
            isfinite(est) ? est : fallback
        catch caught
            absorbable(caught) || rethrow()
            fallback
        end
    end

    pf_inf = nothing   # the walk prefetches of the two boundaries (mode_parallelism > 1), set below, drained in finally
    pf_hor = nothing
    record_h5 = record ? h5open(record_path, "w") : nothing
    try
        if record
            record_h5["meta/a"] = a
            record_h5["meta/p"] = p
            record_h5["meta/e"] = e
            record_h5["meta/x"] = x
            record_h5["meta/N_requested"] = N0
            record_h5["meta/K_requested"] = K0
            record_h5["meta/Nmax"] = Nmax
            record_h5["meta/Kmax"] = Kmax
            record_h5["meta/tol"] = tol
            record_h5["meta/lmax"] = lmax
            record_h5["meta/kmax"] = kmax
            record_h5["meta/nmax"] = nmax
            record_h5["meta/minimum_consecutive"] = minimum_consecutive
            record_h5["meta/sample_tol"] = sample_tol
            record_h5["meta/neg_branch_scale"] = neg_branch_scale
            record_h5["meta/levin_max_depth"] = levin_max_depth
        end

        # resume (see _with_auto_caps): the state of a pass with the same lmax and kmax and a smaller nmax that stopped at
        # its nmax; this pass continues its n loops from there (not with a record, whose file is written anew)
        prev = resume === nothing ? nothing : resume.prev
        track = resume !== nothing && resume.track && !record
        rs = prev isa NamedTuple && get(prev, :kind, nothing) === :generic && prev.lmax == lmax && prev.kmax == kmax && !record ? prev : nothing
        # replay (see _with_auto_caps): a pass with a smaller lmax or kmax; its l walks come from its walk ledgers
        rl = rs === nothing && prev isa NamedTuple && get(prev, :kind, nothing) === :generic && prev.lmax <= lmax && prev.kmax <= kmax &&
            prev.ledgers !== nothing && !record ? prev : nothing
        ledger_in = rl === nothing ? (nothing, nothing) : rl.ledgers
        lmax_prior = rl === nothing ? lmax : rl.lmax
        ledger_out = !track ? (nothing, nothing) : rs !== nothing ? something(rs.ledgers, (nothing, nothing)) : (_WalkLedger(), _WalkLedger())
        reused_modes = [0, 0]
        branch_mode_floor_inf = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_inf
        branch_mode_floor_hor = rs === nothing ? Ref(Float64(mode_abs_floor)) : rs.branch_mode_floor_hor
        levin_floor_tracks_mode_floor = levin_mode_abs_floor == mode_abs_floor
        branch_mode_floor = s -> s == -2 ? branch_mode_floor_inf[] : branch_mode_floor_hor[]
        branch_levin_floor = s -> levin_floor_tracks_mode_floor ? branch_mode_floor(s) : Float64(levin_mode_abs_floor)
        tail_levin_infinity_active = rs === nothing ? Ref(false) : rs.tail_levin_infinity_active
        tail_levin_horizon_active = rs === nothing ? Ref(false) : rs.tail_levin_horizon_active
        tail_levin_infinity_start_n = rs === nothing ? Ref{Union{Nothing, Int}}(nothing) : rs.tail_levin_infinity_start_n
        tail_levin_horizon_start_n = rs === nothing ? Ref{Union{Nothing, Int}}(nothing) : rs.tail_levin_horizon_start_n
        # tail_levin = true forces adaptive Levin for every n shell from levin_nmin on; start_n records the first one.
        # :auto decides per mode (gen_levin_auto).
        use_generic_tail_levin = (s, n) -> begin
            (s == -2 ? tail_policy_inf : tail_policy_hor) === true && _use_levin_tail(n; nmin = levin_nmin) || return false
            start_n = s == -2 ? tail_levin_infinity_start_n : tail_levin_horizon_start_n
            (s == -2 ? tail_levin_infinity_active : tail_levin_horizon_active)[] = true
            start_n[] === nothing && (start_n[] = n)
            return true
        end
        # one tracker per boundary: the two branches may run on different tasks
        # A replay with the same lmax (kmax raised) takes the walks of the earlier pass from its ledgers without solving
        # them again: their modes near the caps come from that pass's trackers, so that its lmax hits are kept.
        cap_trackers = rs !== nothing ? rs.cap_trackers :
            rl !== nothing && rl.lmax == lmax ? map(t -> CapTracker(lmax, kmax, deepcopy(t.l_walks), copy(t.m_shells),
                copy(t.k_shells)), rl.cap_trackers) : (CapTracker(lmax, kmax), CapTracker(lmax, kmax))
        failed_modes = rs === nothing ? [0, 0] : rs.failed_modes
        near_static_zero = rs === nothing ? [0, 0] : rs.near_static_zero
        # One mode from its parameters (max_flux, Levin choice, floors; levin_flux_scale is passed to the adaptive Levin
        # route, which does not use it), and its acceptance, which the walks apply in their order.
        gen_compute = (s, l, m, n, k, max_flux, use_levin, floor, levin_floor, levin_flux_scale, factors, arena) -> begin
            ω_mode = (m * F_orbit["ϒϕ"] + n * F_orbit["ϒr"] + k * F_orbit["ϒθ"]) / F_orbit["ϒt"]
            if _negligible_low_frequency_mode(s, l, ω_mode, F_orbit["ϒϕ"] / F_orbit["ϒt"], a, p, e)
                res = ConvolutionIntegrals._zero_radiative_mode(ω_mode, nothing; reason = "negligible_low_frequency")
                res["NearStaticOverflow"] = true
                res
            else
            let; ConvolutionIntegrals._with_coverage_retry(s, l, m, n, k; omega = ω_mode, Ωφ = F_orbit["ϒϕ"] / F_orbit["ϒt"], rmax = p / (1 - e), negligible_if_failed = _negligible_low_frequency_mode(s, l, ω_mode, F_orbit["ϒϕ"] / F_orbit["ϒt"], a, p, e; bound = ConvolutionIntegrals.NEGLIGIBLE_FAILED_MODE_BOUND)) do
                if use_levin
                    with_submission_mode(scope; mode = (s, l, m, n, k)) do mode_cache
                        mode_cache.walk_factors = factors
                        ConvolutionIntegrals.generic_mode_flux_from_master_cached_adaptive_levin!(mode_cache, KG_master, s, l, m, n, k; sample_tol = sample_tol, tol = tol, max_flux = levin_flux_scale, mode_abs_floor = levin_floor, threaded_sampling = threaded_sampling, confirm_low_flux = true, local_r_intervals = levin_local_n, local_theta_intervals = levin_local_n, max_depth = levin_max_depth, arena)
                    end
                else
                    with_submission_mode(scope; mode = (s, l, m, n, k)) do mode_cache
                        ConvolutionIntegrals.generic_mode_flux_from_master_cached!(mode_cache, KG_master, s, l, m, n, k; N0 = N0, K0 = K0, Nmax = Nmax, Kmax = Kmax, sample_tol = sample_tol, tol = tol, max_flux = max_flux, mode_abs_floor = floor, threaded_sampling = threaded_sampling, arena)
                    end
                end
            end; end
            end
        end
        gen_accept! = (s, l, m, n, k, mode) -> begin
            get(mode, "FailedMode", false) && (failed_modes[s == -2 ? 1 : 2] += 1)
            get(mode, "NearStaticOverflow", false) && (near_static_zero[s == -2 ? 1 : 2] += 1)
            ConvolutionIntegrals._check_finite_mode_flux(mode, s, l, m, n, k)
            _note_cap_mode!(cap_trackers[s == -2 ? 1 : 2], l, m, n, k, mode["EnergyFlux"])
            mode
        end
        # the Levin choice of the auto rule for (s, m, n, k)
        gen_levin_auto = (s, m, n, k) -> ((s == -2 ? tail_policy_inf : tail_policy_hor) === :auto && abs(n) >= levin_nmin &&
            _levin_auto(_mode_radial_trapezoid_n(KG_master, m, n, k, N0, Nmax), Nmax))
        gen_parameters = (s, l, m, n, k, max_flux, use_levin) -> begin
            auto_levin = use_generic_tail_levin(s, n)
            use_levin = use_levin || auto_levin || gen_levin_auto(s, m, n, k)
            floor = MODE_REFINE_FLOOR_FRACTION * branch_mode_floor(s)
            (Float64(max_flux), use_levin, floor, _tail_levin_mode_floor(floor, branch_levin_floor(s)),
                use_levin ? abs(max_flux) : 0.0)   # the n = 0 shell list is empty: levin_flux_scale = |max_flux|
        end
        # the serial evaluation of one mode (the k = 0 walks of the n = 0 shell); the other walks go through
        # _generic_l_walk with gen_compute_q
        eval_mode = (s, l, m, n, k, max_flux, use_levin=false) -> begin
            q = gen_parameters(s, l, m, n, k, max_flux, use_levin)
            gen_accept!(s, l, m, n, k, gen_compute(s, l, m, n, k, q..., nothing, nothing))
        end
        gen_compute_q = (s, l, m, n, k, q, factors, arena) -> gen_compute(s, l, m, n, k, q.max_flux, q.use_levin, q.floor, q.levin_floor, q.levin_flux_scale, factors, arena)

        record_mode = (s, l, m, n, k, mode) -> begin
            _record_current_mode!(record_h5, record, s == -2 ? "I" : "H", mode_index, mode; l, m, n, k, N_sample = N0, K_sample = K0)
            return nothing
        end

        # progress = true: one line per finished n shell of a branch
        t_start = rs === nothing ? time() : rs.t_start
        shells_done = rs === nothing ? Ref(0) : rs.shells_done
        log_shell = (branch, label, idx, E, L, Q, modes) -> begin
            progress || return nothing
            lock(_PROGRESS_LOCK) do
                shells_done[] += 1
                @printf("GSN shell (a, p, e, x) = (%s, %s, %s, %s) lmax=%d branch=%s %s=%d E=%.6e L=%.6e Q=%.6e modes=%d shells_done=%d elapsed=%.1fs\n",
                    a, p, e, x, lmax, branch, label, idx, E, L, Q, modes, shells_done[], time() - t_start)
                flush(stdout)
            end
            return nothing
        end
        # both boundaries (see _generic_branch_summation!); with mode_parallelism > 1 their l walks are computed ahead
        # and, without a record or progress log, the boundaries run on two tasks
        walk_run = (key, q; arena = ConvolutionIntegrals._series_arena()) -> _generic_l_walk(gen_compute_q, key[1], key[2], key[3], key[4], q, lmax, minimum_consecutive, x, mode_parallelism > 1;
            prior = _prior_walk(ledger_in[key[1] == -2 ? 1 : 2], key, lmax, lmax_prior), lmax_prior = lmax_prior, arena)
        prior_counts! = (s, p) -> begin
            i = s == -2 ? 1 : 2
            failed_modes[i] += p.failed
            near_static_zero[i] += p.near_static
            reused_modes[i] += Int(p.count)
        end
        branch_ctx = s -> (e = e, x = x, tol = tol, lmax = lmax, kmax = kmax, nmax = nmax, minimum_consecutive = minimum_consecutive,
            mode_abs_floor = mode_abs_floor, beta = beta, deep = deep, neg_branch_scale = neg_branch_scale,
            radial_over_azimuthal = radial_over_azimuthal, polar_over_azimuthal = polar_over_azimuthal,
            floor_ref = s == -2 ? branch_mode_floor_inf : branch_mode_floor_hor, levin_floor = branch_levin_floor,
            use_shell_levin = use_generic_tail_levin, levin_auto = gen_levin_auto, fit_estimate = fit_estimate, eval_mode = eval_mode, accept! = gen_accept!,
            record_mode = record_mode, log_shell = log_shell, run = let arena = ConvolutionIntegrals._series_arena(); (key, q) -> walk_run(key, q; arena) end,
            resume = rs === nothing ? nothing : s == -2 ? rs.infinity : rs.horizon, ledger_out = ledger_out[s == -2 ? 1 : 2],
            prior_counts! = prior_counts!)
        pf_inf = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_inf) : nothing
        pf_hor = mode_parallelism > 1 ? _walk_prefetch_resumed(walk_run, rs === nothing ? nothing : rs.pf_hor) : nothing
        inf, hor = _run_boundaries(() -> _generic_branch_summation!(-2, branch_ctx(-2), pf_inf),
            () -> _generic_branch_summation!(2, branch_ctx(2), pf_hor), mode_parallelism > 1 && !record)
        total_modes = inf.total_modes + hor.total_modes

        if record
            record_h5["meta/infinity_energy_flux"] = inf.energy_flux
            record_h5["meta/infinity_angular_momentum_flux"] = inf.angular_momentum_flux
            record_h5["meta/infinity_carter_constant_flux"] = inf.carter_constant_flux
            record_h5["meta/horizon_energy_flux"] = hor.energy_flux
            record_h5["meta/horizon_angular_momentum_flux"] = hor.angular_momentum_flux
            record_h5["meta/horizon_carter_constant_flux"] = hor.carter_constant_flux
            record_h5["meta/n_reached_inf"] = something(inf.n_reached, -1)
            record_h5["meta/n_reached_hor"] = something(hor.n_reached, -1)
            record_h5["meta/tail_levin_infinity_enabled"] = _tail_enabled(tail_policy_inf)
            record_h5["meta/tail_levin_horizon_enabled"] = _tail_enabled(tail_policy_hor)
            record_h5["meta/tail_levin_infinity_policy"] = string(tail_policy_inf)
            record_h5["meta/tail_levin_horizon_policy"] = string(tail_policy_hor)
            record_h5["meta/tail_levin_start_inf"] = something(tail_levin_infinity_start_n[], -1)
            record_h5["meta/tail_levin_start_hor"] = something(tail_levin_horizon_start_n[], -1)
            record_h5["meta/tail_levin_nmin"] = levin_nmin
            record_h5["meta/tail_levin_local_n"] = levin_local_n
            record_h5["meta/infinity_truncation_floor"] = inf.truncation_floor
            record_h5["meta/horizon_truncation_floor"] = hor.truncation_floor
            record_h5["meta/total_modes"] = total_modes
            record_h5["meta/infinity_energy_flux_list"] = inf.shell_list
            record_h5["meta/horizon_energy_flux_list"] = hor.shell_list
        end

        cap_hits_inf = _cap_hits(cap_trackers[1], tol, inf.energy_flux)
        cap_hits_hor = _cap_hits(cap_trackers[2], tol, hor.energy_flux)
        return (
            infinity_energy_flux = inf.energy_flux,
            infinity_angular_momentum_flux = inf.angular_momentum_flux,
            infinity_carter_constant_flux = inf.carter_constant_flux,
            horizon_energy_flux = hor.energy_flux,
            horizon_angular_momentum_flux = hor.angular_momentum_flux,
            horizon_carter_constant_flux = hor.carter_constant_flux,
            total_modes = total_modes,
            tolerance = tol,
            minimum_consecutive = minimum_consecutive,
            l_max = lmax,
            k_max = kmax,
            n_max = nmax,
            N_sample = N0,
            K_sample = K0,
            N_max = Nmax,
            K_max = Kmax,
            n_reached_inf = inf.n_reached,
            n_reached_hor = hor.n_reached,
            n_tail_infinity = inf.n_tail,
            n_tail_horizon = hor.n_tail,
            lmax_hit_inf = cap_hits_inf[1],
            lmax_hit_hor = cap_hits_hor[1],
            kmax_hit_inf = cap_hits_inf[2],
            kmax_hit_hor = cap_hits_hor[2],
            nmax_hit_inf = _nmax_hit(inf.n_reached, inf.shell_list, nmax, tol, inf.energy_flux),
            nmax_hit_hor = _nmax_hit(hor.n_reached, hor.shell_list, nmax, tol, hor.energy_flux),
            failed_modes_inf = failed_modes[1],
            failed_modes_hor = failed_modes[2],
            near_static_zero_inf = near_static_zero[1],
            near_static_zero_hor = near_static_zero[2],
            tail_levin_infinity_enabled = _tail_enabled(tail_policy_inf),
            tail_levin_horizon_enabled = _tail_enabled(tail_policy_hor),
            tail_levin_infinity_policy = tail_policy_inf,
            tail_levin_horizon_policy = tail_policy_hor,
            tail_levin_start_inf = tail_levin_infinity_start_n[],
            tail_levin_start_hor = tail_levin_horizon_start_n[],
            tail_levin_nmin = levin_nmin,
            tail_levin_local_n = levin_local_n,
            tail_levin_max_depth = levin_max_depth,
            infinity_truncation_floor = inf.truncation_floor,
            horizon_truncation_floor = hor.truncation_floor,
            mode_parallelism = mode_parallelism,
            modes_computed = pf_inf === nothing ? nothing : pf_inf.modes + pf_hor.modes,
            modes_computed_unused = pf_inf === nothing ? nothing : (pf_inf.modes - pf_inf.modes_used) + (pf_hor.modes - pf_hor.modes_used),
            # every n shell of each branch (summed over k, m, l with symmetry weights)
            infinity_shells = inf.shells,
            horizon_shells = hor.shells,
            infinity_energy_flux_list = inf.shell_list,
            horizon_energy_flux_list = hor.shell_list,
            record_file = record ? record_path : nothing,
            modes_new = total_modes - (rs !== nothing ? rs.total_modes : reused_modes[1] + reused_modes[2]),
            _resume = record ? nothing : (; kind = :generic, lmax = lmax, kmax = kmax, infinity = inf.state, horizon = hor.state,
                total_modes, ledgers = track ? ledger_out : nothing,
                branch_mode_floor_inf, branch_mode_floor_hor, tail_levin_infinity_active, tail_levin_horizon_active,
                tail_levin_infinity_start_n, tail_levin_horizon_start_n, cap_trackers, failed_modes, near_static_zero,
                t_start, shells_done, pf_inf, pf_hor),
        )
    finally
        pf_inf === nothing || _walk_discard!(pf_inf)
        pf_hor === nothing || _walk_discard!(pf_hor)
        record && close(record_h5)
    end
end

# Total generic-orbit fluxes, summed over n, k, m and l shells with the stopping rules described in
# Teukolsky_pointparticle_flux. Equatorial and e = 0 orbits are passed to the eccentric and inclined summations.
function generic_mode_summation(a, p, e, x; submission = nothing, stop_requested = nothing, options = nothing, kwargs...)
    o = _sum_options(options, submission, kwargs)
    caps = _auto_caps_start(a, p, e, x, o.tol, o.lmax, o.kmax, o.nmax)
    scope, owned = ensure_public_submission(submission, a, p, e, x; Nmax = o.Nmax, Kmax = o.Kmax, stop_requested)
    return _with_public_scope(scope, owned) do
        _with_auto_caps(caps) do L, Kc, Nc, resume
            with_submission_operation(scope) do
                _generic_mode_summation_scoped(scope, a, p, e, x, o; lmax = L, kmax = Kc, nmax = Nc, resume)
            end
        end
    end
end

end
