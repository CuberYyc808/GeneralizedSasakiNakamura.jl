const _SCHWARZSCHILD_SEEDS = Dict{NTuple{3,Int},ComplexF64}(
    (-2, 2, 0) => 0.37367168441804185 - 0.08896231568893566im,
    (-2, 2, 1) => 0.34671099687916350 - 0.27391487529123500im,
    (-2, 2, 2) => 0.30105345461236650 - 0.47827698322307200im,
    (-2, 2, 3) => 0.25150496218564372 - 0.70514820243348242im,
    (-2, 2, 4) => 0.20751457981301125 - 0.94684489086636203im,
    (-2, 2, 5) => 0.16929940309302113 - 1.19560805413578164im,
    (-2, 2, 6) => 0.13325234024514371 - 1.44791062616211419im,
    (-2, 2, 7) => 0.09282233367019177 - 1.70384117220627562im,
    # The conventional n=8 algebraically-special endpoint requires separate
    # sheet-aware handling. These are the next regular Schwarzschild roots.
    (-2, 2, 9) => 0.063263505125603553 - 2.3026447651585391im,
    (-2, 2, 10) => 0.07655346288598433 - 2.5608266173815060im,
    (-2, 3, 0) => 0.59944328843749000 - 0.09270304794494700im,
    (-2, 3, 1) => 0.58264380303330066 - 0.28129811343504524im,
    (-2, 3, 2) => 0.55168490077844956 - 0.47909275096696286im,
    (-2, 4, 0) => 0.80917837753223900 - 0.09416396098892300im,
    (-2, 4, 1) => 0.79663153203450299 - 0.28433434940484159im,
    (-2, 4, 2) => 0.77270953260670861 - 0.47990817512116291im,
)

const _SCHWARZSCHILD_L2_N8_UNCONVENTIONAL =
    -0.0153245047607925 - 1.99841184185869im
const _KERR_L2_M2_NEAR_AS_ANCHOR_SPIN = 0.05
const _KERR_L2_M2_MST_SWITCH_SPIN = 0.015
const _KERR_L2_M2_NEAR_AS_ANCHORS = Dict{Symbol,ComplexF64}(
    :unconventional => 0.06177258790671765 - 1.8876918643227598im,
    :algebraically_special =>
        0.060299316438925596 - 2.0888402014191692im,
)

# Frozen qnm 0.4.4 (commit f3abd18) branch anchors for the counterrotating
# (s,l,|m|)=(-2,2,2) family.  They select the physical overtone branch only;
# angular data and the radial continued fraction are recomputed at the target.
const _COUNTERROTATING_NEAR_EXTREMAL_ANCHORS = Dict(
    (0.9999, 4) => 0.10966391100861249 - 1.0484565784875364im,
    (0.9999, 5) => 0.09622818654671349 - 1.3188614163368266im,
    (0.9999, 6) => 0.08552024051137216 - 1.5859422845532654im,
    (0.9999, 7) => 0.07630883605751265 - 1.8541542337264556im,
    (0.9999, 9) => -0.13788097978067113 - 2.2230115092378702im,
    (0.9999, 10) => 0.08172665916503848 - 2.674770803913548im,
    (0.99999, 4) => 0.10967190514278498 - 1.0484624076661069im,
    (0.99999, 5) => 0.09620454978969084 - 1.3188953723824268im,
    (0.99999, 6) => 0.08546785861495902 - 1.5860447304425451im,
    (0.99999, 7) => 0.07632391674711030 - 1.8542317885405424im,
    (0.99999, 9) => -0.14229956759043466 - 2.2191402886345815im,
    (0.99999, 10) => 0.08165609544784828 - 2.6748151109278746im,
)
const _COUNTERROTATING_ANCHOR_SPINS = (0.9999, 0.99999)

const _GENERATED_SCHWARZSCHILD_SEEDS = Dict{NTuple{3,Int},ComplexF64}()
const _GENERATED_SCHWARZSCHILD_SEED_METADATA = Dict{NTuple{3,Int},NamedTuple}()
const _SCHWARZSCHILD_SEED_LOCK = ReentrantLock()

function _working_type(a, guess)
    if a isa BigFloat || guess isa Complex{BigFloat} || guess isa BigFloat
        return BigFloat
    end
    return Float64
end

function _complex_of_type(::Type{T}, value) where {T<:AbstractFloat}
    return Complex{T}(T(real(value)), T(imag(value)))
end

function _branch_consistent(mode::QNMMode, omega)
    if mode.branch == :positive_real && mode.s == -2 && mode.l == 2 &&
            abs(mode.m) == 2 && mode.n in (8, 9)
        # These complete-spectrum trajectories cross the NIA. Their label is
        # fixed by continuation from the finite-spin anchor, not by sign(Re w).
        return true
    end
    if mode.branch == :positive_real
        return real(omega) > 0
    end
    return real(omega) < 0
end

function _schwarzschild_label_contract(mode::QNMMode, convention::Symbol)
    # Negative public spins are evaluated through (a,m) -> (-a,-m), so the
    # same physical near-AS branch appears internally with m=-2.
    l2m2 = mode.s == -2 && mode.l == 2 && abs(mode.m) == 2
    standard_as = convention == :overtone && l2m2 && mode.n == 8
    complete_unconventional = convention == :complete_spectrum && l2m2 &&
        mode.n == 8
    complete_as = convention == :complete_spectrum && l2m2 && mode.n == 9
    near_as_role = standard_as || complete_as ? :algebraically_special :
        complete_unconventional ? :unconventional : :none
    is_algebraically_special_pair = near_as_role != :none
    endpoint_kind = near_as_role == :none ? :regular : near_as_role
    sequence_index = convention == :leaver ? missing :
        convention == :complete_spectrum && l2m2 && mode.n >= 10 ?
            mode.n - 1 :
        complete_as ? missing : mode.n
    branch_label = convention == :leaver ? :unassigned_leaver_root :
        near_as_role == :unconventional ? :kerr_branch_8_0 :
        near_as_role == :algebraically_special ? :kerr_branch_8_1 :
        Symbol(:kerr_overtone_n, mode.n)
    return (
        label_convention=convention == :overtone ? :standard_overtone :
            convention == :complete_spectrum ?
                :chen_complete_spectrum : :leaver_inversion,
        branch_label,
        endpoint_kind,
        near_as_role,
        algebraically_special_pair=is_algebraically_special_pair,
        schwarzschild_sequence_index=sequence_index,
        schwarzschild_multiplicity=1,
    )
end

function _schwarzschild_algebraically_special_frequency(
        mode::QNMMode, ::Type{T}) where {T<:AbstractFloat}
    coefficient = T((mode.l - 1) * mode.l * (mode.l + 1) *
        (mode.l + 2)) / T(12)
    return complex(zero(T), -coefficient)
end

function _algebraically_special_endpoint_result(mode::QNMMode, a::T;
        angular_order::Int, sheet_id::Symbol,
        requested_inversion_index,
        convention::Symbol=:overtone) where {T<:AbstractFloat}
    omega = _schwarzschild_algebraically_special_frequency(mode, T)
    angular = angular_branch(mode, a, omega;
        truncation_order=angular_order, sheet_id)
    nan_complex = complex(T(NaN), T(NaN))
    contract = _schwarzschild_label_contract(mode, convention)
    return LeaverResult(
        mode, convention, mode.n, missing, a, omega,
        angular.angular_A, angular.lambda, angular.mixing,
        nan_complex, T(NaN), 0, zero(T), T(angular.residual),
        _precision_bits(T), :accepted, :exact_algebraically_special_endpoint,
        (
            input_type=string(typeof(a)),
            working_type=string(T),
            guess_source=:exact_algebraically_special_formula,
            label_convention=contract.label_convention,
            branch_label=contract.branch_label,
            endpoint_kind=contract.endpoint_kind,
            schwarzschild_sequence_index=missing,
            schwarzschild_multiplicity=1,
            endpoint_formula=
                :minus_i_lminus1_l_lplus1_lplus2_over_12,
            endpoint_frequency=omega,
            root_equation=:exact_algebraically_special_formula,
            radial_cf_applicable=false,
            requested_inversion_index,
            inversion_index=missing,
            angular_route=angular.metadata.route,
            angular_sheet=angular.sheet_id,
            angular_order=angular.truncation_order,
            root_drift=zero(T),
            family=:schwarzschild_algebraically_special,
            matsubara_index=8,
            matsubara_frequency=omega,
            pole_skipping=true,
            qnm_criterion=:incident_amplitude_pole_zero_collision,
            gsn_amplitude_limit_implemented=false,
            claim_boundary=
                :exact_frequency_and_pole_skipping_endpoint_not_simple_gsn_residue,
        ),
    )
end

function _unconventional_endpoint_result(mode::QNMMode, a::T;
        angular_order::Int, sheet_id::Symbol,
        requested_inversion_index,
        convention::Symbol=:complete_spectrum) where {T<:AbstractFloat}
    omega = _complex_of_type(T, _SCHWARZSCHILD_L2_N8_UNCONVENTIONAL)
    angular = angular_branch(mode, a, omega;
        truncation_order=angular_order, sheet_id)
    nan_complex = complex(T(NaN), T(NaN))
    contract = _schwarzschild_label_contract(mode, convention)
    return LeaverResult(
        mode, convention, mode.n, missing, a, omega,
        angular.angular_A, angular.lambda, angular.mixing,
        nan_complex, T(NaN), 0, zero(T), T(angular.residual),
        _precision_bits(T), :accepted, :frozen_unconventional_endpoint,
        (
            input_type=string(typeof(a)),
            working_type=string(T),
            guess_source=:complete_spectrum_unconventional_endpoint,
            label_convention=contract.label_convention,
            branch_label=contract.branch_label,
            endpoint_kind=contract.endpoint_kind,
            schwarzschild_sequence_index=8,
            schwarzschild_multiplicity=1,
            endpoint_frequency=omega,
            root_equation=:unconventional_sheet_gsn_incidence,
            radial_cf_applicable=false,
            requested_inversion_index,
            inversion_index=missing,
            angular_route=angular.metadata.route,
            angular_sheet=angular.sheet_id,
            angular_order=angular.truncation_order,
            root_drift=zero(T),
            family=:schwarzschild_unconventional,
            source_reference=:chen_et_al_complete_qnm_spectrum_2025,
            claim_boundary=:frequency_endpoint_with_separate_mst_validation,
        ),
    )
end

function _resolve_continuation_coordinate(value, a, mode::QNMMode)
    coordinate = Symbol(lowercase(String(value)))
    if coordinate == :auto
        low_overtone_zdm = mode.n <= 4 &&
            _near_extremal_zdm_seed_scope(mode, a)
        # A nu path is efficient for a single near-horizon ZDM endpoint, but
        # it does not by itself preserve the overtone label once damped and
        # zero-damping branches crowd together. Track n >= 5 from the
        # Schwarzschild sequence in spin unless the caller explicitly asks
        # for a nu continuation.
        return low_overtone_zdm ? :nu : :spin
    end
    coordinate in (:spin, :nu) || throw(ArgumentError(
        "continuation_coordinate must be :auto, :spin, or :nu."))
    return coordinate
end

function _near_extremal_family(mode::QNMMode, a, omega)
    abs(a) >= oftype(a, 0.99) || return :unclassified
    kappa = sqrt(max(zero(a), one(a) - a^2))
    rplus = one(a) + kappa
    horizon_frequency = a / (2rplus)
    detuning = abs(omega - mode.m * horizon_frequency)
    damping = abs(imag(omega))
    near_horizon_scale = max(oftype(a, 50) * kappa,
        oftype(a, 256) * eps(typeof(a)))
    if mode.m * a > zero(a) &&
            detuning <= near_horizon_scale &&
            damping <= near_horizon_scale
        return :ZDM
    elseif damping > oftype(a, 0.05) &&
            max(detuning, damping) > near_horizon_scale
        return :DM
    end
    return :unclassified
end

function _extremal_zdm_endpoint_scope(mode::QNMMode, a)
    # Equatorial corotating families have the unambiguous synchronous ZDM
    # endpoint. Other families can contain both zero-damping and damped
    # branches, so their endpoint is classified by continuation instead.
    abs(mode.m) == mode.l || return false
    synchronous_frequency = mode.m * a / 2
    iszero(synchronous_frequency) && return false
    return mode.branch == :positive_real ?
        real(synchronous_frequency) > 0 : real(synchronous_frequency) < 0
end

function _near_extremal_zdm_seed_scope(mode::QNMMode, a)
    abs(a) >= oftype(a, 0.999) || return false
    abs(mode.m) == mode.l || return false
    return _extremal_zdm_endpoint_scope(mode, sign(a))
end

function _near_extremal_zdm_seed(mode::QNMMode, a::T,
        inversion_index::Int) where {T<:AbstractFloat}
    horizon_gap = sqrt(max(zero(T), one(T) - a^2))
    rplus = one(T) + horizon_gap
    horizon_frequency = a / (2rplus)
    surface_gravity = horizon_gap / (2rplus)
    omega = complex(mode.m * horizon_frequency,
        -surface_gravity * (T(mode.n) + T(0.5)))
    return (
        seed=omega,
        seed_spin=a,
        source=:near_extremal_zdm_asymptotic_seed,
        seed_inversion_index=inversion_index,
        seed_predictor=:synchronous_frequency_surface_gravity,
        seed_duplicate_distance=T(Inf),
    )
end

function _extremal_zdm_endpoint_result(mode::QNMMode, a::T;
        angular_order::Int, sheet_id::Symbol, convention::Symbol,
        requested_inversion_index) where {T<:AbstractFloat}
    omega = complex(T(mode.m) * a / 2, zero(T))
    angular = angular_branch(mode, a, omega;
        truncation_order=angular_order, sheet_id)
    contract = _schwarzschild_label_contract(mode, convention)
    nan_complex = complex(T(NaN), T(NaN))
    return LeaverResult(
        mode, convention, convention == :leaver ? missing : mode.n,
        missing, a, omega, angular.angular_A, angular.lambda,
        angular.mixing, nan_complex, T(NaN), 0, zero(T),
        T(angular.residual), _precision_bits(T), :accepted,
        :exact_extremal_zdm_accumulation_endpoint,
        (
            input_type=string(typeof(a)),
            working_type=string(T),
            label_convention=contract.label_convention,
            branch_label=contract.branch_label,
            endpoint_kind=contract.endpoint_kind,
            extremal_endpoint=true,
            endpoint_method=:analytic_synchronous_frequency,
            spectrum_object=:zdm_branch_point_accumulation,
            simple_pole=false,
            coalesced_overtone_labels=true,
            requested_overtone_label=mode.n,
            distinct_overtone_frequency=false,
            radial_observables=:not_applicable_to_isolated_simple_pole,
            gsn_amplitude_limit_implemented=false,
            horizon_frequency=a / 2,
            synchronous_frequency=omega,
            requested_inversion_index,
            inversion_index=missing,
            angular_route=angular.metadata.route,
            angular_sheet=angular.sheet_id,
            angular_order=angular.truncation_order,
            root_equation=:analytic_extremal_synchronous_limit,
            family=:ZDM,
            claim_boundary=
                :frequency_endpoint_and_branch_point_not_simple_pole_residue,
        ),
    )
end

function _spin_sequence_anchor(mode::QNMMode, ::Type{T}) where {
        T<:AbstractFloat}
    if mode.m < 0 && mode.n >= 5
        # Counterrotating damped branches remain separated from the ZDM
        # accumulation but become poorly conditioned close to extremality.
        # Continue to the last supported finite-spin guard instead of making
        # one cubic jump from a=0.9995.
        return T(0.99999)
    end
    if mode.s == -2 && mode.l == 2 && abs(mode.m) == 2 && mode.n >= 7
        return T(0.99)
    end
    return T(0.9995)
end

function _default_cf_maximum_iterations(mode::QNMMode, a=nothing,
        guess=nothing)
    # This is a ceiling, not a forced iteration count. Near-extremal
    # retrograde modes can require the same depth as high overtones.
    baseline = mode.n >= 8 ? 48000 : 12000
    high_damping_near_extremal = a !== nothing && guess !== nothing &&
        abs(a) >= 0.99 && abs(imag(guess)) >= 0.3
    high_damping_near_extremal || return baseline
    ceiling = mode.branch == :negative_real ? 240000 : 120000
    return max(baseline, ceiling)
end

function _root_derivative_step(::Type{Float64}, omega)
    return eps(Float64)^(1 / 5) * max(1.0, abs(omega))
end

function _root_derivative_step(::Type{BigFloat}, omega)
    return eps(BigFloat)^(BigFloat(1) / 5) *
        max(one(BigFloat), abs(omega))
end

function _near_extremal_derivative_scale(mode::QNMMode, a::T) where {
        T<:AbstractFloat}
    abs(a) >= T(0.99) && mode.n >= 5 || return one(T)
    kappa = sqrt(max(zero(T), one(T) - a^2))
    return clamp(T(2) * kappa, T(0.002), one(T))
end

function _conditioned_root_residual_limit(mode::QNMMode, a::T,
        root_tolerance::T) where {T<:AbstractFloat}
    abs(a) >= T(0.999) && mode.n >= 5 || return root_tolerance
    kappa = sqrt(max(zero(T), one(T) - a^2))
    conditioning = clamp(T(0.01) / max(kappa, sqrt(eps(T))), one(T), T(10))
    return max(root_tolerance, T(1.0e-8) * conditioning)
end

function _conditioned_root_stability_limit(mode::QNMMode, a::T,
        stability_tolerance::T) where {T<:AbstractFloat}
    abs(a) >= T(0.999) && mode.n >= 5 || return stability_tolerance
    return max(stability_tolerance, T(1.0e-6))
end

function _counterrotating_near_extremal_seed(
        mode::QNMMode, a::T) where {T<:AbstractFloat}
    mode.s == -2 && mode.l == 2 && abs(mode.m) == 2 || return nothing
    mode.branch == :positive_real || return nothing
    mode.m * a < zero(T) || return nothing
    mode.n in (4, 5, 6, 7, 9, 10) || return nothing
    abs(a) >= T(first(_COUNTERROTATING_ANCHOR_SPINS)) || return nothing

    spin = abs(a)
    spin1, spin2 = T.(_COUNTERROTATING_ANCHOR_SPINS)
    omega1 = _complex_of_type(T,
        _COUNTERROTATING_NEAR_EXTREMAL_ANCHORS[(Float64(spin1), mode.n)])
    omega2 = _complex_of_type(T,
        _COUNTERROTATING_NEAR_EXTREMAL_ANCHORS[(Float64(spin2), mode.n)])
    if spin == spin1
        return (seed=omega1, source=:frozen_qnm_0p4p4_exact_anchor,
            anchor_spins=(spin1, spin2), interpolation_weight=zero(T))
    elseif spin == spin2
        return (seed=omega2, source=:frozen_qnm_0p4p4_exact_anchor,
            anchor_spins=(spin1, spin2), interpolation_weight=one(T))
    end

    kappa = sqrt(max(zero(T), one(T) - spin^2))
    kappa1 = sqrt(max(zero(T), one(T) - spin1^2))
    kappa2 = sqrt(max(zero(T), one(T) - spin2^2))
    weight = (kappa - kappa1) / (kappa2 - kappa1)
    return (
        seed=omega1 + weight * (omega2 - omega1),
        source=:frozen_qnm_0p4p4_kappa_predictor,
        anchor_spins=(spin1, spin2),
        interpolation_weight=weight,
    )
end

function _counterrotating_conditioned_result(
        mode::QNMMode, a::T, seed;
        convention::Symbol, overtone_index,
        inversion_index::Int, angular_order::Int, sheet_id::Symbol,
        cf_tolerance::T, cf_minimum_iterations::Int,
        cf_maximum_iterations::Int, root_tolerance::T,
        stability_tolerance::T) where {T<:AbstractFloat}
    primary = leaver_radial_residual(mode, a, seed.seed;
        angular_order, sheet_id, inversion_index, cf_tolerance,
        cf_minimum_iterations, cf_maximum_iterations,
        require_cf_convergence=false)
    guard = leaver_radial_residual(mode, a, seed.seed;
        angular_order=angular_order + 8,
        sheet_id=Symbol(sheet_id, :_counterrotating_guard),
        inversion_index, cf_tolerance,
        cf_minimum_iterations=cf_minimum_iterations + 100,
        cf_maximum_iterations, require_cf_convergence=false)
    root_residual = T(abs(guard.value))
    angular_drift = T(abs(guard.angular.angular_A -
        primary.angular.angular_A) / max(one(T),
        abs(guard.angular.angular_A), abs(primary.angular.angular_A)))
    strict = root_residual <= root_tolerance &&
        guard.error <= cf_tolerance && guard.cf_converged &&
        angular_drift <= stability_tolerance
    conditioned_limit = max(
        _conditioned_root_residual_limit(mode, a, root_tolerance), T(1.0e-6))
    conditioned = !strict && root_residual <= conditioned_limit &&
        guard.error <= max(cf_tolerance, T(1.0e-10)) &&
        isfinite(guard.error) && angular_drift <= stability_tolerance
    status = strict ? :accepted : conditioned ? :estimated : :failed
    stop_reason = strict ? :accepted : conditioned ?
        :conditioned_near_extremal_reference_seed :
        root_residual > conditioned_limit ? :reference_seed_residual_gate :
        angular_drift > stability_tolerance ? :angular_stability_gate :
        :lentz_error_gate
    contract = _schwarzschild_label_contract(mode, convention)
    return LeaverResult(
        mode, convention, overtone_index, inversion_index, a, seed.seed,
        guard.angular.angular_A, guard.angular.lambda, guard.angular.mixing,
        guard.value, T(guard.error), guard.iterations, root_residual,
        T(guard.angular.residual), _precision_bits(T), status, stop_reason,
        (
            input_type=string(typeof(a)),
            working_type=string(T),
            guess_source=seed.source,
            reference_role=:branch_selection_only,
            target_recomputation=:angular_and_leaver_cf,
            upstream_qnm_version="0.4.4",
            upstream_qnm_commit=
                "f3abd18e59828e7e7d75d07f20c7cbc87925edfa",
            anchor_spins=seed.anchor_spins,
            interpolation_weight=seed.interpolation_weight,
            convention,
            overtone_index,
            inversion_index,
            label_convention=contract.label_convention,
            branch_label=contract.branch_label,
            endpoint_kind=contract.endpoint_kind,
            near_as_role=contract.near_as_role,
            schwarzschild_sequence_index=contract.schwarzschild_sequence_index,
            schwarzschild_multiplicity=contract.schwarzschild_multiplicity,
            family=_near_extremal_family(mode, a, seed.seed),
            continuation_coordinate=:frozen_kappa_predictor,
            angular_route=guard.angular.metadata.route,
            angular_sheet=guard.angular.sheet_id,
            angular_order=guard.angular.truncation_order,
            angular_order_drift=angular_drift,
            cf_minimum_iterations=cf_minimum_iterations + 100,
            cf_maximum_iterations,
            cf_tolerance,
            root_tolerance,
            conditioned_root_limit=conditioned_limit,
            conditioned_root=conditioned,
            stability_tolerance,
            root_drift=zero(T),
            root_equation=:leaver_continued_fraction,
            acceptance_basis=strict ? :direct_target_cf :
                :conditioned_reference_branch_target_cf,
            cf_tail_gate_passed=guard.error <= cf_tolerance,
            radial_cf_applicable=true,
            scientific_acceptance=status == :accepted,
        ),
    )
end

function _polynomial_predictor(xs, ys, x)
    length(xs) == length(ys) || throw(ArgumentError(
        "Continuation abscissas and values must have equal lengths."))
    isempty(xs) && throw(ArgumentError(
        "The continuation predictor requires at least one point."))
    value = zero(first(ys))
    for index in eachindex(xs)
        weight = one(x)
        for other in eachindex(xs)
            index == other && continue
            weight *= (x - xs[other]) / (xs[index] - xs[other])
        end
        value += weight * ys[index]
    end
    return value
end

function _continuation_curvature(spins, roots)
    length(spins) >= 3 || return zero(real(first(roots)))
    x1, x2, x3 = spins[(end - 2):end]
    y1, y2, y3 = roots[(end - 2):end]
    slope12 = (y2 - y1) / (x2 - x1)
    slope23 = (y3 - y2) / (x3 - x2)
    return 2abs((slope23 - slope12) / (x3 - x1))
end

function _continuation_predictor(spins, roots, next_spin)
    length(spins) >= 3 && return _polynomial_predictor(
        spins[(end - 2):end], roots[(end - 2):end], next_spin)
    if length(spins) == 2
        slope = (roots[end] - roots[end - 1]) /
            (spins[end] - spins[end - 1])
        return roots[end] + slope * (next_spin - spins[end])
    end
    return roots[end]
end


function _terminal_extrapolation_predictor(spins, roots, target_spin)
    count = length(spins)
    count >= 4 || return _polynomial_predictor(
        spins, roots, target_spin)
    T = typeof(first(spins))
    matrix = zeros(T, count, count)
    rhs = zeros(eltype(roots), count)
    steps = diff(spins)

    matrix[1, 1] = -steps[2]
    matrix[1, 2] = steps[1] + steps[2]
    matrix[1, 3] = -steps[1]
    for index in 2:(count - 1)
        left_step = steps[index - 1]
        right_step = steps[index]
        matrix[index, index - 1] = left_step
        matrix[index, index] = 2(left_step + right_step)
        matrix[index, index + 1] = right_step
        rhs[index] = 6((roots[index + 1] - roots[index]) / right_step -
            (roots[index] - roots[index - 1]) / left_step)
    end
    previous_step = steps[end - 1]
    last_step = steps[end]
    matrix[end, end - 2] = last_step
    matrix[end, end - 1] = -(last_step + previous_step)
    matrix[end, end] = previous_step
    second_derivatives = matrix \ rhs

    left = spins[end - 1]
    right = spins[end]
    width = right - left
    left_weight = (right - target_spin) / width
    right_weight = (target_spin - left) / width
    return second_derivatives[end - 1] *
        (right - target_spin)^3 / (6width) +
        second_derivatives[end] *
        (target_spin - left)^3 / (6width) +
        (roots[end - 1] - second_derivatives[end - 1] * width^2 / 6) *
        left_weight +
        (roots[end] - second_derivatives[end] * width^2 / 6) *
        right_weight
end

function _continuation_branch_guard(spins, roots, next_spin, predictor,
        candidate, root_tolerance)
    T = typeof(real(candidate))
    length(spins) < 2 && return (accepted=true, discrepancy=zero(T),
        allowance=T(Inf))
    previous_step = abs(spins[end] - spins[end - 1])
    step_ratio = abs(next_spin - spins[end]) / previous_step
    history_move = abs(roots[end] - roots[end - 1]) * step_ratio
    predicted_move = abs(predictor - roots[end])
    scale = max(one(T), abs(candidate), abs(predictor))
    allowance = max(
        T(32) * sqrt(T(root_tolerance)) * scale,
        T(0.2) * max(history_move, predicted_move),
    )
    discrepancy = abs(candidate - predictor)
    return (accepted=discrepancy <= allowance,
        discrepancy=discrepancy, allowance=allowance)
end

function _polish_leaver_root(mode::QNMMode, a::T, initial::Complex{T};
        angular_order::Int,
        sheet_id::Symbol,
        inversion_index::Int=mode.n,
        excluded_roots=Complex{T}[],
        cf_tolerance,
        cf_minimum_iterations::Int,
        cf_maximum_iterations::Int,
        root_tolerance::T,
        maximum_root_iterations::Int,
        trust_radius::T,
        derivative_step_scale::T=one(T)) where {T<:AbstractFloat}
    omega = initial
    evaluation = leaver_radial_residual(mode, a, omega;
        angular_order, sheet_id, inversion_index, cf_tolerance,
        cf_minimum_iterations, cf_maximum_iterations,
        require_cf_convergence=false)
    history = NamedTuple[]

    function deflated_value(value, frequency)
        isempty(excluded_roots) && return value
        separation = minimum(abs(frequency - root) for root in excluded_roots)
        scale = max(one(T), abs(frequency),
            maximum(abs, excluded_roots))
        separation <= T(64) * eps(T) * scale &&
            return complex(T(Inf), zero(T))
        denominator = prod(frequency - root for root in excluded_roots)
        return value / denominator
    end

    for iteration in 1:maximum_root_iterations
        residual = T(abs(evaluation.value))
        deflated_residual = T(abs(deflated_value(evaluation.value, omega)))
        push!(history, (
            iteration=iteration,
            omega=omega,
            residual=residual,
            deflated_residual=deflated_residual,
            cf_error=evaluation.error,
            cf_iterations=evaluation.iterations,
            angular_residual=evaluation.angular.residual,
        ))
        residual <= root_tolerance && return (
            omega=omega, evaluation=evaluation, history=history,
            converged=true, conditioned=false,
            stop_reason=:root_tolerance)

        h = T(_root_derivative_step(T, omega)) * derivative_step_scale
        xplus = leaver_radial_residual(mode, a, omega + h;
            angular_order, sheet_id, inversion_index, cf_tolerance,
            cf_minimum_iterations, cf_maximum_iterations,
            require_cf_convergence=false)
        xminus = leaver_radial_residual(mode, a, omega - h;
            angular_order, sheet_id, inversion_index, cf_tolerance,
            cf_minimum_iterations, cf_maximum_iterations,
            require_cf_convergence=false)
        xplus2 = leaver_radial_residual(mode, a, omega + 2h;
            angular_order, sheet_id, inversion_index, cf_tolerance,
            cf_minimum_iterations, cf_maximum_iterations,
            require_cf_convergence=false)
        xminus2 = leaver_radial_residual(mode, a, omega - 2h;
            angular_order, sheet_id, inversion_index, cf_tolerance,
            cf_minimum_iterations, cf_maximum_iterations,
            require_cf_convergence=false)
        derivative = (-deflated_value(xplus2.value, omega + 2h) +
            8 * deflated_value(xplus.value, omega + h) -
            8 * deflated_value(xminus.value, omega - h) +
            deflated_value(xminus2.value, omega - 2h)) / (12h)
        if !isfinite(derivative) || abs(derivative) <= sqrt(eps(T))
            return (
                omega=omega, evaluation=evaluation, history=history,
                converged=false, conditioned=false,
                stop_reason=:singular_root_jacobian)
        end
        delta = -deflated_value(evaluation.value, omega) / derivative
        if abs(delta) > trust_radius
            delta *= trust_radius / abs(delta)
        end

        accepted = false
        for backtrack in 0:10
            scale = T(2)^(-backtrack)
            candidate = omega + scale * delta
            _branch_consistent(mode, candidate) || continue
            candidate_evaluation = leaver_radial_residual(
                mode, a, candidate;
                angular_order, sheet_id, inversion_index, cf_tolerance,
                cf_minimum_iterations, cf_maximum_iterations,
                require_cf_convergence=false)
            if abs(deflated_value(candidate_evaluation.value, candidate)) <
                    abs(deflated_value(evaluation.value, omega))
                omega = candidate
                evaluation = candidate_evaluation
                accepted = true
                break
            end
        end
        if !accepted
            conditioned_limit = _conditioned_root_residual_limit(
                mode, a, root_tolerance)
            conditioned = residual <= conditioned_limit &&
                evaluation.cf_converged && evaluation.error <= cf_tolerance
            return (
                omega=omega, evaluation=evaluation, history=history,
                converged=conditioned, conditioned,
                stop_reason=conditioned ? :conditioned_cf_floor :
                    :root_line_search_stagnation)
        end
    end
    final_residual = T(abs(evaluation.value))
    conditioned_limit = _conditioned_root_residual_limit(
        mode, a, root_tolerance)
    strict = final_residual <= root_tolerance
    conditioned = !strict && final_residual <= conditioned_limit &&
        evaluation.cf_converged && evaluation.error <= cf_tolerance
    return (
        omega=omega, evaluation=evaluation, history=history,
        converged=strict || conditioned, conditioned,
        stop_reason=strict ? :root_tolerance : conditioned ?
            :conditioned_cf_floor : :root_iteration_limit)
end

_near_as_complete_scope(mode::QNMMode, near_as_role::Symbol) =
    mode.s == -2 && mode.l == 2 && abs(mode.m) == 2 &&
    near_as_role in (:unconventional, :algebraically_special) &&
    mode.branch == :positive_real

function _near_as_mst_required(mode::QNMMode, a, ::Type{T},
        near_as_role::Symbol) where {T<:AbstractFloat}
    return T === Float64 && _near_as_complete_scope(mode, near_as_role) &&
        !iszero(a) && abs(a) < T(_KERR_L2_M2_MST_SWITCH_SPIN)
end

function _near_as_mst_finalization_required(
        mode::QNMMode, a, ::Type{T}, near_as_role::Symbol) where {
        T<:AbstractFloat}
    return T === Float64 &&
        _near_as_complete_scope(mode, near_as_role) &&
        near_as_role == :algebraically_special &&
        abs(a) >= T(_KERR_L2_M2_MST_SWITCH_SPIN) &&
        abs(a) < T(_KERR_L2_M2_NEAR_AS_ANCHOR_SPIN)
end

function _mst_incidence_evaluation(mode::QNMMode, a::Float64,
        omega::ComplexF64; angular_order::Int, sheet_id::Symbol)
    route = ISEM.DirectGSN.direct_complex_route(
        mode.s, mode.l, mode.m, a, omega, :IN; backend=:mst)
    scale = max(1.0, abs(route.transmission), abs(route.incidence),
        abs(route.reflection))
    metadata = route.metadata
    representation_error = maximum((
        abs(metadata.mst_amplitude_truncation),
        abs(metadata.mst_amplitude_spread),
        abs(metadata.mst_amplitude_nearest),
        abs(metadata.mst_nu_residual),
        abs(metadata.mst_anchor_error),
        abs(metadata.mst_anchor_residual),
    ))
    angular = angular_branch(mode, a, omega;
        truncation_order=angular_order, sheet_id)
    return (
        value=route.incidence,
        scaled_residual=abs(route.incidence) / scale,
        error=Float64(representation_error),
        iterations=0,
        angular,
        route,
        scale,
        root_equation=:gsn_mst_incidence,
    )
end

function _polish_mst_incidence_root(mode::QNMMode, a::Float64,
        initial::ComplexF64; angular_order::Int, sheet_id::Symbol,
        root_tolerance::Float64, maximum_root_iterations::Int,
        trust_radius::Float64, derivative_step_scale::Float64=1.0)
    omega = initial
    evaluation = _mst_incidence_evaluation(
        mode, a, omega; angular_order, sheet_id)
    history = NamedTuple[]
    for iteration in 1:maximum_root_iterations
        residual = evaluation.scaled_residual
        push!(history, (
            iteration,
            omega,
            residual,
            deflated_residual=residual,
            cf_error=evaluation.error,
            cf_iterations=0,
            angular_residual=evaluation.angular.residual,
            root_equation=:gsn_mst_incidence,
        ))
        residual <= root_tolerance && return (
            omega, evaluation, history, converged=true,
            stop_reason=:root_tolerance)

        h = derivative_step_scale * 2.0e-5 * max(1.0, abs(omega))
        minus2 = _mst_incidence_evaluation(
            mode, a, omega - 2h; angular_order, sheet_id)
        minus1 = _mst_incidence_evaluation(
            mode, a, omega - h; angular_order, sheet_id)
        plus1 = _mst_incidence_evaluation(
            mode, a, omega + h; angular_order, sheet_id)
        plus2 = _mst_incidence_evaluation(
            mode, a, omega + 2h; angular_order, sheet_id)
        derivative = (minus2.value - 8minus1.value + 8plus1.value -
            plus2.value) / (12h)
        if !isfinite(derivative) || abs(derivative) <= sqrt(eps(Float64))
            return (
                omega, evaluation, history, converged=false,
                stop_reason=:singular_root_jacobian)
        end
        delta = -evaluation.value / derivative
        if abs(delta) > trust_radius
            delta *= trust_radius / abs(delta)
        end

        accepted = false
        for backtrack in 0:10
            candidate = omega + delta / 2.0^backtrack
            candidate_evaluation = _mst_incidence_evaluation(
                mode, a, candidate; angular_order, sheet_id)
            if candidate_evaluation.scaled_residual < residual
                omega = candidate
                evaluation = candidate_evaluation
                accepted = true
                break
            end
        end
        accepted || return (
            omega, evaluation, history, converged=false,
            stop_reason=:root_line_search_stagnation)
    end
    return (
        omega, evaluation, history,
        converged=evaluation.scaled_residual <= root_tolerance,
        stop_reason=evaluation.scaled_residual <= root_tolerance ?
            :root_tolerance : :root_iteration_limit)
end

function _polish_frequency_root(mode::QNMMode, a::T, initial::Complex{T};
        angular_order::Int, sheet_id::Symbol, inversion_index::Int,
        excluded_roots=Complex{T}[],
        cf_tolerance, cf_minimum_iterations::Int,
        cf_maximum_iterations::Int, root_tolerance::T,
        maximum_root_iterations::Int, trust_radius::T,
        derivative_step_scale=one(T),
        near_as_role::Symbol=:none) where {T<:AbstractFloat}
    if _near_as_mst_required(mode, a, T, near_as_role)
        return _polish_mst_incidence_root(
            mode, Float64(a), ComplexF64(initial);
            angular_order, sheet_id,
            root_tolerance=Float64(max(root_tolerance, T(2.0e-11))),
            maximum_root_iterations,
            trust_radius=Float64(trust_radius),
            derivative_step_scale=Float64(derivative_step_scale),
        )
    end
    return _polish_leaver_root(
        mode, a, initial;
        angular_order, sheet_id, inversion_index, excluded_roots,
        cf_tolerance, cf_minimum_iterations, cf_maximum_iterations,
        root_tolerance, maximum_root_iterations, trust_radius,
        derivative_step_scale=T(derivative_step_scale) *
            _near_extremal_derivative_scale(mode, a))
end

_root_equation(evaluation) = hasproperty(evaluation, :root_equation) ?
    evaluation.root_equation : :leaver_continued_fraction
_root_residual(evaluation) = hasproperty(evaluation, :scaled_residual) ?
    evaluation.scaled_residual : abs(evaluation.value)

function _near_as_anchor_seed(mode::QNMMode, target_spin::T,
        near_as_role::Symbol) where {T<:AbstractFloat}
    _near_as_complete_scope(mode, near_as_role) || throw(ArgumentError(
        "The automatic near-AS anchor is available only for " *
        "the explicitly selected (s,l,abs(m))=(-2,2,2) near-AS branch."))
    anchor_inversion = near_as_role == :algebraically_special ? 9 : 8
    positive_seed = _complex_of_type(T,
        _KERR_L2_M2_NEAR_AS_ANCHORS[near_as_role])
    effective_spin = mode.m < 0 ? -target_spin : target_spin
    seed = effective_spin < zero(T) ? -conj(positive_seed) : positive_seed
    seed_spin = copysign(T(_KERR_L2_M2_NEAR_AS_ANCHOR_SPIN), target_spin)
    return (
        seed,
        seed_spin,
        source=:published_finite_spin_anchor_repolished_locally,
        seed_inversion_index=anchor_inversion,
        seed_predictor=mode.m < 0 ?
            :complete_spectrum_a0p05_spin_reflected :
            :complete_spectrum_a0p05,
        seed_duplicate_distance=T(Inf),
    )
end

function _failed_result(mode::QNMMode, a::T, omega::Complex{T}, reason::Symbol,
        provenance; convention::Symbol=:overtone,
        overtone_index::Union{Int,Missing}=mode.n,
        inversion_index::Int=mode.n) where {T<:AbstractFloat}
    nan_complex = complex(T(NaN), T(NaN))
    return LeaverResult(
        mode, convention, overtone_index, inversion_index,
        a, omega, nan_complex, nan_complex, Complex{T}[],
        nan_complex, T(Inf), 0, T(Inf), T(Inf), _precision_bits(T),
        :failed, reason, provenance)
end

function _schwarzschild_eikonal_guess(mode::QNMMode, ::Type{T}) where {
        T<:AbstractFloat}
    scale = T(3) * sqrt(T(3))
    return complex((T(mode.l) + T(0.5)) / scale,
        -(T(mode.n) + T(0.5)) / scale)
end

function _schwarzschild_overtone_predictor(
        roots, mode::QNMMode, ::Type{T}) where {T<:AbstractFloat}
    overtone = mode.n
    if overtone >= 2 && haskey(roots, overtone - 1) &&
            haskey(roots, overtone - 2)
        predictor = 2 * roots[overtone - 1] - roots[overtone - 2]
        if real(predictor) <= zero(T)
            predictor = complex(abs(real(predictor)), imag(predictor))
        end
        return predictor
    end
    return _schwarzschild_eikonal_guess(mode, T)
end

function _schwarzschild_inversion_candidates(overtone::Int)
    candidates = Int[]
    for candidate in (overtone, overtone - 1, overtone + 1,
            overtone - 2, overtone + 2)
        candidate >= 0 && candidate ∉ candidates && push!(candidates, candidate)
    end
    return candidates
end

function _schwarzschild_seed(mode::QNMMode, ::Type{T};
        angular_order::Int,
        sheet_id::Symbol,
        cf_tolerance,
        cf_minimum_iterations::Int,
        cf_maximum_iterations::Int,
        root_tolerance::T,
        maximum_root_iterations::Int,
        trust_radius::T) where {T<:AbstractFloat}
    mode.branch == :positive_real || throw(ArgumentError(
        "A negative-real QNM requires an explicit directly sourced guess; " *
        "it is not inferred by conjugation."))
    target_key = (mode.s, mode.l, mode.n)
    if haskey(_SCHWARZSCHILD_SEEDS, target_key)
        return (
            seed=_complex_of_type(T, _SCHWARZSCHILD_SEEDS[target_key]),
            seed_spin=zero(T),
            source=:frozen_schwarzschild_seed,
            seed_inversion_index=mode.n,
            seed_predictor=:frozen,
            seed_duplicate_distance=T(Inf),
        )
    end

    return lock(_SCHWARZSCHILD_SEED_LOCK) do
        if haskey(_GENERATED_SCHWARZSCHILD_SEEDS, target_key)
            metadata = _GENERATED_SCHWARZSCHILD_SEED_METADATA[target_key]
            return (
                seed=_complex_of_type(
                    T, _GENERATED_SCHWARZSCHILD_SEEDS[target_key]),
                seed_spin=zero(T),
                source=:generated_schwarzschild_overtone_sequence,
                seed_inversion_index=metadata.inversion_index,
                seed_predictor=metadata.predictor,
                seed_duplicate_distance=T(metadata.duplicate_distance),
            )
        end

        roots = Dict{Int,Complex{T}}()
        for (key, value) in _SCHWARZSCHILD_SEEDS
            key[1] == mode.s && key[2] == mode.l || continue
            roots[key[3]] = _complex_of_type(T, value)
        end
        for (key, value) in _GENERATED_SCHWARZSCHILD_SEEDS
            key[1] == mode.s && key[2] == mode.l || continue
            roots[key[3]] = _complex_of_type(T, value)
        end

        for overtone in 0:mode.n
            haskey(roots, overtone) && continue
            if mode.s == -2 && mode.l == 2 && abs(mode.m) == 2 && overtone == 8
                # The complete-spectrum n=8/n=9 endpoints are not ordinary
                # seeds in the distinct Leaver-root sequence. Their separate
                # unconventional and pole-skipping contracts handle them.
                continue
            end
            overtone_mode = QNMMode(
                mode.s, mode.l, mode.m, overtone, :positive_real)
            predictor = _schwarzschild_overtone_predictor(
                roots, overtone_mode, T)
            eikonal = _schwarzschild_eikonal_guess(overtone_mode, T)
            predictors = predictor == eikonal ? Complex{T}[predictor] :
                Complex{T}[predictor, eikonal]
            excluded_roots = Complex{T}[
                roots[index] for index in sort(collect(keys(roots)))
                if index < overtone]
            candidates = NamedTuple[]
            trial_specs = Tuple{Int,Int,Complex{T}}[]
            for (predictor_index, trial_predictor) in pairs(predictors)
                for inversion in _schwarzschild_inversion_candidates(overtone)
                    push!(trial_specs,
                        (predictor_index, inversion, trial_predictor))
                end
            end
            for (predictor_index, inversion, trial_predictor) in trial_specs
                    polished = _polish_frequency_root(
                        overtone_mode, zero(T), trial_predictor;
                        angular_order,
                        sheet_id=Symbol(sheet_id,
                            :_schwarzschild_overtone_, inversion, :_,
                            predictor_index),
                        inversion_index=inversion,
                        excluded_roots,
                        cf_tolerance,
                        cf_minimum_iterations,
                        cf_maximum_iterations,
                        root_tolerance,
                        maximum_root_iterations=max(
                            maximum_root_iterations, 60),
                        trust_radius,
                    )
                    polished.converged || continue
                    duplicate_distance = isempty(excluded_roots) ? T(Inf) :
                        minimum(abs(polished.omega - root)
                            for root in excluded_roots)
                    duplicate_tolerance = T(1.0e-6) *
                        max(one(T), abs(polished.omega))
                    duplicate_distance > duplicate_tolerance || continue
                    overtone > 0 &&
                        imag(polished.omega) >= imag(roots[overtone - 1]) &&
                        continue
                    seed_cf_limit = max(
                        T(100) * T(cf_tolerance), sqrt(eps(T)))
                    polished.evaluation.error <= seed_cf_limit || continue
                    push!(candidates, (
                        polished,
                        inversion,
                        predictor=trial_predictor,
                        predictor_distance=abs(
                            polished.omega - trial_predictor),
                        duplicate_distance,
                    ))
                    # The physical-overtone predictor with n_inv=n is the
                    # least ambiguous route.  Once it passes all guards there
                    # is no benefit in solving the remaining fallback roots.
                    predictor_index == 1 && inversion == overtone && break
            end
            isempty(candidates) && throw(ErrorException(
                "Automatic Schwarzschild overtone continuation failed for " *
                "(s=$(mode.s), l=$(mode.l), n=$overtone): no unique " *
                "continued-fraction root survived the inversion and " *
                "duplicate-root gates."))
            selected = candidates[argmin(
                candidate.predictor_distance for candidate in candidates)]
            roots[overtone] = selected.polished.omega
            generated_key = (mode.s, mode.l, overtone)
            _GENERATED_SCHWARZSCHILD_SEEDS[generated_key] = ComplexF64(
                selected.polished.omega)
            _GENERATED_SCHWARZSCHILD_SEED_METADATA[generated_key] = (
                inversion_index=selected.inversion,
                predictor=selected.predictor,
                duplicate_distance=Float64(selected.duplicate_distance),
                root_residual=Float64(abs(
                    selected.polished.evaluation.value)),
                cf_error=Float64(selected.polished.evaluation.error),
            )
        end
        metadata = _GENERATED_SCHWARZSCHILD_SEED_METADATA[target_key]
        return (
            seed=roots[mode.n],
            seed_spin=zero(T),
            source=:generated_schwarzschild_overtone_sequence,
            seed_inversion_index=metadata.inversion_index,
            seed_predictor=metadata.predictor,
            seed_duplicate_distance=T(metadata.duplicate_distance),
        )
    end
end

function _automatic_precision_fallback(mode::QNMMode, a, omega;
        angular_order, sheet_id, cf_minimum_iterations,
        cf_maximum_iterations, maximum_root_iterations, trust_radius,
        fallback_precision_bits, float64_provenance,
        cf_tolerance, root_tolerance, stability_tolerance,
        continuation_coordinate,
        convention::Symbol, overtone_index::Union{Int,Missing},
        inversion_index::Int)
    fallback_precision_bits >= 96 || throw(ArgumentError(
        "fallback_precision_bits must be at least 96."))
    high_damping_near_extremal = abs(a) >= 0.99 &&
        abs(imag(omega)) >= 0.3
    fallback_cf_ceiling = high_damping_near_extremal ?
        (mode.branch == :negative_real ? 240000 : 120000) : 24000
    fallback_cf_maximum_iterations = max(
        cf_maximum_iterations, fallback_cf_ceiling)
    result = setprecision(BigFloat, fallback_precision_bits) do
        qnm_frequency(
            mode, BigFloat(a);
            guess=Complex{BigFloat}(
                BigFloat(real(omega)), BigFloat(imag(omega))),
            angular_order,
            sheet_id=Symbol(sheet_id, :_automatic_bigfloat),
            # A precision fallback must re-evaluate the same scientific
            # contract. Tightening the acceptance gates here makes the result
            # depend on whether Float64 happened to request the fallback.
            cf_tolerance=BigFloat(cf_tolerance),
            cf_minimum_iterations=max(cf_minimum_iterations, 600),
            cf_maximum_iterations=fallback_cf_maximum_iterations,
            root_tolerance=BigFloat(root_tolerance),
            stability_tolerance=BigFloat(stability_tolerance),
            maximum_root_iterations,
            maximum_spin_step=BigFloat("0.004"),
            minimum_spin_step=BigFloat("1e-5"),
            trust_radius=BigFloat(trust_radius),
            automatic_precision_fallback=false,
            fallback_precision_bits,
            continuation_coordinate,
            convention,
            inversion_index,
        )
    end
    precision_drift = abs(result.omega - omega) /
        max(one(real(result.omega)), abs(result.omega), abs(omega))
    branch_reference = _complex_of_type(BigFloat, omega)
    raw_branch_allowance = zero(BigFloat)
    if hasproperty(float64_provenance, :terminal_predictor)
        branch_reference = _complex_of_type(
            BigFloat, float64_provenance.terminal_predictor)
    elseif hasproperty(float64_provenance, :terminal_extrapolation_rows) &&
            !isempty(float64_provenance.terminal_extrapolation_rows)
        terminal_row = last(float64_provenance.terminal_extrapolation_rows)
        branch_reference = _complex_of_type(BigFloat, terminal_row.predictor)
        if hasproperty(terminal_row, :branch_allowance)
            raw_branch_allowance = BigFloat(terminal_row.branch_allowance)
        end
    end
    branch_scale = max(one(BigFloat), abs(result.omega),
        abs(branch_reference))
    precision_candidate_allowance =
        BigFloat(64) * sqrt(BigFloat(stability_tolerance))
    precision_predictor_allowance = max(
        precision_candidate_allowance,
        raw_branch_allowance / branch_scale,
    )
    precision_predictor_drift = abs(result.omega - branch_reference) /
        branch_scale
    precision_branch_drift = min(
        BigFloat(precision_drift), precision_predictor_drift)
    precision_branch_allowance = precision_candidate_allowance
    precision_branch_guard_passed =
        BigFloat(precision_drift) <= precision_candidate_allowance ||
        precision_predictor_drift <= precision_predictor_allowance
    provenance = merge(result.provenance, (
        precision_policy=:automatic_bigfloat_fallback,
        high_damping_near_extremal,
        fallback_cf_maximum_iterations,
        float64_candidate=omega,
        float64_bigfloat_drift=precision_drift,
        precision_branch_reference=branch_reference,
        precision_candidate_drift=precision_drift,
        precision_candidate_allowance,
        precision_predictor_drift,
        precision_predictor_allowance,
        precision_branch_drift,
        precision_branch_allowance,
        precision_branch_guard_passed,
        float64_provenance=float64_provenance,
    ))
    status = result.status in (:accepted, :estimated) &&
        precision_branch_guard_passed ? result.status : :failed
    stop_reason = !(result.status in (:accepted, :estimated)) ?
        result.stop_reason :
        precision_branch_guard_passed ? result.stop_reason :
        :precision_fallback_branch_guard
    return LeaverResult(
        result.mode, result.convention, result.overtone_index,
        result.inversion_index, result.a, result.omega, result.angular_A,
        result.lambda, result.mixing, result.cf_value, result.cf_error,
        result.cf_iterations, result.root_residual,
        result.angular_residual, result.precision_bits, status,
        stop_reason, provenance)
end

function _extremal_damped_limit_result(mode::QNMMode, a::T;
        guess,
        convention::Symbol, inversion_index,
        angular_order::Int, sheet_id::Symbol,
        cf_tolerance, cf_minimum_iterations::Int,
        cf_maximum_iterations::Int, root_tolerance,
        stability_tolerance, maximum_root_iterations::Int,
        trust_radius, automatic_precision_fallback::Bool,
        fallback_precision_bits::Int) where {T<:AbstractFloat}
    kappas = T[T(0.04), T(0.02), T(0.01), T(0.005)]
    spins = T[copysign(sqrt(one(T) - kappa^2), a)
        for kappa in kappas]
    spin_defects = T[one(T) - abs(spin) for spin in spins]
    roots = Complex{T}[]
    rows = NamedTuple[]
    previous = guess === nothing ? nothing : _complex_of_type(T, guess)
    selected_inversion = inversion_index

    for (index, spin) in pairs(spins)
        source_guess = if length(roots) >= 2
            first_index = max(1, length(roots) - 2)
            _polynomial_predictor(
                spin_defects[first_index:length(roots)],
                roots[first_index:end], spin_defects[index])
        else
            previous
        end
        result = qnm_frequency(
            mode, spin;
            guess=source_guess,
            convention,
            inversion_index=selected_inversion,
            continuation_coordinate=:nu,
            angular_order,
            sheet_id=Symbol(sheet_id, :_extremal_limit_, index),
            cf_tolerance,
            cf_minimum_iterations,
            cf_maximum_iterations,
            root_tolerance,
            stability_tolerance,
            maximum_root_iterations,
            trust_radius,
            automatic_precision_fallback,
            fallback_precision_bits,
        )
        initial_failure = result.status == :accepted ? nothing : result
        source_precision_fallback = false
        source_usable = result.status == :accepted ||
            (mode.n >= 5 && result.status == :estimated)
        if !source_usable && T === Float64
            bits = max(192, fallback_precision_bits)
            source_precision_fallback = true
            result = setprecision(bits) do
                kappa_big = BigFloat(kappas[index])
                spin_big = copysign(
                    sqrt(one(BigFloat) - kappa_big^2), BigFloat(a))
                guess_big = source_guess === nothing ? nothing :
                    _complex_of_type(BigFloat, source_guess)
                qnm_frequency(
                    mode, spin_big;
                    guess=guess_big,
                    convention,
                    inversion_index=selected_inversion,
                    continuation_coordinate=:nu,
                    angular_order=angular_order + 8,
                    sheet_id=Symbol(
                        sheet_id, :_extremal_limit_high_precision_, index),
                    # Preserve the caller's continued-fraction tail gate.  The
                    # endpoint fallback raises arithmetic precision so that
                    # the root can be polished; it must not silently replace
                    # a 1e-13 tail contract by an unattainable 1e-18 contract.
                    cf_tolerance=BigFloat(cf_tolerance),
                    cf_minimum_iterations=max(cf_minimum_iterations, 600),
                    cf_maximum_iterations=max(cf_maximum_iterations, 60000),
                    root_tolerance=min(BigFloat(root_tolerance), big"1e-14"),
                    stability_tolerance=min(
                        BigFloat(stability_tolerance), big"1e-14"),
                    maximum_root_iterations=max(maximum_root_iterations, 60),
                    trust_radius=min(BigFloat(trust_radius), big"0.02"),
                    automatic_precision_fallback=false,
                    fallback_precision_bits=bits,
                )
            end
        end
        (result.status == :accepted ||
            (mode.n >= 5 && result.status == :estimated)) ||
            return _failed_result(
            mode, a, isempty(roots) ? complex(T(NaN), T(NaN)) : roots[end],
            :extremal_limit_source_failure,
            (
                extremal_endpoint=true,
                endpoint_method=:spin_defect_richardson_extrapolation,
                failed_spin=spin,
                failed_source_reason=result.stop_reason,
                initial_source_reason=initial_failure === nothing ?
                    :not_applicable : initial_failure.stop_reason,
                source_rows=rows,
            );
            convention,
            overtone_index=convention == :leaver ? missing : mode.n,
            inversion_index=selected_inversion === :auto ? mode.n :
                Int(selected_inversion),
        )
        omega = _complex_of_type(T, result.omega)
        push!(roots, omega)
        push!(rows, (
            kappa=kappas[index],
            spin_defect=spin_defects[index],
            a=spin,
            omega,
            root_residual=T(result.root_residual),
            cf_error=T(result.cf_error),
            cf_iterations=result.cf_iterations,
            precision_bits=result.precision_bits,
            inversion_index=result.inversion_index,
            status=result.status,
            stop_reason=result.stop_reason,
            source_precision_fallback,
        ))
        previous = omega
        if selected_inversion === :auto &&
                result.inversion_index isa Integer
            selected_inversion = result.inversion_index
        end
    end

    cubic = _polynomial_predictor(spin_defects, roots, zero(T))
    quadratic = _polynomial_predictor(
        spin_defects[2:4], roots[2:4], zero(T))
    drift = abs(cubic - quadratic) /
        max(one(T), abs(cubic), abs(quadratic))
    limit_tolerance = max(T(1.0e-8), T(100) * T(stability_tolerance))
    angular = angular_branch(mode, a, cubic;
        truncation_order=angular_order + 8,
        sheet_id=Symbol(sheet_id, :_exact_extremal_limit))
    all_sources_accepted = all(row.status == :accepted for row in rows)
    angular_usable = angular.status in (
        :continued, :predictor_corrected, :spherical_anchor,
        :high_precision_refined)
    accepted = drift <= limit_tolerance && all_sources_accepted &&
        angular_usable
    estimated = !accepted && isfinite(drift) && angular_usable &&
        all(row.status in (:accepted, :estimated) for row in rows)
    contract = _schwarzschild_label_contract(mode, convention)
    selected_output_inversion = selected_inversion === :auto ? mode.n :
        Int(selected_inversion)
    nan_complex = complex(T(NaN), T(NaN))
    return LeaverResult(
        mode, convention, convention == :leaver ? missing : mode.n,
        selected_output_inversion, a, cubic, angular.angular_A,
        angular.lambda, angular.mixing, nan_complex,
        maximum(row.cf_error for row in rows),
        maximum(row.cf_iterations for row in rows), drift,
        T(angular.residual), _precision_bits(T),
        accepted ? :accepted : estimated ? :estimated : :failed,
        accepted ? :accepted_extremal_damped_limit :
            estimated ? :estimated_extremal_damped_limit :
            :extremal_limit_stability_gate,
        (
            input_type=string(typeof(a)),
            working_type=string(T),
            label_convention=contract.label_convention,
            branch_label=contract.branch_label,
            endpoint_kind=contract.endpoint_kind,
            extremal_endpoint=true,
            endpoint_method=:spin_defect_richardson_extrapolation,
            spectrum_object=:discrete_damped_pole_limit,
            simple_pole=true,
            radial_observables=
                :exact_extremal_gsn_scattering_basis,
            gsn_amplitude_limit_implemented=true,
            source_rows=rows,
            extrapolation_coordinate=:one_minus_abs_a,
            maximum_source_precision_bits=
                maximum(row.precision_bits for row in rows),
            extrapolation_orders=(quadratic=2, cubic=3),
            quadratic_limit=quadratic,
            cubic_limit=cubic,
            limit_drift=drift,
            limit_tolerance,
            all_sources_accepted,
            angular_route=angular.metadata.route,
            angular_sheet=angular.sheet_id,
            angular_order=angular.truncation_order,
            root_equation=:nonextremal_kappa_limit_extrapolation,
            family=:DM,
            claim_boundary=
                :frequency_limit_requires_separate_exact_gsn_simple_pole_gate,
        ),
    )
end

"""
    qnm_frequency(mode::QNMMode, a; guess=nothing, kwargs...)

Solve the coupled angular/radial Leaver equations in `M=1` units. Without an
explicit guess, the labeled Schwarzschild mode is continued in spin. The final
root is independently re-polished with a higher angular truncation and deeper
continued fraction. Failed scientific gates return `status=:failed` and an
explicit `stop_reason`.
"""
function _spin_reflected_result(
        requested_mode::QNMMode, requested_a, source::LeaverResult;
        angular_order::Int, sheet_id::Symbol)
    mapped_a = -source.a
    angular = angular_branch(
        requested_mode, mapped_a, source.omega;
        truncation_order=angular_order,
        sheet_id=Symbol(sheet_id, :_spin_reflection_guard),
    )
    scale = max(one(real(abs(angular.angular_A))),
        abs(angular.angular_A), abs(source.angular_A))
    angular_drift = abs(angular.angular_A - source.angular_A) / scale
    return LeaverResult(
        requested_mode,
        source.convention,
        source.overtone_index,
        source.inversion_index,
        mapped_a,
        source.omega,
        angular.angular_A,
        angular.lambda,
        angular.mixing,
        source.cf_value,
        source.cf_error,
        source.cf_iterations,
        source.root_residual,
        typeof(mapped_a)(angular.residual),
        source.precision_bits,
        source.status,
        source.stop_reason,
        merge(source.provenance, (
            spin_reflection_applied=true,
            spin_reflection_map=:a_m_to_minus_a_minus_m,
            spin_reflection_source_a=source.a,
            spin_reflection_source_m=source.mode.m,
            spin_reflection_target_a=mapped_a,
            spin_reflection_target_m=requested_mode.m,
            spin_reflection_frequency_map=:identity,
            spin_reflection_angular_drift=angular_drift,
            angular_route=angular.metadata.route,
            angular_sheet=angular.sheet_id,
            angular_order=angular.truncation_order,
        )),
    )
end

function qnm_frequency(mode::QNMMode, a; guess=nothing,
        convention=:overtone,
        inversion_index=:auto,
        continuation_coordinate=:auto,
        angular_order::Int=32,
        sheet_id::Symbol=:qnm_straight_from_spherical,
        cf_tolerance=nothing,
        cf_minimum_iterations::Int=300,
        cf_maximum_iterations=nothing,
        root_tolerance=nothing,
        stability_tolerance=nothing,
        maximum_root_iterations::Int=30,
        initial_spin_step=0.005,
        maximum_spin_step=0.004,
        minimum_spin_step=1.0e-6,
        trust_radius=0.2,
        automatic_precision_fallback::Bool=true,
        fallback_precision_bits::Int=128)
    _require_qnm_angular_backend()
    selected_convention = _qnm_convention(convention)
    T = _working_type(a, guess)
    aT = T(a)
    abs(aT) > one(T) && throw(DomainError(a,
        "A Kerr spin must satisfy abs(a) <= 1."))
    if aT < zero(T)
        source_mode = QNMMode(
            mode.s, mode.l, -mode.m, mode.n, mode.branch)
        source = qnm_frequency(
            source_mode, -aT;
            guess,
            convention=selected_convention,
            inversion_index,
            continuation_coordinate,
            angular_order,
            sheet_id=Symbol(sheet_id, :_positive_spin_source),
            cf_tolerance,
            cf_minimum_iterations,
            cf_maximum_iterations,
            root_tolerance,
            stability_tolerance,
            maximum_root_iterations,
            initial_spin_step,
            maximum_spin_step,
            minimum_spin_step,
            trust_radius,
            automatic_precision_fallback,
            fallback_precision_bits,
        )
        mapped_order = hasproperty(source.provenance, :angular_order) ?
            source.provenance.angular_order : angular_order
        return _spin_reflected_result(
            mode, aT, source; angular_order=mapped_order, sheet_id)
    end
    label_contract = _schwarzschild_label_contract(
        mode, selected_convention)
    near_as_role = label_contract.near_as_role
    automatic_inversion = inversion_index === :auto
    default_inversion = selected_convention == :leaver ||
        label_contract.algebraically_special_pair ? mode.n :
        label_contract.schwarzschild_sequence_index
    selected_inversion = automatic_inversion ? default_inversion :
        Int(inversion_index)
    selected_inversion >= 0 || throw(ArgumentError(
        "inversion_index must be nonnegative."))
    overtone_index = selected_convention == :leaver ? missing : mode.n
    selected_convention == :leaver && guess === nothing &&
        throw(ArgumentError(
            "convention=:leaver requires an explicit guess because a " *
            "continued-fraction inversion index does not uniquely label a root."))
    if abs(aT) == one(T) && _extremal_zdm_endpoint_scope(mode, aT)
        return _extremal_zdm_endpoint_result(
            mode, aT; angular_order, sheet_id,
            convention=selected_convention,
            requested_inversion_index=inversion_index)
    end
    if label_contract.algebraically_special_pair && iszero(aT)
        return near_as_role == :unconventional ?
            _unconventional_endpoint_result(
                mode, aT; angular_order, sheet_id,
                requested_inversion_index=inversion_index,
                convention=selected_convention) :
            _algebraically_special_endpoint_result(
                mode, aT; angular_order, sheet_id,
                requested_inversion_index=inversion_index,
                convention=selected_convention)
    end
    selected_coordinate = _resolve_continuation_coordinate(
        continuation_coordinate, aT, mode)
    tolerance_cf = cf_tolerance === nothing ?
        T(mode.n >= 8 ? 1.0e-12 : 1.0e-13) : T(cf_tolerance)
    maximum_cf_iterations = cf_maximum_iterations === nothing ?
        _default_cf_maximum_iterations(mode, aT, guess) :
        Int(cf_maximum_iterations)
    maximum_cf_iterations >= cf_minimum_iterations || throw(ArgumentError(
        "cf_maximum_iterations must not be below cf_minimum_iterations."))
    tolerance_root = root_tolerance === nothing ? T(1.0e-12) :
        T(root_tolerance)
    tolerance_stability = stability_tolerance === nothing ? T(1.0e-10) :
        T(stability_tolerance)
    continuation_root_tolerance = T === Float64 ?
        max(tolerance_root, T(5.0e-10)) : tolerance_root
    radius = T(trust_radius)
    initial_step = T(initial_spin_step)
    maximum_step = T(maximum_spin_step)
    minimum_step = T(minimum_spin_step)
    initial_step > zero(T) || throw(ArgumentError(
        "initial_spin_step must be positive."))
    maximum_step > zero(T) || throw(ArgumentError(
        "maximum_spin_step must be positive."))
    minimum_step > zero(T) || throw(ArgumentError(
        "minimum_spin_step must be positive."))
    minimum_step <= maximum_step || throw(ArgumentError(
        "minimum_spin_step must not exceed maximum_spin_step."))

    if abs(aT) == one(T)
        return _extremal_damped_limit_result(
            mode, aT;
            guess,
            convention=selected_convention,
            inversion_index,
            angular_order,
            sheet_id,
            cf_tolerance=tolerance_cf,
            cf_minimum_iterations,
            cf_maximum_iterations=maximum_cf_iterations,
            root_tolerance=tolerance_root,
            stability_tolerance=tolerance_stability,
            maximum_root_iterations,
            trust_radius=radius,
            automatic_precision_fallback,
            fallback_precision_bits,
        )
    end

    label_provenance = (
        label_convention=label_contract.label_convention,
        branch_label=label_contract.branch_label,
        endpoint_kind=label_contract.endpoint_kind,
        near_as_role,
        schwarzschild_sequence_index=
            label_contract.schwarzschild_sequence_index,
        schwarzschild_multiplicity=
            label_contract.schwarzschild_multiplicity,
    )
    counterrotating_seed = guess === nothing ?
        _counterrotating_near_extremal_seed(mode, aT) : nothing
    if counterrotating_seed !== nothing
        return _counterrotating_conditioned_result(
            mode, aT, counterrotating_seed;
            convention=selected_convention,
            overtone_index,
            inversion_index=selected_inversion,
            angular_order,
            sheet_id,
            cf_tolerance=tolerance_cf,
            cf_minimum_iterations,
            cf_maximum_iterations=maximum_cf_iterations,
            root_tolerance=tolerance_root,
            stability_tolerance=tolerance_stability,
        )
    end
    failed_result(omega, reason, provenance) = _failed_result(
        mode, aT, omega, reason, merge(label_provenance, provenance);
        convention=selected_convention,
        overtone_index,
        inversion_index=selected_inversion)

    seed_mode = guess === nothing && selected_convention != :leaver &&
        !ismissing(label_contract.schwarzschild_sequence_index) &&
        label_contract.schwarzschild_sequence_index != mode.n ?
        QNMMode(mode.s, mode.l, mode.m,
            label_contract.schwarzschild_sequence_index, mode.branch) : mode
    automatic_coordinate = Symbol(lowercase(String(
        continuation_coordinate))) == :auto
    seed_result = automatic_coordinate && guess === nothing &&
            selected_convention != :leaver &&
            mode.n <= 4 && _near_extremal_zdm_seed_scope(mode, aT) ?
        _near_extremal_zdm_seed(mode, aT, selected_inversion) :
        guess === nothing &&
            label_contract.algebraically_special_pair ?
        _near_as_anchor_seed(mode, aT, near_as_role) : guess === nothing ?
        _schwarzschild_seed(
            seed_mode, T;
            angular_order,
            sheet_id,
            cf_tolerance=tolerance_cf,
            cf_minimum_iterations,
            cf_maximum_iterations=maximum_cf_iterations,
            root_tolerance=continuation_root_tolerance,
            maximum_root_iterations,
            trust_radius=radius,
        ) : (
        seed=_complex_of_type(T, guess),
        seed_spin=aT,
        source=:explicit_guess,
        seed_inversion_index=selected_inversion,
        seed_predictor=:explicit,
        seed_duplicate_distance=T(Inf),
    )
    if automatic_inversion && selected_convention != :leaver &&
            guess === nothing
        selected_inversion = seed_result.seed_inversion_index
    end
    inversion_selection_policy = automatic_inversion ?
        (selected_convention != :leaver && guess === nothing ?
            (label_contract.algebraically_special_pair ?
                :finite_spin_complete_spectrum_anchor :
                :schwarzschild_seed_stability_selection) :
            :input_n_leaver_inversion) : :explicit
    initial = seed_result.seed
    guess_source = seed_result.source
    _branch_consistent(mode, initial) || throw(ArgumentError(
        "The supplied guess is inconsistent with mode.branch."))
    current = initial
    continuation_rows = NamedTuple[]
    rejected_continuation_rows = NamedTuple[]
    terminal_extrapolation_rows = NamedTuple[]
    accepted_spins = T[]
    accepted_coordinates = T[]
    accepted_roots = Complex{T}[]
    accepted_angular_A = Complex{T}[]
    last_polish = nothing

    try
        if guess === nothing
            spin = seed_result.seed_spin
            coordinate = selected_coordinate == :nu ? asin(spin) : spin
            polished = _polish_frequency_root(mode, spin, current;
                angular_order, sheet_id,
                inversion_index=selected_inversion,
                cf_tolerance=tolerance_cf,
                cf_minimum_iterations,
                cf_maximum_iterations=maximum_cf_iterations,
                root_tolerance=continuation_root_tolerance,
                maximum_root_iterations, trust_radius=radius,
                near_as_role)
            append!(continuation_rows, [merge(row, (a=spin,
                continuation_attempt=1, predictor=current))
                for row in polished.history])
            last_polish = polished
            if !polished.converged
                return failed_result(polished.omega,
                    polished.stop_reason, (
                        input_type=string(typeof(a)),
                        guess_source=guess_source,
                        spin_path=accepted_spins,
                        continuation_rows=continuation_rows,
                        rejected_continuation_rows=rejected_continuation_rows,
                    ))
            end
            current = polished.omega
            push!(accepted_spins, spin)
            push!(accepted_coordinates, coordinate)
            push!(accepted_roots, current)
            push!(accepted_angular_A,
                polished.evaluation.angular.angular_A)

            target_coordinate = selected_coordinate == :nu ? asin(aT) : aT
            sequence_anchor = _spin_sequence_anchor(mode, T)
            sequence_target = selected_coordinate == :nu ?
                target_coordinate : abs(aT) > sequence_anchor ?
                copysign(sequence_anchor, aT) : aT
            direction = sign(sequence_target - accepted_coordinates[end])
            proposed_step = min(initial_step,
                abs(sequence_target - accepted_coordinates[end]))
            while direction *
                    (sequence_target - accepted_coordinates[end]) > zero(T)
                if length(accepted_coordinates) < 3
                    proposed_step = initial_step
                else
                    curvature = max(
                        _continuation_curvature(
                            accepted_coordinates, accepted_roots),
                        _continuation_curvature(
                            accepted_coordinates, accepted_angular_A),
                    )
                    if curvature > zero(T) && isfinite(curvature)
                        proposed_step = clamp(T(0.05) / sqrt(curvature),
                            minimum_step, maximum_step)
                    else
                        proposed_step = min(proposed_step, maximum_step)
                    end
                end
                remaining = abs(sequence_target - accepted_coordinates[end])
                attempted_step = min(proposed_step, remaining)
                attempt = 0
                accepted_step = false

                while !accepted_step
                    attempt += 1
                    next_coordinate = accepted_coordinates[end] +
                        direction * attempted_step
                    next_spin = selected_coordinate == :nu ?
                        sin(next_coordinate) : next_coordinate
                    predictor = _continuation_predictor(
                        accepted_coordinates, accepted_roots, next_coordinate)
                    polished = _polish_frequency_root(
                        mode, next_spin, predictor;
                        angular_order, sheet_id,
                        inversion_index=selected_inversion,
                        cf_tolerance=tolerance_cf,
                        cf_minimum_iterations,
                        cf_maximum_iterations=maximum_cf_iterations,
                        root_tolerance=continuation_root_tolerance,
                        maximum_root_iterations, trust_radius=radius,
                        near_as_role)
                    append!(continuation_rows, [merge(row, (
                        a=next_spin, continuation_attempt=attempt,
                        continuation_coordinate=selected_coordinate,
                        coordinate_value=next_coordinate,
                        predictor=predictor)) for row in polished.history])
                    last_polish = polished
                    branch_guard = polished.converged ?
                        _continuation_branch_guard(
                            accepted_coordinates, accepted_roots,
                            next_coordinate,
                            predictor, polished.omega,
                            continuation_root_tolerance) :
                        (accepted=false, discrepancy=T(Inf),
                            allowance=zero(T))

                    if polished.converged && branch_guard.accepted
                        current = polished.omega
                        push!(accepted_spins, next_spin)
                        push!(accepted_coordinates, next_coordinate)
                        push!(accepted_roots, current)
                        push!(accepted_angular_A,
                            polished.evaluation.angular.angular_A)
                        proposed_step = attempted_step
                        accepted_step = true
                        continue
                    end

                    push!(rejected_continuation_rows, (
                        a=next_spin,
                        continuation_coordinate=selected_coordinate,
                        coordinate_value=next_coordinate,
                        attempted_step=attempted_step,
                        attempt=attempt,
                        predictor=predictor,
                        candidate=polished.omega,
                        converged=polished.converged,
                        stop_reason=polished.stop_reason,
                        branch_discrepancy=branch_guard.discrepancy,
                        branch_allowance=branch_guard.allowance,
                    ))
                    attempted_step /= 2
                    if attempted_step < minimum_step
                        reason = polished.converged ?
                            :continuation_branch_guard :
                            polished.stop_reason
                        return failed_result(polished.omega,
                            reason, (
                                input_type=string(typeof(a)),
                                guess_source=guess_source,
                                spin_path=accepted_spins,
                                coordinate_path=accepted_coordinates,
                                continuation_coordinate=selected_coordinate,
                                continuation_rows=continuation_rows,
                                rejected_continuation_rows=
                                    rejected_continuation_rows,
                                failed_spin=next_spin,
                            ))
                    end
                end
            end

            if selected_coordinate == :spin && sequence_target != aT
                predictor = _terminal_extrapolation_predictor(
                    accepted_coordinates, accepted_roots, aT)
                polished = _polish_frequency_root(mode, aT, predictor;
                    angular_order, sheet_id,
                    inversion_index=selected_inversion,
                    cf_tolerance=tolerance_cf,
                    cf_minimum_iterations,
                    cf_maximum_iterations=maximum_cf_iterations,
                    root_tolerance=continuation_root_tolerance,
                    maximum_root_iterations, trust_radius=radius,
                    near_as_role)
                append!(continuation_rows, [merge(row, (
                    a=aT, continuation_attempt=1,
                    predictor=predictor,
                    terminal_extrapolation=true))
                    for row in polished.history])
                last_polish = polished
                if !polished.converged
                    terminal_provenance = (
                        input_type=string(typeof(a)),
                        guess_source=:terminal_cubic_extrapolation,
                        spin_path=accepted_spins,
                        coordinate_path=accepted_coordinates,
                        continuation_coordinate=selected_coordinate,
                        continuation_rows=continuation_rows,
                        rejected_continuation_rows=
                            rejected_continuation_rows,
                        terminal_predictor=predictor,
                        terminal_stop_reason=polished.stop_reason,
                    )
                    if T === Float64 && automatic_precision_fallback
                        return _automatic_precision_fallback(
                            mode, aT, polished.omega;
                            angular_order, sheet_id,
                            cf_minimum_iterations,
                            cf_maximum_iterations=maximum_cf_iterations,
                            maximum_root_iterations,
                            trust_radius=radius,
                            fallback_precision_bits,
                            cf_tolerance=tolerance_cf,
                            root_tolerance=tolerance_root,
                            stability_tolerance=tolerance_stability,
                            continuation_coordinate=selected_coordinate,
                            convention=selected_convention,
                            overtone_index,
                            inversion_index=selected_inversion,
                            float64_provenance=terminal_provenance)
                    end
                    return failed_result(polished.omega,
                        polished.stop_reason, terminal_provenance)
                end
                branch_guard = _continuation_branch_guard(
                    accepted_coordinates, accepted_roots, aT, predictor,
                    polished.omega, continuation_root_tolerance)
                push!(terminal_extrapolation_rows, (
                    a=aT,
                    attempted_step=abs(aT - sequence_target),
                    attempt=1,
                    predictor=predictor,
                    candidate=polished.omega,
                    converged=polished.converged,
                    stop_reason=:terminal_extrapolation_diagnostic,
                    branch_discrepancy=branch_guard.discrepancy,
                    branch_allowance=branch_guard.allowance,
                ))
                current = polished.omega
                push!(accepted_spins, aT)
                push!(accepted_coordinates, aT)
                push!(accepted_roots, current)
                push!(accepted_angular_A,
                    polished.evaluation.angular.angular_A)
            end
        else
            polished = _polish_frequency_root(mode, aT, current;
                angular_order, sheet_id,
                inversion_index=selected_inversion,
                cf_tolerance=tolerance_cf,
                cf_minimum_iterations,
                cf_maximum_iterations=maximum_cf_iterations,
                root_tolerance=continuation_root_tolerance,
                maximum_root_iterations, trust_radius=radius,
                near_as_role)
            append!(continuation_rows, [merge(row, (a=aT,
                continuation_attempt=1, predictor=current))
                for row in polished.history])
            last_polish = polished
            if !polished.converged
                return failed_result(polished.omega,
                    polished.stop_reason, (
                        input_type=string(typeof(a)),
                        guess_source=:explicit_guess,
                        spin_path=T[aT],
                        continuation_rows=continuation_rows,
                        rejected_continuation_rows=rejected_continuation_rows,
                    ))
            end
            current = polished.omega
            push!(accepted_spins, aT)
            push!(accepted_coordinates,
                selected_coordinate == :nu ? asin(aT) : aT)
            push!(accepted_roots, current)
            push!(accepted_angular_A,
                polished.evaluation.angular.angular_A)
        end

        mst_primary = nothing
        final_primary = nothing
        if _near_as_mst_finalization_required(
                mode, aT, T, near_as_role)
            mst_primary = _polish_mst_incidence_root(
                mode, Float64(aT), ComplexF64(current);
                angular_order,
                sheet_id=Symbol(sheet_id, :_mst_final),
                root_tolerance=Float64(max(tolerance_root, T(2.0e-11))),
                maximum_root_iterations,
                trust_radius=Float64(radius),
            )
            guard = mst_primary.converged ? _polish_mst_incidence_root(
                mode, Float64(aT), ComplexF64(mst_primary.omega);
                angular_order=angular_order + 8,
                sheet_id=Symbol(sheet_id, :_mst_guard),
                root_tolerance=Float64(max(tolerance_root, T(2.0e-11))),
                maximum_root_iterations,
                trust_radius=Float64(radius),
                derivative_step_scale=0.5,
            ) : mst_primary
            current = mst_primary.omega
        else
            # Continuation deliberately uses a looser Float64 tolerance so that
            # an ill-conditioned high-overtone root can be tracked cheaply.
            # Polish at the final tolerance before measuring the independent
            # angular-order drift; otherwise ordinary Newton correction is
            # misclassified as a branch jump.
            final_primary = _polish_frequency_root(mode, aT, current;
                angular_order,
                sheet_id=Symbol(sheet_id, :_final_primary),
                inversion_index=selected_inversion,
                cf_tolerance=tolerance_cf,
                cf_minimum_iterations,
                cf_maximum_iterations=maximum_cf_iterations,
                root_tolerance=tolerance_root,
                maximum_root_iterations,
                trust_radius=radius,
                near_as_role)
            current = final_primary.omega
            guard = final_primary.converged ? _polish_frequency_root(
                mode, aT, current;
                angular_order=angular_order + 8,
                sheet_id=Symbol(sheet_id, :_guard),
                inversion_index=selected_inversion,
                cf_tolerance=tolerance_cf,
                cf_minimum_iterations=cf_minimum_iterations + 100,
                cf_maximum_iterations=maximum_cf_iterations,
                root_tolerance=tolerance_root,
                maximum_root_iterations,
                trust_radius=radius,
                derivative_step_scale=T(0.5), near_as_role) : final_primary
        end
        float64_provenance = (
            input_type=string(typeof(a)),
            working_type=string(T),
            guess_source=guess_source,
            inversion_selection_policy,
            seed_inversion_index=seed_result.seed_inversion_index,
            seed_predictor=seed_result.seed_predictor,
            seed_duplicate_distance=seed_result.seed_duplicate_distance,
            spin_path=accepted_spins,
            coordinate_path=accepted_coordinates,
            continuation_coordinate=selected_coordinate,
            spin_sequence_anchor=_spin_sequence_anchor(mode, T),
            terminal_extrapolation=selected_coordinate == :spin &&
                guess === nothing &&
                abs(aT) > _spin_sequence_anchor(mode, T),
            initial_spin_step=initial_step,
            maximum_spin_step=maximum_step,
            minimum_spin_step=minimum_step,
            continuation_rows=continuation_rows,
            rejected_continuation_rows=rejected_continuation_rows,
            terminal_extrapolation_rows=terminal_extrapolation_rows,
            near_as_mst_finalization=mst_primary !== nothing,
            mst_primary_rows=mst_primary === nothing ?
                NamedTuple[] : mst_primary.history,
            final_primary_rows=final_primary === nothing ?
                NamedTuple[] : final_primary.history,
            guard_rows=guard.history,
        )
        guard_root_equation = _root_equation(guard.evaluation)
        if !guard.converged
            if T === Float64 && automatic_precision_fallback &&
                    guard_root_equation == :leaver_continued_fraction
                return _automatic_precision_fallback(
                    mode, aT, guard.omega;
                    angular_order, sheet_id, cf_minimum_iterations,
                    cf_maximum_iterations=maximum_cf_iterations,
                    maximum_root_iterations,
                    trust_radius=radius,
                    fallback_precision_bits,
                    cf_tolerance=tolerance_cf,
                    root_tolerance=tolerance_root,
                    stability_tolerance=tolerance_stability,
                    continuation_coordinate=selected_coordinate,
                    convention=selected_convention,
                    overtone_index,
                    inversion_index=selected_inversion,
                    float64_provenance=merge(float64_provenance, (
                        float64_stop_reason=guard.stop_reason,)))
            end
            return failed_result(
                guard.omega, guard.stop_reason, (
                input_type=string(typeof(a)), spin_path=accepted_spins,
                coordinate_path=accepted_coordinates,
                continuation_coordinate=selected_coordinate,
                continuation_rows=continuation_rows,
                rejected_continuation_rows=rejected_continuation_rows,
                guard_rows=guard.history,
            ))
        end
        root_drift = T(abs(guard.omega - current) /
            max(one(T), abs(guard.omega), abs(current)))
        final = guard.evaluation
        root_equation = _root_equation(final)
        final_root_residual = T(_root_residual(final))
        effective_root_tolerance = root_equation == :gsn_mst_incidence ?
            max(tolerance_root, T(2.0e-11)) : tolerance_root
        cf_tail_gate_passed = final.error <= tolerance_cf
        near_as_cf_tail_ceiling = T(1.0e-8)
        near_as_root_position_certificate =
            root_equation == :leaver_continued_fraction &&
            _near_as_complete_scope(mode, near_as_role) &&
            near_as_role == :unconventional &&
            zero(T) < abs(aT) < T(_KERR_L2_M2_NEAR_AS_ANCHOR_SPIN) &&
            root_drift <= tolerance_stability &&
            final_root_residual <= effective_root_tolerance &&
            final.error <= near_as_cf_tail_ceiling
        accepted = root_drift <= tolerance_stability &&
            final_root_residual <= effective_root_tolerance &&
            (cf_tail_gate_passed || near_as_root_position_certificate)
        conditioned_root_limit = _conditioned_root_residual_limit(
            mode, aT, tolerance_root)
        conditioned_stability_limit = _conditioned_root_stability_limit(
            mode, aT, tolerance_stability)
        conditioned_root = root_equation == :leaver_continued_fraction &&
            !accepted && root_drift <= conditioned_stability_limit &&
            final_root_residual <= conditioned_root_limit &&
            cf_tail_gate_passed &&
            ((hasproperty(guard, :conditioned) && guard.conditioned) ||
             (final_primary !== nothing &&
              hasproperty(final_primary, :conditioned) &&
              final_primary.conditioned))
        if !accepted && !conditioned_root && T === Float64 &&
                automatic_precision_fallback &&
                root_equation == :leaver_continued_fraction
            return _automatic_precision_fallback(
                mode, aT, guard.omega;
                angular_order, sheet_id, cf_minimum_iterations,
                cf_maximum_iterations=maximum_cf_iterations,
                maximum_root_iterations,
                trust_radius=radius,
                fallback_precision_bits,
                cf_tolerance=tolerance_cf,
                root_tolerance=tolerance_root,
                stability_tolerance=tolerance_stability,
                continuation_coordinate=selected_coordinate,
                convention=selected_convention,
                overtone_index,
                inversion_index=selected_inversion,
                float64_provenance=merge(float64_provenance, (
                    float64_root_drift=root_drift,
                    float64_root_residual=final_root_residual,
                    float64_cf_error=final.error,
                )))
        end
        stop_reason = accepted ? :accepted : conditioned_root ?
            :conditioned_near_extremal_cf_floor :
            (root_drift > tolerance_stability ? :root_stability_gate :
             final_root_residual > effective_root_tolerance ?
                (root_equation == :gsn_mst_incidence ?
                    :mst_incidence_residual_gate : :raw_cf_residual_gate) :
             root_equation == :gsn_mst_incidence ?
                :mst_representation_error_gate : :lentz_error_gate)
        output_inversion = root_equation == :gsn_mst_incidence ?
            missing : selected_inversion
        return LeaverResult(
            mode, selected_convention, overtone_index,
            output_inversion, aT, guard.omega,
            final.angular.angular_A, final.angular.lambda,
            final.angular.mixing, final.value, T(final.error),
            final.iterations, final_root_residual,
            T(final.angular.residual), _precision_bits(T),
            accepted ? :accepted : conditioned_root ? :estimated : :failed,
            stop_reason,
            (
                input_type=string(typeof(a)),
                working_type=string(T),
                guess_source=guess_source,
                inversion_selection_policy,
                seed_inversion_index=seed_result.seed_inversion_index,
                seed_spin=seed_result.seed_spin,
                seed_predictor=seed_result.seed_predictor,
                seed_duplicate_distance=seed_result.seed_duplicate_distance,
                convention=selected_convention,
                overtone_index=overtone_index,
                inversion_index=selected_inversion,
                label_convention=label_contract.label_convention,
                branch_label=label_contract.branch_label,
                endpoint_kind=label_contract.endpoint_kind,
                near_as_role,
                schwarzschild_sequence_index=
                    label_contract.schwarzschild_sequence_index,
                schwarzschild_multiplicity=
                    label_contract.schwarzschild_multiplicity,
                family=_near_extremal_family(mode, aT, guard.omega),
                spin_path=accepted_spins,
                coordinate_path=accepted_coordinates,
                continuation_coordinate=selected_coordinate,
                maximum_spin_step=maximum_step,
                minimum_spin_step=minimum_step,
                angular_route=final.angular.metadata.route,
                angular_sheet=final.angular.sheet_id,
                angular_order=final.angular.truncation_order,
                cf_minimum_iterations=cf_minimum_iterations + 100,
                cf_maximum_iterations=maximum_cf_iterations,
                cf_tolerance=tolerance_cf,
                root_tolerance=tolerance_root,
                effective_root_tolerance,
                conditioned_root_limit,
                conditioned_stability_limit,
                conditioned_root,
                stability_tolerance=tolerance_stability,
                root_drift=root_drift,
                root_equation,
                acceptance_basis=near_as_root_position_certificate ?
                    :near_as_root_position_stability : root_equation,
                cf_tail_gate_passed,
                near_as_cf_tail_ceiling,
                radial_cf_applicable=
                    root_equation == :leaver_continued_fraction,
                near_as_mst_finalization=mst_primary !== nothing,
                mst_primary_rows=mst_primary === nothing ?
                    NamedTuple[] : mst_primary.history,
                final_primary_rows=final_primary === nothing ?
                    NamedTuple[] : final_primary.history,
                continuation_rows=continuation_rows,
                rejected_continuation_rows=rejected_continuation_rows,
                guard_rows=guard.history,
                upstream_qnm_commit=
                    "f3abd18e59828e7e7d75d07f20c7cbc87925edfa",
            ),
        )
    catch error
        return failed_result(current, :evaluation_error, (
            input_type=string(typeof(a)),
            spin_path=accepted_spins,
            continuation_rows=continuation_rows,
            rejected_continuation_rows=rejected_continuation_rows,
            error=sprint(showerror, error),
            last_polish=last_polish,
        ))
    end
end
