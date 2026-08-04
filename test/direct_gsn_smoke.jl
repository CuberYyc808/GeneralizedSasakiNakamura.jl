@testset "DirectGSN adaptive SWSH eigenvalue" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    cases = [
        (1, 2, -2, -0.99999, -0.03510321969441066),
        (0, 2, 0, -9.9999, 54.50971251592067),
        (1, 2, 0, -9.0, 47.5455735438706),
        (2, 2, 0, -45.0, 87.01704710570348),
        (1, 50, -50, -99.999, -143.41923491519816),
    ]
    for (s, l, m, c, reference) in cases
        result = DirectGSN.direct_swsh_eigenvalue(
            s, l, m, c; diagnostics=true)
        @test isapprox(result.value, reference; rtol=3e-14, atol=5e-13)
        @test result.delta <= 5e-14 + 5e-15 * abs(result.value)
    end
end

@testset "DirectGSN smoke" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN

    params = DirectGSN.direct_gsn_parameters(-2, 2, 2, 0.9, 0.5; lambda=1.0)
    controls = DirectGSN.direct_gsn_controls(
        params;
        horizon_order=8,
        ordinary_order=8,
        infinity_order=16,
        xm=0.5,
        tol=1e-9,
    )
    coefficients = DirectGSN.direct_gsn_coefficients(params; controls=controls)

    @test DirectGSN.coefficient_tables_available()
    @test coefficients.metadata.horizon_basis == :hc
    @test coefficients.metadata.infinity_basis == :ic
    @test coefficients.ordinary.value_count > 0

    horizon_in = DirectGSN.direct_zero_local_solution(coefficients, :in, 8)
    infinity_out = DirectGSN.direct_infinity_local_solution(coefficients, :out, 8)
    @test all(isfinite, horizon_in.coefficients)
    @test all(isfinite, infinity_out.coefficients)

    route_in = DirectGSN.direct_match(coefficients, :IN; controls=controls)
    route_up = DirectGSN.direct_match(coefficients, :UP; controls=controls)
    infinity_seed = route_up.infinity_out
    @test DirectGSN.DirectIteration.direct_basis_value(
        infinity_seed, infinity_seed.seed_x) == infinity_seed.seed_X
    @test DirectGSN.DirectIteration.direct_basis_state(
        infinity_seed, infinity_seed.seed_x) ==
        (infinity_seed.seed_X, infinity_seed.seed_dXdx)
    xs = (0.1, 0.5, 0.9)
    @test all(x -> isfinite(DirectGSN.direct_evaluate(route_in, x)), xs)
    @test all(x -> isfinite(DirectGSN.direct_evaluate(route_up, x)), xs)
    @test DirectGSN.direct_route_patch_count(route_in) > 0
    @test DirectGSN.direct_route_patch_count(route_up) > 0
    plan = DirectGSN.direct_route_plan(route_in)
    @test plan.frequency == :ordinary
    @test plan.spin == :regular
    @test plan.endpoint == :taylor
    @test plan.matching == :abel
    @test plan.ordinary_patches + plan.eikonal_patches +
        plan.endpoint_patches == DirectGSN.direct_route_patch_count(route_in)

    in_amplitudes = DirectGSN.direct_asymptotic_amplitudes(route_in)
    up_amplitudes = DirectGSN.direct_asymptotic_amplitudes(route_up)
    @test in_amplitudes.incidence_name == :Binc
    @test in_amplitudes.reflection_name == :Bref
    @test up_amplitudes.incidence_name == :Cinc
    @test up_amplitudes.reflection_name == :Cref
    @test DirectGSN.branch_pair_relative_error(route_in, in_amplitudes) == 0.0

    x = 0.4
    r = DirectGSN.direct_x_to_r(params, x)
    @test DirectGSN.direct_r_to_x(params, r) ≈ x
    @test DirectGSN.direct_evaluate_x(route_in, x) == DirectGSN.direct_evaluate_y(route_in, 1 - x)
    @test DirectGSN.direct_evaluate_x(route_in, x) == DirectGSN.direct_evaluate_r(route_in, r)
    state_r = DirectGSN.direct_state_r(route_in, r)
    state_rstar = DirectGSN.direct_state_rstar(route_in, x)
    @test isfinite(state_r.X)
    @test isfinite(state_r.dXdr)
    @test isfinite(state_rstar.X)
    @test isfinite(state_rstar.dXdrstar)

    rsol = DirectGSN.direct_teukolsky_solution(route_in)
    rval = rsol(r)
    @test all(isfinite, rval)
    @test rsol.pair_value(r) == rval[1:2]
    @test rsol.radial_value(r) == rval[1]
    teuk = DirectGSN.direct_teukolsky_radial_function(route_in)
    @test teuk(r) == teuk.Teukolsky_solution(r)[1]
    @test isfinite(teuk.Teukolsky_solution(r)[1])
    @test DirectGSN.direct_y_branch_supported(route_in)
    ysol = DirectGSN.direct_y_solution(route_in)
    yval = ysol(r)
    @test all(isfinite, yval)
    yfunc = DirectGSN.direct_y_radial_function(route_in)
    @test yfunc(r) == yval[1]
    @test yfunc.X_solution(r) == yval[3]
    @test isfinite(yfunc.Y_solution(r)[1])
end

@testset "DirectGSN strategy preflight" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    compressed = DirectGSN.direct_gsn_parameters(
        1, 16, -16, 0.999999, 5.0; lambda=1.0)
    regular = DirectGSN.direct_gsn_parameters(
        -2, 2, 2, 0.9, 0.5; lambda=1.0)

    spin = DirectGSN.direct_spin_regime(compressed)
    @test spin.regime == :near_extremal
    @test spin.angular_compression > 10_000
    @test DirectGSN.DirectEikonal.eikonal_preflight(compressed).candidate
    @test !DirectGSN.DirectEikonal.eikonal_preflight(regular).candidate

    @test DirectGSN.direct_frequency_regime(
        compressed; sfe=:auto, lfe=:auto).regime == :ordinary
    @test_throws ArgumentError DirectGSN.direct_frequency_regime(
        compressed; sfe=true, lfe=true)

    complex_low = DirectGSN.direct_gsn_parameters(
        -2, 2, 0, 0.0, 5.0e-14 - 0.05im; lambda=4.0)
    complex_high = DirectGSN.direct_gsn_parameters(
        -2, 2, 2, 0.9, 25.0 - 0.5im; lambda=4.0 + 0.1im)
    complex_selection = DirectGSN.direct_frequency_regime(
        complex_low; sfe=:auto, lfe=:auto)
    @test !DirectGSN.direct_sfe_regime(complex_low)
    @test !DirectGSN.direct_lfe_regime(complex_high)
    @test complex_selection.regime == :ordinary
    @test complex_selection.reason == :complex_frequency
    @test_throws ArgumentError DirectGSN.direct_frequency_regime(
        complex_low; sfe=true, lfe=false)
    @test_throws ArgumentError DirectGSN.direct_frequency_regime(
        complex_high; sfe=false, lfe=true)
end

@testset "DirectGSN binary64 precision contract" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN

    function check_contract(f)
        err = try
            f()
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        if err isa ArgumentError
            message = lowercase(sprint(showerror, err))
            @test occursin("binary64", message)
            @test occursin("will not silently downcast", message)
        end
    end

    setprecision(BigFloat, 128) do
        big_a = BigFloat("0.3")
        big_omega = Complex{BigFloat}(BigFloat("0.25"), BigFloat("0.01"))
        big_lambda = Complex{BigFloat}(BigFloat("6.0"), BigFloat("0.02"))

        check_contract(() -> DirectGSN.direct_gsn_parameters(
            -2, 2, 0, big_a, 0.25; lambda=4.0))
        check_contract(() -> DirectGSN.direct_gsn_parameters(
            -2, 2, 0, 0.3, big_omega; lambda=4.0))
        check_contract(() -> DirectGSN.direct_gsn_parameters(
            -2, 2, 0, 0.3, 0.25; lambda=BigFloat("4.0")))
        check_contract(() -> DirectGSN.direct_gsn_parameters(
            -2, 2, 0, 0.3, 0.25; lambda=4.0, nu=BigFloat("0.3")))
        check_contract(() -> DirectGSN.direct_complex_parameters(
            0, 2, 0, big_a, 0.25 - 0.01im, 6.0 + 0.02im))
        check_contract(() -> DirectGSN.direct_complex_parameters(
            0, 2, 0, 0.3, big_omega, 6.0 + 0.02im))
        check_contract(() -> DirectGSN.direct_complex_parameters(
            0, 2, 0, 0.3, 0.25 - 0.01im, big_lambda))
        check_contract(() -> GSN_radial(
            -2, 2, 0, big_a, big_omega, IN; method="direct_ISEM"))
    end

    @test DirectGSN.direct_gsn_parameters(
        -2, 2, 0, Float32(0.3), ComplexF64(0.25 - 0.01im);
        lambda=ComplexF64(4.0 + 0.02im)).omega isa ComplexF64
    @test DirectGSN.direct_complex_parameters(
        0, 2, 0, Float32(0.3), ComplexF64(0.25 - 0.01im),
        ComplexF64(6.0 + 0.02im)).omega isa ComplexF64
end

@testset "DirectGSN ordinary default order" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN

    ordinary = DirectGSN.direct_gsn_parameters(
        -2, 2, 2, 0.9, 0.5; lambda=1.0)
    ordinary_default = DirectGSN.direct_gsn_controls(
        ordinary; sfe=:auto, lfe=:auto)
    @test ordinary_default.frequency_selection.regime == :ordinary
    @test ordinary_default.ordinary_order == 40

    ordinary_explicit = DirectGSN.direct_gsn_controls(
        ordinary; ordinary_order=24, sfe=:auto, lfe=:auto)
    @test ordinary_explicit.ordinary_order == 24

    sfe = DirectGSN.direct_gsn_parameters(
        -2, 2, 2, 0.9, 1.0e-3; lambda=1.0)
    sfe_default = DirectGSN.direct_gsn_controls(
        sfe; sfe=:auto, lfe=:auto)
    @test sfe_default.frequency_selection.regime == :sfe
    @test sfe_default.ordinary_order == 24

    lfe = DirectGSN.direct_gsn_parameters(
        -2, 2, 2, 0.9, 100.0; lambda=1.0)
    lfe_default = DirectGSN.direct_gsn_controls(
        lfe; sfe=:auto, lfe=:auto)
    @test lfe_default.frequency_selection.regime == :lfe
    @test lfe_default.ordinary_order == 24
end

@testset "DirectGSN SFE transition matching" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    Matching = DirectGSN.DirectMatching

    transition = DirectGSN.direct_gsn_parameters(
        -2, 35, 35, 0.999999, 0.15; lambda=1.0)
    base = DirectGSN.direct_gsn_controls(
        transition; sfe=:auto, lfe=:auto)
    @test base.sfe
    @test base.match_x == 0.5

    in_controls = Matching._with_low_frequency_defaults(
        base, transition, :IN)
    @test in_controls.match_x == 0.9
    @test in_controls.horizon_order == base.horizon_order
    @test in_controls.ordinary_order == base.ordinary_order
    @test in_controls.basis == base.basis
    @test in_controls.source == :sfe_transition_match

    up_controls = Matching._with_low_frequency_defaults(
        base, transition, :UP)
    @test up_controls.match_x == base.match_x
    @test up_controls.source == base.source

    explicit = DirectGSN.direct_gsn_controls(
        transition; xm=0.5, sfe=:auto, lfe=:auto)
    explicit_in = Matching._with_low_frequency_defaults(
        explicit, transition, :IN)
    @test explicit_in.match_x == 0.5
    @test explicit_in.source == :user

    low = DirectGSN.direct_gsn_parameters(
        -2, 35, 35, 0.999999, 0.1; lambda=1.0)
    low_base = DirectGSN.direct_gsn_controls(low; sfe=:auto, lfe=:auto)
    low_up = Matching._with_low_frequency_defaults(low_base, low, :UP)
    @test low_up.match_x == 0.9
    @test low_up.horizon_order == 40
    @test low_up.ordinary_order == 40
end

@testset "DirectGSN conditioned horizon shift" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    params = DirectGSN.direct_gsn_parameters(
        1, 50, -50, -0.99999, 100.0)
    lfe_controls = DirectGSN.direct_gsn_controls(params; lfe=true)
    base_controls = DirectGSN.direct_gsn_controls(params; lfe=false)
    shifted = DirectGSN.direct_gsn_coefficients(
        params; controls=lfe_controls)
    generated = DirectGSN.direct_gsn_coefficients(
        params; controls=base_controls)
    @test shifted.metadata.horizon_shift
    @test !generated.metadata.horizon_shift

    order = lfe_controls.ordinary_order
    a_shift = Vector{ComplexF64}(undef, order + 1)
    b_shift = similar(a_shift)
    a_path = similar(a_shift)
    b_path = similar(a_shift)
    a_generated = similar(a_shift)
    b_generated = similar(a_shift)
    work = Vector{ComplexF64}(undef, shifted.ordinary.value_count)

    accepted, _, _, condition =
        DirectGSN.DirectOrdinaryPointExpansion.direct_shifted_ab!(
            a_shift, b_shift, work, shifted.horizon, 0.01, order)
    @test accepted
    @test condition <= 1e-13
    DirectGSN.DirectOrdinaryPointExpansion.direct_ordinary_ab_series!(
        a_path, b_path, work, shifted, 0.01, order)
    @test a_path == a_shift
    @test b_path == b_shift

    accepted_mid, _, _, _ =
        DirectGSN.DirectOrdinaryPointExpansion.direct_shifted_ab!(
            a_shift, b_shift, work, shifted.horizon, 0.5, order)
    @test !accepted_mid
    DirectGSN.DirectOrdinaryPointExpansion.direct_ordinary_ab_series!(
        a_path, b_path, work, shifted, 0.5, order)
    DirectGSN.DirectOrdinaryPointExpansion.direct_ordinary_ab_series!(
        a_generated, b_generated, work, generated, 0.5, order)
    @test a_path == a_generated
    @test b_path == b_generated
end

@testset "DirectGSN public method smoke" begin
    xin = GSN_radial(-2, 2, 2, 0.9, 0.5, IN; method="direct_ISEM")
    xauto = GSN_radial(-2, 2, 2, 0.9, 0.5, IN)
    xin_explicit = GSN_radial(
        -2, 2, 2, 0.9, 0.5, IN; method="direct_ISEM", tol=1e-12)
    xin_pair, xup_pair = GSN_radial(
        -2, 2, 2, 0.9, 0.5; method="direct_ISEM")
    rup = Teukolsky_radial(-2, 2, 2, 0.9, 0.5, UP; method="direct_ISEM")
    rauto = Teukolsky_radial(-2, 2, 2, 0.9, 0.5, UP)
    yin = Y_radial(-2, 2, 2, 0.9, 0.5, IN; method="direct_ISEM")
    yauto = Y_radial(-2, 2, 2, 0.9, 0.5, IN)
    yup = Y_radial(2, 2, 2, 0.9, 0.5, UP; method="direct_ISEM")

    r = 2.0
    @test xin.numerical_GSN_solution.tol == 1e-14
    @test xin_explicit.numerical_GSN_solution.tol == 1e-12
    @test xin_pair.numerical_GSN_solution.tol == 1e-14
    @test xup_pair.numerical_GSN_solution.tol == 1e-14
    @test xin.incidence_amplitude === xin_pair.incidence_amplitude
    @test xin.reflection_amplitude === xin_pair.reflection_amplitude
    @test xauto.method == "direct_ISEM"
    @test xauto.incidence_amplitude == xin.incidence_amplitude
    @test xauto.reflection_amplitude == xin.reflection_amplitude
    @test rauto.GSN_solution.method == "direct_ISEM"
    @test yauto.mode == yin.mode
    @test fieldtype(typeof(xin), :GSN_solution) !== Any
    @test xin(0.5) === xin.GSN_solution(0.5)[1]
    @test isfinite(xin(0.5))
    @test isfinite(rup(r))
    @test isfinite(yin(r))
    @test isfinite(yup(r))
end

@testset "DirectGSN scaled SFE bridge" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = GeneralizedSasakiNakamura._direct_isem_route(
        -2, 2, -2, -0.999, 1.0e-5, IN)

    @test route.controls.sfe
    @test route.branch == :IN
    @test route.bridge !== nothing
    @test !isempty(route.bridge.patches)
    @test all(patch -> patch isa
        DirectGSN.DirectIteration.DirectScaledPatch, route.bridge.patches)

    plan = DirectGSN.direct_route_plan(route)
    @test plan.ordinary_patches + plan.eikonal_patches +
        plan.endpoint_patches == DirectGSN.direct_route_patch_count(route)

    requested_x = 1.0 - 1.0e-5
    r = DirectGSN.direct_x_to_r(route.params, requested_x)
    rstar = GeneralizedSasakiNakamura.rstar_from_r(route.params.a, r)
    actual_x = DirectGSN.DirectTransformation._x_from_rstar(
        route.params, rstar)
    state = DirectGSN.direct_state(route, actual_x)
    wrapped = DirectGSN.direct_gsn_radial_function(route)
    @test wrapped(rstar) == state.X
    @test isfinite(state.X)
    @test isfinite(state.dXdx)

    handoff = GeneralizedSasakiNakamura._direct_isem_route(
        -2, 2, -1, -0.9, 1.0e-3, IN)
    @test handoff.endpoint_plan !== nothing
    @test DirectGSN.direct_route_plan(handoff).endpoint ==
        :mst_scaled_y_handoff
    handoff_x = handoff.endpoint_plan.seed_x
    bridge_X, bridge_dXdx =
        DirectGSN.DirectIteration.direct_basis_state(handoff.bridge, handoff_x)
    endpoint_state =
        DirectGSN.DirectMatching._endpoint_plan_state(handoff, handoff_x)
    state_scale = max(abs(bridge_X), abs(bridge_dXdx),
        abs(endpoint_state.X), abs(endpoint_state.dXdx))
    @test max(abs(bridge_X - endpoint_state.X),
        abs(bridge_dXdx - endpoint_state.dXdx)) <= 1.0e-10 * state_scale
    handoff_public = DirectGSN.direct_gsn_radial_function(handoff)
    @test handoff_public.numerical_GSN_solution.endpoint_handoff_x == handoff_x
    @test handoff_public.numerical_GSN_solution.endpoint_representation ==
        :mst_scaled_y_handoff

    up_endpoint = GeneralizedSasakiNakamura._direct_isem_route(
        2, 35, 35, 0.1, 0.1, UP)
    @test up_endpoint.endpoint_bridge isa
        DirectGSN.DirectMatching.DirectScaledBasis
    @test DirectGSN.direct_route_plan(up_endpoint).endpoint ==
        :direct_endpoint_basis
    endpoint_X, endpoint_dXdx = DirectGSN.DirectMatching._bridge_state(
        up_endpoint.endpoint_bridge, up_endpoint.match_x)
    mst_target = DirectGSN.DirectMatching._scaled_state(
        DirectGSN.DirectMatching._basis_tuple(up_endpoint.infinity_out),
        up_endpoint.solution_scale)
    endpoint_scale = max(abs(endpoint_X), abs(endpoint_dXdx),
        abs(mst_target.X), abs(mst_target.dXdx))
    @test max(abs(endpoint_X - mst_target.X),
        abs(endpoint_dXdx - mst_target.dXdx)) <=
        1.0e-10 * endpoint_scale
    endpoint_state = DirectGSN.direct_state(up_endpoint, 0.999995)
    @test isfinite(endpoint_state.X)
    @test isfinite(endpoint_state.dXdx)

    up_scaled_y = GeneralizedSasakiNakamura._direct_isem_route(
        0, 35, -1, -0.9, 1.0e-4, UP)
    @test !(up_scaled_y.endpoint_bridge isa
        DirectGSN.DirectMatching.DirectScaledBasis)
    @test DirectGSN.direct_route_plan(up_scaled_y).endpoint !=
        :direct_endpoint_basis
    scaled_y_state = DirectGSN.direct_state(up_scaled_y, 0.999995)
    @test isfinite(scaled_y_state.X)
    @test isfinite(scaled_y_state.dXdx)
end
