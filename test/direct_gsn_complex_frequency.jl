using Logging

@testset "DirectGSN complex-frequency geometry" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    params = DirectGSN.direct_complex_parameters(
        0, 2, 0, 0.3, 0.25 - 0.01im, 6.0 + 0.02im)
    contour = DirectGSN.complex_contour_parameters(params; rho_match=1.25)

    @test contour.z == params.omega / abs(params.omega)
    @test contour.dxdrho == -inv(contour.z)
    @test DirectGSN.complex_x_from_rho(contour, 1.25) ==
        -1.25 / contour.z
    @test DirectGSN.complex_r_from_rho(params, contour, 0.0) == params.rplus
end

@testset "DirectGSN complex R and Y transformations" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    omega = 0.25 - 0.01im
    r = 4.0

    for (s, branch) in ((-2, :IN), (2, :UP))
        route = DirectGSN.direct_complex_route(
            s, 2, 0, 0.3, omega, branch; N=24, tol=1e-11)
        state = DirectGSN.direct_complex_state_r(route, r)
        teukolsky = DirectGSN.direct_teukolsky_radial_function(route)
        y = DirectGSN.direct_y_radial_function(route)

        rtuple = teukolsky.Teukolsky_solution(r)
        ytuple = y.Y_solution(r)
        @test all(isfinite, rtuple)
        @test all(isfinite, ytuple)
        @test teukolsky(r) == rtuple[1]
        @test teukolsky.Teukolsky_solution.pair_value(r) == rtuple[1:2]
        @test teukolsky.transmission_amplitude == 1.0 + 0.0im
        @test y(r) == ytuple[1]
        @test y.X_solution(r) == ytuple[3]
        @test ytuple[3] == state.X
        @test DirectGSN.direct_y_branch_supported(route)
    end

    public_r = Teukolsky_radial(
        -2, 2, 0, 0.3, omega, IN;
        method="direct_ISEM", N=24, tol=1e-11)
    public_y = Y_radial(
        2, 2, 0, 0.3, omega, UP;
        method="direct_ISEM", N=24, tol=1e-11)
    @test isfinite(public_r(r))
    @test isfinite(public_y(r))
    @test public_r.GSN_solution.method == "direct_complex_ISEM"
    @test public_y.mode.omega == omega
end

@testset "DirectGSN exact-pole reciprocal UP consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2,
            4,
            2,
            0.999,
            1.004175291833628 - 0.08931836928679044im,
            IN;
            pole_normalization=true,
        )
    end
    reference_raw = (
        -0.014935179976297413 + 0.009002351031816489im,
        2.5250010464897233e-13 - 1.5835275002765294e-13im,
        -4.698659078389904e-5 - 0.002059675940231128im,
    )
    candidate_raw = (
        route.teukolsky_transmission,
        route.teukolsky_incidence,
        route.teukolsky_reflection,
    )
    alignment = reference_raw[1] / candidate_raw[1]
    aligned_raw = alignment .* candidate_raw
    scale = max(maximum(abs, reference_raw),
        maximum(abs, aligned_raw), floatmin(Float64))
    raw_metric = max(
        abs(aligned_raw[2]) / scale,
        abs(aligned_raw[3] - reference_raw[3]) / scale,
    )

    @test route.metadata.match_policy ==
        :exact_pole_reciprocal_up_consensus
    @test route.metadata.pole_reciprocal_source == :up_dual_order
    @test route.metadata.pole_reciprocal_orders == (40, 44)
    @test route.metadata.pole_reciprocal_up_residual <= 1e-6
    @test route.metadata.pole_reciprocal_confirmation_residual <= 1e-6
    @test route.metadata.pole_reciprocal_order_agreement <= 5e-10
    @test route.metadata.pole_reciprocal_state_error <= 2e-5
    @test route.metadata.pole_reciprocal_separation >= 1e-10
    @test route.metadata.pole_reciprocal_split_mismatch <= 1e-10
    @test route.metadata.pole_incidence_residual_before >= 1e-10
    @test iszero(route.incidence)
    @test iszero(route.teukolsky_incidence)
    @test raw_metric <= 1e-8
end

@testset "DirectGSN complex-frequency rational route" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    omega = 0.25 - 0.01im
    route = DirectGSN.direct_complex_route(
        0, 2, 0, 0.3, omega, IN; N=24, tol=1e-11)
    amplitudes = DirectGSN.direct_complex_amplitudes(route)

    @test amplitudes.transmission == 1.0 + 0.0im
    @test amplitudes.incidence_name == :Binc
    @test amplitudes.reflection_name == :Bref
    @test isfinite(abs(amplitudes.incidence))
    @test isfinite(abs(amplitudes.reflection))
    @test DirectGSN.direct_complex_route_patch_count(route) > 0
    @test route.metadata.backend == :direct_gsn_two_ray_rational
    @test route.p_solution === nothing

    states = DirectGSN.evaluate_complex_route_on_real_axis(
        route, (-2.0, 0.0, 2.0); coordinate=:rstar, state=true)
    @test all(state -> isfinite(abs(state.X)), states)
    @test all(state -> isfinite(abs(state.dXdrstar)), states)

    radial = DirectGSN.direct_complex_gsn_radial_function(route)
    @test radial.normalization_convention == UNIT_GSN_TRANS
    @test radial.transmission_amplitude == 1.0 + 0.0im
    @test radial.GSN_solution(0.0)[1] ==
        DirectGSN.direct_complex_evaluate_rstar(route, 0.0)

    grid = collect(range(-2.0, 2.0; length=41))
    batch_states = DirectGSN.evaluate_complex_route_on_real_axis(
        route, grid; coordinate=:rstar, state=true)
    scalar_states = [
        DirectGSN.direct_complex_state_rstar(route, value)
        for value in grid
    ]
    @test maximum(
        abs(batch_states[index].X - scalar_states[index].X) /
            max(
                abs(batch_states[index].X),
                abs(scalar_states[index].X),
                floatmin(Float64),
            )
        for index in eachindex(grid)
    ) <= 1e-10
    @test radial(grid) == getproperty.(batch_states, :X)

    pole_route = DirectGSN.direct_complex_route(
        0, 2, 0, 0.3, omega, IN;
        N=24, tol=1e-11, pole_normalization=true)
    @test !iszero(route.incidence)
    @test !iszero(route.teukolsky_incidence)
    @test iszero(pole_route.incidence)
    @test iszero(pole_route.teukolsky_incidence)
    @test pole_route.reflection == route.reflection
    @test pole_route.transmission == route.transmission
    @test pole_route.teukolsky_reflection == route.teukolsky_reflection
    @test pole_route.teukolsky_transmission == route.teukolsky_transmission
    @test pole_route.state_evaluator.match_x == route.state_evaluator.match_x
    @test pole_route.metadata.pole_normalization ==
        :exact_incidence_zero
    @test pole_route.metadata.pole_incidence_before == route.incidence
    @test pole_route.metadata.pole_teukolsky_incidence_before ==
        route.teukolsky_incidence
    @test pole_route.plan.amplitudes.gsn.incidence == 0
    @test pole_route.plan.amplitudes.teukolsky.incidence == 0
end

@testset "DirectGSN complex two-ray amplitude control" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = DirectGSN.direct_complex_route(
        -2,
        2,
        -2,
        0.5,
        0.32430731434882792 - 0.09793469966436999im,
        IN;
        N=40,
        tol=1e-14,
    )
    reference_incidence =
        -0.06679705892940039 + 0.09028574698042936im
    reference_reflection =
        -1.1822922477944375 - 0.35956758956428296im
    scale = max(
        abs(route.incidence),
        abs(route.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )
    @test abs(route.incidence - reference_incidence) / scale <= 1e-10
    @test abs(route.reflection - reference_reflection) / scale <= 1e-10

    extended_route = DirectGSN.direct_complex_route(
        -2,
        9,
        -5,
        -0.999,
        2.579235892929073 - 0.43281371863064516im,
        IN,
    )
    extended_reference_incidence =
        8.482168727043259e-8 - 7.611069378608804e-7im
    extended_reference_reflection =
        0.0039945169137575376 - 0.015399257737282247im
    extended_scale = max(
        abs(extended_route.incidence),
        abs(extended_route.reflection),
        abs(extended_reference_incidence),
        abs(extended_reference_reflection),
        floatmin(Float64),
    )
    @test extended_route.metadata.match_policy == :horizon_stokes_consensus
    @test extended_route.metadata.horizon_stokes_candidate_count == 4
    @test extended_route.metadata.horizon_stokes_agreement <= 2e-11
    @test extended_route.metadata.horizon_stokes_angle_offset > pi / 24
    @test abs(extended_route.incidence - extended_reference_incidence) /
        extended_scale <= 1e-10
    @test abs(extended_route.reflection - extended_reference_reflection) /
        extended_scale <= 1e-10
end

@testset "DirectGSN complex P-consensus fallbacks" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN

    pole_route = DirectGSN.direct_complex_route(
        2, 4, 4, 0.999,
        1.9229928773337701 - 0.010536400989477163im,
        UP,
    )
    @test pole_route.metadata.backend == :direct_gsn_p_consensus
    @test pole_route.metadata.consensus_acceptance_kind == :pole_residual
    @test pole_route.metadata.consensus_candidate_residual <= 1e-10

    scattering_route = DirectGSN.direct_complex_route(
        -2, 9, -9, -0.999,
        4.335267465995174 - 0.07393144370973294im,
        IN,
    )
    @test scattering_route.metadata.backend == :direct_gsn_p_consensus
    @test scattering_route.metadata.consensus_acceptance_kind ==
        :scattering_cross_branch
    reference_incidence =
        -0.0028900187169920023 + 0.0028430224645636im
    reference_reflection =
        -140.20118555993158 - 138.28572011222im
    scale = max(
        abs(scattering_route.incidence),
        abs(scattering_route.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )
    @test abs(scattering_route.incidence - reference_incidence) / scale <= 2e-10
    @test abs(scattering_route.reflection - reference_reflection) / scale <= 2e-10

    up_dual_route = DirectGSN.direct_complex_route(
        2, 2, -2, -0.999,
        0.9557174353764935 - 0.11693831444311616im,
        UP,
    )
    reference_incidence =
        0.2967358652686705 - 0.07879772785157275im
    reference_reflection =
        0.0046132978691429535 - 4.793600205908035e-5im
    scale = max(
        abs(up_dual_route.incidence),
        abs(up_dual_route.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )
    @test up_dual_route.metadata.backend == :direct_gsn_p_consensus
    @test up_dual_route.metadata.consensus_acceptance_kind == :up_dual_order
    @test up_dual_route.metadata.consensus_confirmation_order == 44
    @test up_dual_route.metadata.consensus_confirmation_agreement <= 1e-10
    @test up_dual_route.metadata.consensus_candidate_separation >= 1e-10
    @test up_dual_route.metadata.consensus_candidate_split_mismatch <= 1e-10
    @test up_dual_route.metadata.consensus_confirmation_split_mismatch <= 1e-10
    @test abs(up_dual_route.incidence - reference_incidence) / scale <= 1e-10
    @test abs(up_dual_route.reflection - reference_reflection) / scale <= 1e-10

    plateau_after_rejected_up_dual = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 6, -3, -0.999,
            1.7331969058265766 - 0.23365693795230845im,
            UP,
        )
    end
    plateau_reference = (
        -1.460283661366911e-11 - 3.090790134898699e-11im,
        -5.169297670670265 + 0.12330810926029158im,
    )
    plateau_scale = max(
        abs(plateau_after_rejected_up_dual.incidence),
        abs(plateau_after_rejected_up_dual.reflection),
        abs(plateau_reference[1]), abs(plateau_reference[2]),
        floatmin(Float64),
    )
    @test plateau_after_rejected_up_dual.metadata.backend ==
        :direct_gsn_two_ray_rational
    @test plateau_after_rejected_up_dual.metadata.match_policy ==
        :offpole_match_plateau
    @test plateau_after_rejected_up_dual.metadata.
        offpole_match_plateau_accepted
    @test abs(plateau_after_rejected_up_dual.incidence -
        plateau_reference[1]) / plateau_scale <= 1e-10
    @test abs(plateau_after_rejected_up_dual.reflection -
        plateau_reference[2]) / plateau_scale <= 1e-10

    pole_dual_route = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 4, 2, 0.999,
            1.004175291833628 - 0.08931836928679044im,
            UP,
        )
    end
    reference_raw = (
        -5.237411053760357 - 9.141803796804906im,
        -3.4708309449053754e-10 + 5.115827572012591e-10im,
        -45.152220494634705 + 76.90426078406239im,
    )
    alignment = reference_raw[1] / pole_dual_route.teukolsky_transmission
    aligned_raw = (
        alignment * pole_dual_route.teukolsky_transmission,
        alignment * pole_dual_route.teukolsky_incidence,
        alignment * pole_dual_route.teukolsky_reflection,
    )
    raw_scale = max(
        maximum(abs, reference_raw),
        maximum(abs, aligned_raw),
        floatmin(Float64),
    )
    raw_metric = max(
        abs(aligned_raw[2]) / raw_scale,
        abs(aligned_raw[2] - reference_raw[2]) / raw_scale,
        abs(aligned_raw[3] - reference_raw[3]) / raw_scale,
    )
    @test pole_dual_route.metadata.backend == :direct_gsn_p_consensus
    @test pole_dual_route.metadata.consensus_acceptance_kind ==
        :pole_dual_order
    @test pole_dual_route.metadata.consensus_confirmation_order == 44
    @test pole_dual_route.metadata.consensus_confirmation_agreement <= 5e-9
    @test pole_dual_route.metadata.consensus_candidate_separation >= 1e-8
    @test pole_dual_route.metadata.consensus_reflection_error <= 1e-12
    @test pole_dual_route.metadata.consensus_candidate_split_mismatch <= 2e-10
    @test pole_dual_route.metadata.consensus_confirmation_split_mismatch <=
        2e-10
    @test raw_metric <= 1e-8

    rejected_pole_dual = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 2, -1, -0.999,
            0.5806241581225255 - 0.03917865825276994im,
            UP,
        )
    end
    @test rejected_pole_dual.metadata.backend ==
        :direct_gsn_two_ray_rational
end

@testset "DirectGSN complex UP initial-match consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    accepted = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 5, 3, 0.999,
            1.5067047111599727 - 0.09102568803055272im,
            UP,
        )
    end
    reference_raw = (
        0.09792347692317993 + 0.10758430338137506im,
        -1.522413394656089e-15 - 1.58276699598299e-13im,
        0.0680308285952327 - 0.2082571875735093im,
    )
    alignment = reference_raw[1] / accepted.teukolsky_transmission
    aligned_raw = (
        alignment * accepted.teukolsky_transmission,
        alignment * accepted.teukolsky_incidence,
        alignment * accepted.teukolsky_reflection,
    )
    raw_scale = max(
        maximum(abs, reference_raw),
        maximum(abs, aligned_raw),
        floatmin(Float64),
    )
    raw_metric = max(
        abs(aligned_raw[2]) / raw_scale,
        abs(aligned_raw[2] - reference_raw[2]) / raw_scale,
        abs(aligned_raw[3] - reference_raw[3]) / raw_scale,
    )
    @test accepted.metadata.match_policy == :up_initial_match_consensus
    @test accepted.metadata.up_initial_match_initial_residual <= 1e-8
    @test accepted.metadata.up_initial_match_far_mid_agreement <= 1.5e-8
    @test accepted.metadata.up_initial_match_mid_initial_agreement <= 5e-9
    @test accepted.metadata.up_initial_match_far_initial_agreement <= 1.5e-8
    @test accepted.metadata.up_initial_match_reflection_agreement <= 3e-11
    @test accepted.metadata.up_initial_match_selected_separation >= 1e-8
    @test accepted.metadata.up_initial_match_split_mismatch <= 1e-10
    @test raw_metric <= 1e-8

    rejected = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 7, -7, -0.99,
            3.1155332626398833 - 0.2667801840563771im,
            UP,
        )
    end
    @test rejected.metadata.backend == :direct_gsn_two_ray_rational
    @test rejected.metadata.match_policy == :up_condition_retry
end

@testset "DirectGSN complex UP horizon-in order consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    accepted = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            -2, 7, -7, -0.999,
            3.370342305506561 - 0.0738820914087744im,
            UP,
        )
    end
    reference_incidence =
        -2.811729060382041e-13 - 3.838138841115436e-13im
    reference_reflection =
        0.005648441504419825 - 0.0002678337107009332im
    scale = max(
        abs(accepted.incidence),
        abs(accepted.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )
    @test accepted.metadata.match_policy ==
        :up_horizon_in_order_consensus
    @test accepted.metadata.up_horizon_in_order_orders == (40, 44, 48)
    @test accepted.metadata.up_horizon_in_order_window == :upper
    @test accepted.metadata.up_horizon_in_order_first_agreement <= 1e-10
    @test accepted.metadata.up_horizon_in_order_second_agreement <= 1e-10
    @test accepted.metadata.up_horizon_in_order_separation >= 1e-10
    @test accepted.metadata.up_horizon_in_order_split_mismatch <= 1e-10
    @test abs(accepted.incidence - reference_incidence) / scale <= 1e-8
    @test abs(accepted.reflection - reference_reflection) / scale <= 1e-8

    accepted_lower = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            -2, 2, -2, -0.999,
            0.9557174353764935 - 0.11693831444311616im,
            UP,
        )
    end
    lower_reference_incidence =
        -0.061039452963655504 - 0.3557531159429848im
    lower_reference_reflection =
        -0.0038483378869312397 - 0.003253382693545535im
    lower_scale = max(
        abs(accepted_lower.incidence),
        abs(accepted_lower.reflection),
        abs(lower_reference_incidence),
        abs(lower_reference_reflection),
        floatmin(Float64),
    )
    @test accepted_lower.metadata.match_policy ==
        :up_horizon_in_order_consensus
    @test accepted_lower.metadata.up_horizon_in_order_orders == (28, 32, 36)
    @test accepted_lower.metadata.up_horizon_in_order_window == :lower
    @test accepted_lower.metadata.up_horizon_in_order_first_agreement <= 1e-10
    @test accepted_lower.metadata.up_horizon_in_order_second_agreement <= 1e-10
    @test accepted_lower.metadata.up_horizon_in_order_separation >= 1e-10
    @test accepted_lower.metadata.up_horizon_in_order_split_mismatch <= 1e-10
    @test abs(accepted_lower.incidence - lower_reference_incidence) /
        lower_scale <= 1e-8
    @test abs(accepted_lower.reflection - lower_reference_reflection) /
        lower_scale <= 1e-8

    rejected_pole = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            -2, 4, -4, -0.99,
            1.7722180053326861 - 0.26532691036005634im,
            UP,
        )
    end
    pole_scale = max(
        abs(rejected_pole.teukolsky_transmission),
        abs(rejected_pole.teukolsky_incidence),
        abs(rejected_pole.teukolsky_reflection),
        floatmin(Float64),
    )
    @test rejected_pole.metadata.match_policy == :up_condition_retry
    @test abs(rejected_pole.teukolsky_incidence) / pole_scale <= 1e-12
end

@testset "DirectGSN complex matching-point consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = DirectGSN.direct_complex_route(
        -2,
        2,
        -2,
        -0.99,
        0.87016767472218104 - 0.14402522028159442im,
        UP,
    )
    reference_incidence =
        -0.26128168640199684 - 0.31629505159643445im
    reference_reflection =
        -0.014892985792605682 - 0.016706671822852532im
    scale = max(
        abs(route.incidence),
        abs(route.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )

    @test route.metadata.backend == :direct_gsn_two_ray_rational
    @test route.metadata.match_policy == :branch_shift_consensus
    @test route.metadata.matching_consensus_triggered
    @test route.metadata.matching_consensus_accepted
    @test route.metadata.matching_consensus_agreement <= 1e-9
    @test route.metadata.matching_consensus_separation >= 1e-9
    @test abs(route.incidence - reference_incidence) / scale <= 1e-10
    @test abs(route.reflection - reference_reflection) / scale <= 1e-10
end

@testset "DirectGSN complex off-pole matching plateau" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    in_route = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            -2, 6, 6, 0.99,
            2.6659045260113738 - 0.20719740356444682im,
            IN,
        )
    end
    in_reference = (
        -0.002644086048774402 - 0.0134440594687207im,
        -439.23812468845836 - 212.40297574366144im,
    )
    in_scale = max(
        abs(in_route.incidence), abs(in_route.reflection),
        abs(in_reference[1]), abs(in_reference[2]),
        floatmin(Float64),
    )
    @test in_route.metadata.match_policy == :offpole_match_plateau
    @test in_route.metadata.offpole_match_plateau_triggered
    @test in_route.metadata.offpole_match_plateau_accepted
    @test in_route.metadata.offpole_match_plateau_agreement <= 2.5e-11
    @test in_route.metadata.offpole_match_plateau_separation_ratio >= 11
    @test in_route.metadata.offpole_match_plateau_split_mismatch <= 1e-10
    @test in_route.metadata.offpole_match_plateau_continuity_error <= 7.5e-10
    @test abs(in_route.incidence - in_reference[1]) / in_scale <= 1e-10
    @test abs(in_route.reflection - in_reference[2]) / in_scale <= 1e-10

    up_route = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            -2, 2, 2, 0.999,
            0.955846913891749 - 0.03053858967077589im,
            UP,
        )
    end
    up_reference = (
        0.3143510832883426 + 0.5308563023045644im,
        0.2181094209584884 - 0.04745698562799662im,
    )
    up_scale = max(
        abs(up_route.incidence), abs(up_route.reflection),
        abs(up_reference[1]), abs(up_reference[2]),
        floatmin(Float64),
    )
    @test up_route.metadata.match_policy == :offpole_match_plateau
    @test up_route.metadata.offpole_match_plateau_accepted
    @test up_route.metadata.offpole_match_plateau_reused_candidate_count >= 1
    @test up_route.metadata.offpole_match_plateau_agreement <= 1e-10
    @test up_route.metadata.offpole_match_plateau_separation_ratio >= 10
    @test up_route.metadata.offpole_match_plateau_continuity_error <= 7.5e-10
    @test abs(up_route.incidence - up_reference[1]) / up_scale <= 1e-10
    @test abs(up_route.reflection - up_reference[2]) / up_scale <= 1e-10

    p_priority = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            1, 6, -6, -0.999,
            2.8962294193076357 - 0.03173751988413717im,
            UP,
        )
    end
    p_reference = (
        2.3736418029557534e-13 - 6.182238714508809e-13im,
        -0.4896886320585622 - 0.46797909529139414im,
    )
    p_scale = max(
        abs(p_priority.incidence), abs(p_priority.reflection),
        abs(p_reference[1]), abs(p_reference[2]),
        floatmin(Float64),
    )
    @test hasproperty(p_priority.metadata, :consensus_source)
    @test !hasproperty(p_priority.metadata, :match_policy)
    @test abs(p_priority.incidence - p_reference[1]) / p_scale <= 1e-10
    @test abs(p_priority.reflection - p_reference[2]) / p_scale <= 1e-10
end

@testset "DirectGSN complex conditioned pole-incidence consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    accepted = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            0, 9, -9, 0.999,
            1.3702472050317736 - 0.27916232323160656im,
            IN,
        )
    end
    reference_raw = (
        -151.5191549556269 - 118.08734110541054im,
        4.761426669427024e-10 - 7.030228213502459e-9im,
        189.05437500537752 - 260.79145636185257im,
    )
    candidate_raw = (
        accepted.teukolsky_transmission,
        accepted.teukolsky_incidence,
        accepted.teukolsky_reflection,
    )
    alignment = reference_raw[1] / candidate_raw[1]
    aligned_raw = alignment .* candidate_raw
    raw_scale = max(maximum(abs, reference_raw),
        maximum(abs, aligned_raw), floatmin(Float64))
    raw_metric = max(
        abs(aligned_raw[2]) / raw_scale,
        abs(aligned_raw[2] - reference_raw[2]) / raw_scale,
        abs(aligned_raw[3] - reference_raw[3]) / raw_scale,
    )
    @test accepted.metadata.match_policy == :branch_shift_consensus
    @test accepted.metadata.matching_condition <=
        0.02 * accepted.metadata.initial_matching_condition
    @test raw_metric <= 1e-10

    rejected = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            2, 5, 3, 0.999,
            1.5067047111599728 - 0.09102568803055272im,
            IN,
        )
    end
    @test rejected.metadata.match_policy == :rstar_zero
end

@testset "DirectGSN complex horizon Stokes consensus" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = DirectGSN.direct_complex_route(
        -2,
        7,
        -4,
        -0.999,
        2.020876178088777 - 0.3152829644626118im,
        IN,
    )
    reference_incidence =
        -0.00013534034327481535 + 6.738620184625593e-5im
    reference_reflection =
        -0.02392770749089541 + 0.04523012452154202im
    scale = max(
        abs(route.incidence),
        abs(route.reflection),
        abs(reference_incidence),
        abs(reference_reflection),
        floatmin(Float64),
    )

    @test route.metadata.backend == :direct_gsn_two_ray_rational
    @test route.metadata.match_policy == :horizon_stokes_consensus
    @test route.metadata.horizon_stokes_triggered
    @test route.metadata.horizon_stokes_accepted
    @test route.metadata.horizon_stokes_candidate_count == 2
    @test route.metadata.horizon_stokes_agreement <= 2e-11
    @test route.metadata.horizon_stokes_separation >= 1e-10
    @test route.metadata.horizon_stokes_angle_offset > 0
    @test route.plan.contour.horizon_direction ==
        route.metadata.horizon_direction
    @test abs(route.incidence - reference_incidence) / scale <= 1e-10
    @test abs(route.reflection - reference_reflection) / scale <= 1e-10
end

@testset "DirectGSN complex contour domain intersection" begin
    DirectGSN = GeneralizedSasakiNakamura.ISEM.DirectGSN
    route = Logging.with_logger(Logging.NullLogger()) do
        DirectGSN.direct_complex_route(
            1,
            2,
            -2,
            -0.999,
            0.9705487192034973 - 0.010654970822759232im,
            IN;
            N=78,
            tol=1e-13,
        )
    end

    @test route.plan.contour.model == :two_physical_rstar_rays
    @test route.plan.contour.rho_match >= 1.0
    @test route.plan.contour.rho_match == route.plan.controls.rhom
    @test isfinite(abs(route.incidence))
    @test isfinite(abs(route.reflection))
end
