@testset "point-particle submissions" begin
    orbit = (0.9, 10.0, 0.2, 0.3)
    mode_values(sol) = (sol.amplitude, sol.energy_flux, sol.angular_momentum_flux, sol.Carter_const_flux)
    jobs = ((-2, 2, 2, 1, 0), (-2, 2, 1, 0, 1), (2, 2, 2, 1, 1))
    # A submission with the standalone grid caps reproduces the standalone calls bit for bit.
    standalone = [mode_values(Teukolsky_pointparticle_mode(j..., orbit...)) for j in jobs]
    shared = with_pointparticle_submission(orbit...; Nmax = 2^12, Kmax = 2^9) do submission
        [mode_values(Teukolsky_pointparticle_mode(j..., orbit...; submission)) for j in jobs]
    end
    @test isequal(shared, standalone)
    # Tasks started (and joined) inside the block give the serial values.
    if Threads.nthreads() > 1
        threaded = with_pointparticle_submission(orbit...; Nmax = 2^12, Kmax = 2^9) do submission
            fetch.([Threads.@spawn mode_values(Teukolsky_pointparticle_mode(j..., orbit...; submission)) for j in jobs])
        end
        @test isequal(threaded, standalone)
    end
    # A submission belongs to one orbit and one grid.
    with_pointparticle_submission(orbit...) do submission
        @test_throws ArgumentError Teukolsky_pointparticle_mode(-2, 2, 2, 0, 0, 0.9, 10.0, 0.2, 0.4; submission)
        @test_throws ArgumentError Teukolsky_pointparticle_mode(jobs[1]..., orbit...; submission, Nmax = 2^12)
    end
    # Cancellation: later calls throw PublicSubmissionCancelled with the reason; results already returned stay valid.
    with_pointparticle_submission(orbit...; Nmax = 2^12, Kmax = 2^9) do submission
        first = Teukolsky_pointparticle_mode(jobs[1]..., orbit...; submission)
        cancel_pointparticle_submission!(submission; reason = "test stop")
        err = try
            Teukolsky_pointparticle_mode(jobs[2]..., orbit...; submission)
            nothing
        catch caught
            caught
        end
        @test err isa MS.PublicSubmissionCancelled && err.reason == "test stop"
        @test isequal(mode_values(first), standalone[1])
    end
    # stop_requested returning true cancels; a flux summation stops at its next mode.
    calls = Threads.Atomic{Int}(0)
    stop = () -> (Threads.atomic_add!(calls, 1) >= 3)
    @test_throws MS.PublicSubmissionCancelled Teukolsky_pointparticle_flux(0.9, 10.0, 0.0, 0.3; tol = 1e-5, stop_requested = stop)
    # The submission is released when the block returns; using it afterwards is an error.
    kept = with_pointparticle_submission(identity, orbit...)
    @test_throws ArgumentError Teukolsky_pointparticle_mode(jobs[1]..., orbit...; submission = kept)
end
