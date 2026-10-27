if !@isdefined(GSN)
    include(joinpath(@__DIR__, "..", "load_package.jl"))
end

@testset "automatic legacy fallback finite and uncertified" begin
    C = GSN.ISEM.Certification
    wrapped = C.LegacyFallbackSolution(x -> (x, 2x))
    @test wrapped(3.0) == (3.0, 6.0)
    @test wrapped.fallback === :legacy_uncertified
    @test isequal(C.certificate(wrapped), (false, NaN, (:legacy_uncertified,)))
    @test_throws ErrorException C.LegacyFallbackSolution(x -> (NaN, 1.0))(0.0)
    @test_throws ErrorException C.LegacyFallbackSolution(x -> Inf)(0.0)
    point = (-2, 2, -2, 0.999999, 0.010000000000000004-0.5im, GSN.UP)
    for method in ("ISEM", "auto")
        err = try
            GSN.GSN_radial(point...; method)
            nothing
        catch e
            e isa InterruptException && rethrow()
            e
        end
        @test err isa ErrorException
        message = sprint(showerror, err)
        println("POINT1080 ", method, " ", message); flush(stdout)
        @test occursin(method == "auto" ? "nonfinite" : "NaN", message)
    end
end