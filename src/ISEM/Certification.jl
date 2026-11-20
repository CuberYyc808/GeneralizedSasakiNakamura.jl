module Certification

export certificate

# Errors that are not numerical failures of a route: programming errors, invalid input and interrupts. Callers that
# turn a numerical failure into a recorded status (QNM observables) let these propagate.
const _PASS_THROUGH = (MethodError, BoundsError, TypeError, UndefVarError, AssertionError, InterruptException,
    OutOfMemoryError, StackOverflowError, ArgumentError, DomainError)

second_pass_error(err) = !any(T -> err isa T, _PASS_THROUGH)

"""
    certificate(x) -> (certified, estimated_error, uncertified_stages)

Certification of an ISEM route, GSN or Teukolsky radial function or Y solution. Results the strict computation
accepted are certified; their estimated error is the route's amplitude error estimate where one exists (`NaN` when
it cannot be formed, `missing` for Y solutions).
Automatic legacy fallbacks are uncertified with a `NaN` estimate and the stage
`:legacy_uncertified`; their numerical solution also exposes `fallback=:legacy_uncertified`.
"""
certificate(x) = (true, NaN, ())

# Callable stand-in for a solution function of an uncertified result (Y, and Teukolsky DOWN/OUT, whose
# solution functions have no route to carry the certificate).
struct UncertifiedSolution{F} <: Function
    solution::F
    estimated_error::Float64
    stages::Tuple{Vararg{Symbol}}
end

(solution::UncertifiedSolution)(args...) = solution.solution(args...)

certificate(solution::UncertifiedSolution) = (false, solution.estimated_error, solution.stages)

# Keep the legacy numerical solution callable while carrying the fallback provenance.
struct LegacyFallbackSolution{F}
    solution::F
    fallback::Symbol
end
LegacyFallbackSolution(solution) = LegacyFallbackSolution(solution, :legacy_uncertified)
function (solution::LegacyFallbackSolution)(args...; kwargs...)
    value = solution.solution(args...; kwargs...)
    finite = value isa Number ? isfinite(value) : all(isfinite, value)
    finite || error("uncertified legacy fallback produced nonfinite radial state")
    return value
end
certificate(solution::LegacyFallbackSolution) = (false, NaN, (solution.fallback,))

end
