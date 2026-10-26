module Certification

# Certification state of a public ISEM call. The first pass is the strict computation. Only when it fails at the
# outermost public entry with an error that is neither a programming error nor an invalid input is the call run a
# second time with `_RELAXED[] == true`: registered relaxation sites then record (stage, score) in the collector and
# may hand back their best uncertified candidate. An empty collector, a nonfinite result or an error in the second
# pass rethrows the first-pass error. Results of the second pass carry certified = false.

using Base.ScopedValues: ScopedValue, with

export certificate, relaxed, strict, record_uncertified!

# True only inside the second pass of a public entry.
const _RELAXED = ScopedValue(false)
# Nesting depth of public entries: 0 outside, 1 in the outermost one. Nested public calls throw as today.
const _PUBLIC_DEPTH = ScopedValue(0)
# (stage, score) records of the relaxation sites taken in the current second pass.
const _COLLECTOR = ScopedValue{Union{Nothing,Vector{Tuple{Symbol,Float64}}}}(nothing)

relaxed() = _RELAXED[]

"Run `f` with relaxation off (a choice made by trying alternatives, or a nested public call)."
strict(f) = with(f, _RELAXED => false)

"Record a relaxation site taken in the current second pass; `score` is its best (not a bounding) error estimate."
function record_uncertified!(stage::Symbol, score::Real)
    collector = _COLLECTOR[]
    collector === nothing && return nothing
    push!(collector, (stage, Float64(score)))
    count!(stage)
    return nothing
end

# Counters of the certification branches (T2: all zero on results that are certified today).
const _COUNTERS = Dict{Symbol,Int}()
const _COUNTER_LOCK = ReentrantLock()

count!(name::Symbol) = lock(() -> (_COUNTERS[name] = get(_COUNTERS, name, 0) + 1; nothing), _COUNTER_LOCK)
counters() = lock(() -> copy(_COUNTERS), _COUNTER_LOCK)
reset_counters!() = lock(() -> (empty!(_COUNTERS); nothing), _COUNTER_LOCK)

# Errors that a second pass never handles: programming errors, invalid input and interrupts.
const _PASS_THROUGH = (MethodError, BoundsError, TypeError, UndefVarError, AssertionError, InterruptException,
    OutOfMemoryError, StackOverflowError, ArgumentError, DomainError)

second_pass_error(err) = !any(T -> err isa T, _PASS_THROUGH)

# Second pass deferred to the caller: method = "auto" tries the legacy route before relaxing ISEM.
const _DEFERRED = ScopedValue(false)

"""
    public_scope(f)

Run the body `f` of a public ISEM entry (GSN, Teukolsky or Y radial function). Inside another public entry it runs
strictly one level deeper, so that its routes take no second pass and its errors propagate as today.
"""
function public_scope(f)
    depth = _PUBLIC_DEPTH[]
    depth > 0 && return with(f, _PUBLIC_DEPTH => depth + 1, _RELAXED => false)
    return with(f, _PUBLIC_DEPTH => 1)
end

"Run `f` with the second pass of its routes left to the caller."
deferred(f) = with(f, _DEFERRED => true)

"""
    certified_route(f, mark; eligible=true, valid=() -> true)

Build a route with `f()`. In the outermost public entry, when not deferred and not already relaxed, a failure that
`eligible`, `valid()` and `second_pass_error` admit runs `f` once more with relaxation on; `mark(route, records)`
returns the route marked uncertified, or `nothing` when it is not finite. Otherwise, and whenever the second pass
records nothing, the first-pass error is rethrown.
"""
function certified_route(f, mark; eligible::Bool=true, valid=() -> true)
    (_PUBLIC_DEPTH[] > 1 || _DEFERRED[] || _RELAXED[]) && return f()
    first_error = nothing
    result = try
        f()
    catch err
        (eligible && second_pass_error(err) && valid()) || rethrow()
        first_error = err
        nothing
    end
    first_error === nothing && return result
    return _second_pass(f, mark, first_error)
end

function _second_pass(f, mark, first_error)
    count!(:second_pass)
    records = Tuple{Symbol,Float64}[]
    candidate = try
        with(f, _PUBLIC_DEPTH => max(_PUBLIC_DEPTH[], 1), _RELAXED => true, _COLLECTOR => records)
    catch err
        err isa InterruptException && rethrow()
        count!(:second_pass_error)
        throw(first_error)
    end
    isempty(records) && (count!(:second_pass_empty); throw(first_error))
    marked = mark(candidate, records)
    marked === nothing && (count!(:second_pass_nonfinite); throw(first_error))
    _warn_uncertified(marked)
    return marked
end

"Combined estimate of the recorded scores: the largest one (each is a best estimate, not a bound)."
estimated_error(records) = isempty(records) ? 0.0 : maximum(last, records)
stages(records) = Tuple(unique(first.(records)))

function _warn_uncertified(result)
    certified, error, stage_list = certificate(result)
    @warn "ISEM result not certified: estimated error ≈ $error (stages: $(join(stage_list, ", "))). " *
        "The estimate is the best score found, not a strict bound; certified = false is the reliable signal." maxlog = 10
    return nothing
end

"""
    certificate(x) -> (certified, estimated_error, uncertified_stages)

Certification of an ISEM route, GSN or Teukolsky radial function or Y solution. Results the strict computation
accepted are certified; their estimated error is the route's amplitude error estimate where one exists (`NaN` when
it cannot be formed, `missing` for Y solutions).
"""
certificate(x) = (true, NaN, ())

# Callable stand-in for a solution function of an uncertified result (Y, and Teukolsky DOWN/OUT, whose
# solution functions have no route to carry the certificate).
struct UncertifiedSolution{F}
    solution::F
    estimated_error::Float64
    stages::Tuple{Vararg{Symbol}}
end

(solution::UncertifiedSolution)(args...) = solution.solution(args...)

certificate(solution::UncertifiedSolution) = (false, solution.estimated_error, solution.stages)

end
