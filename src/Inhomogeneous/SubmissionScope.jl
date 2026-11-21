# Included inside ModeSummation. A submission holds one orbit's lazily built, read-only trajectory, master sample
# and geometry owner, shared by the mode and flux calls that pass it; every mode keeps its own private cache.
# Callers join the tasks they start before the submission is released. Releasing drops the submission's references;
# it never empties returned scientific objects.

export PublicSubmission, PublicSubmissionCancelled
export create_public_submission, ensure_public_submission, with_public_submission
export submission_orbit!, submission_master!, submission_geometry!
export with_submission_mode, with_submission_operation, request_submission_cancel!, release_public_submission!

# Thrown when a mode or operation starts after the submission was cancelled; `reason` is the cancel reason.
struct PublicSubmissionCancelled <: Exception
    reason::String
end
Base.showerror(io::IO, e::PublicSubmissionCancelled) = print(io, "Submission cancelled: ", e.reason)
# No retry, fallback or skip may absorb a cancellation.
absorbable(::PublicSubmissionCancelled) = false

mutable struct PublicSubmission
    parameters::Tuple
    family::Symbol
    Nmax::Int
    Kmax::Int
    context::NamedTuple
    orbit::Any
    master::Any
    geometry::Any
    alternate_masters::Dict{Symbol, Any}
    alternate_geometries::Dict{Symbol, Any}
    lock::ReentrantLock
    stop_requested::Any                       # nothing or a zero-argument callback; `true` cancels
    cancel_reason::Union{Nothing, String}     # nothing while not cancelled
    released::Bool
end

_public_family(e, x) = kerr_geo_orbit_shape(e, x)
# the same context as a geometry owner's (precision, rounding, loaded KerrGeodesics)
_public_context() = ConvolutionIntegrals._submission_geometry_context()

# New submission for one orbit (a, p, e, x) and sampling-grid caps Nmax, Kmax. Orbit, dense sample and
# geometry owner are built lazily on first use.
function create_public_submission(a, p, e, x; Nmax::Int = 2^14, Kmax::Int = 2^12, stop_requested = nothing)
    Nmax > 0 && Kmax > 0 || throw(ArgumentError("Submission grid caps must be positive"))
    return PublicSubmission((a, p, e, x), _public_family(e, x), Nmax, Kmax, _public_context(), nothing, nothing,
        nothing, Dict{Symbol, Any}(), Dict{Symbol, Any}(), ReentrantLock(), stop_requested, nothing, false)
end

# Called under scope.lock.
function _public_check(scope::PublicSubmission)
    scope.context == _public_context() || throw(ArgumentError("Submission precision/rounding/dependency context changed"))
    scope.released && throw(ArgumentError("Submission is released"))
    scope.cancel_reason === nothing || throw(PublicSubmissionCancelled(scope.cancel_reason))
    return nothing
end

# Return (scope, owned): a new submission (owned = true) when `submission` is nothing, otherwise the given
# one after checking that it belongs to the same orbit and grid caps (owned = false).
function ensure_public_submission(submission, a, p, e, x; Nmax::Int = submission === nothing ? 2^14 : submission.Nmax, Kmax::Int = submission === nothing ? 2^12 : submission.Kmax,
        stop_requested = nothing)
    submission === nothing && return (create_public_submission(a, p, e, x; Nmax, Kmax, stop_requested), true)
    submission isa PublicSubmission || throw(ArgumentError("submission must be a PublicSubmission"))
    lock(submission.lock) do
        _public_check(submission)
        isequal(submission.parameters, (a, p, e, x)) || throw(ArgumentError("Submission belongs to another orbit"))
        (submission.Nmax == Nmax && submission.Kmax == Kmax) ||
            throw(ArgumentError("Submission belongs to another sampling grid"))
        stop_requested !== nothing && (submission.stop_requested = stop_requested)
    end
    return (submission, false)
end

# Cancel the submission: later modes and operations throw PublicSubmissionCancelled; running ones finish.
function request_submission_cancel!(scope::PublicSubmission; reason = "cooperative cancellation")
    lock(scope.lock) do
        scope.cancel_reason === nothing && (scope.cancel_reason = string(reason))
    end
    return nothing
end

_public_cancelled(scope::PublicSubmission) = lock(() -> scope.cancel_reason !== nothing, scope.lock)

# Ask the stop callback; `true` cancels. A callback that throws cancels and its error propagates.
function _public_check_stop(scope)
    callback = lock(() -> scope.stop_requested, scope.lock)
    callback === nothing && return nothing
    requested = try
        callback()
    catch caught
        absorbable(caught) || rethrow()
        request_submission_cancel!(scope; reason = "stop callback raised an exception")
        rethrow()
    end
    requested === true && request_submission_cancel!(scope)
    return nothing
end

# Lazily built KerrGeodesics orbit shared by all modes of the submission.
function submission_orbit!(scope::PublicSubmission)
    return lock(scope.lock) do
        _public_check(scope)
        scope.orbit === nothing && (scope.orbit = kerr_geo_orbit(scope.parameters...))
        scope.orbit
    end
end

function _public_check_family(family::Symbol)
    family in (:circular, :eccentric, :inclined, :generic) ||
        throw(ArgumentError("Unknown submission orbit family"))
    return family
end

function _public_build_master(scope::PublicSubmission, family::Symbol)
    KG = submission_orbit!(scope)
    x = scope.parameters[4]
    return family === :generic ? GridSampling.kerr_geo_generic_sample_dense(KG, scope.Nmax, scope.Kmax) :
        family === :eccentric ? GridSampling.kerr_geo_eccentric_sample_dense(KG, scope.Nmax) :
        family === :inclined ? GridSampling.kerr_geo_inclined_sample_dense(KG, x, scope.Kmax) : KG
end

# Lazily built dense sample (master grid) of the orbit for `family`; non-default families are kept separately.
function submission_master!(scope::PublicSubmission; family::Symbol = scope.family)
    _public_check_family(family)
    return lock(scope.lock) do
        _public_check(scope)
        if family === scope.family
            scope.master === nothing && (scope.master = _public_build_master(scope, family))
            return scope.master
        end
        return get!(() -> _public_build_master(scope, family), scope.alternate_masters, family)
    end
end

function _public_build_geometry(scope::PublicSubmission, family::Symbol)
    master = submission_master!(scope; family)
    return family === :generic ? ConvolutionIntegrals.GenericGeometryOwner(master) :
        family === :eccentric ? ConvolutionIntegrals.EccentricGeometryOwner(master) :
        ConvolutionIntegrals.InclinedGeometryOwner(master)
end

# Lazily built geometry owner for `family` (nothing for circular orbits), shared read-only by the mode caches.
function submission_geometry!(scope::PublicSubmission; family::Symbol = scope.family)
    _public_check_family(family)
    return lock(scope.lock) do
        _public_check(scope)
        family === :circular && return nothing
        if family === scope.family
            scope.geometry === nothing && (scope.geometry = _public_build_geometry(scope, family))
            return scope.geometry
        end
        return get!(() -> _public_build_geometry(scope, family), scope.alternate_geometries, family)
    end
end

function _public_private_cache(scope; family::Symbol = scope.family, share_geometry::Bool = true)
    _public_check_family(family)
    geometry = share_geometry ? submission_geometry!(scope; family) : nothing
    return family === :generic ? (geometry === nothing ? ConvolutionIntegrals.GenericFluxCache() : ConvolutionIntegrals.GenericFluxCache(geometry)) :
        family === :eccentric ? (geometry === nothing ? ConvolutionIntegrals.EccentricFluxCache() : ConvolutionIntegrals.EccentricFluxCache(geometry)) :
        family === :inclined ? (geometry === nothing ? ConvolutionIntegrals.InclinedFluxCache() : ConvolutionIntegrals.InclinedFluxCache(geometry)) : nothing
end

function _public_clear_cache!(cache, family::Symbol)
    cache === nothing && return nothing
    family === :generic ? ConvolutionIntegrals.clear_generic_mode_cache!(cache) :
        family === :eccentric ? ConvolutionIntegrals.clear_eccentric_mode_cache!(cache) :
        ConvolutionIntegrals.clear_inclined_mode_cache!(cache)
    return nothing
end

# Run f() as one operation (a mode-flux helper or a summation) of the submission.
function with_submission_operation(f, scope::PublicSubmission)
    _public_check_stop(scope)
    lock(() -> _public_check(scope), scope.lock)
    return f()
end

# Run f(cache) for one mode with a private mode cache, then clear that cache. Returns f's result.
function with_submission_mode(f, scope::PublicSubmission; mode = nothing, family::Symbol = scope.family, share_geometry::Bool = true)
    _public_check_family(family)
    _public_check_stop(scope)
    lock(() -> _public_check(scope), scope.lock)
    cache = _public_private_cache(scope; family, share_geometry)
    try
        return f(cache)
    finally
        _public_clear_cache!(cache, family)
    end
end

# Release the geometry owners and drop the submission's references. The caller has joined every task that uses the
# submission. Idempotent.
function release_public_submission!(scope::PublicSubmission)
    geometries = lock(scope.lock) do
        scope.released && return nothing
        scope.released = true
        owners = Pair{Symbol, Any}[]
        scope.geometry === nothing || push!(owners, scope.family => scope.geometry)
        append!(owners, collect(scope.alternate_geometries))
        scope.geometry = scope.master = scope.orbit = nothing
        empty!(scope.alternate_masters)
        empty!(scope.alternate_geometries)
        owners
    end
    geometries === nothing && return nothing
    for (family, geometry) in geometries
        family === :generic ? ConvolutionIntegrals.release_generic_geometry!(geometry) :
            family === :eccentric ? ConvolutionIntegrals.release_eccentric_geometry!(geometry) :
            ConvolutionIntegrals.release_inclined_geometry!(geometry)
    end
    return nothing
end

# Create a submission, call f(scope) and release the submission on exit.
function with_public_submission(f, a, p, e, x; kwargs...)
    scope = create_public_submission(a, p, e, x; kwargs...)
    return _with_public_scope(() -> f(scope), scope, true)
end

function _with_public_scope(f, scope, owned::Bool)
    try
        return f()
    finally
        owned && release_public_submission!(scope)
    end
end
