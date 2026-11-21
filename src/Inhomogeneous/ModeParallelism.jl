# mode_parallelism > 1: the l walks of a flux summation computed ahead on the Julia threads of one process.
# Included in ModeSummation.

# l walks computed ahead of a flux summation on the Julia threads of one process (mode_parallelism > 1). A walk
# (s, m, n, k) is the serial l walk of that (m, n, k) in one shell: it evaluates its modes in the serial order with the
# serial stop rule from the shell's parameters only (no summation state), so it is pure; the summation asks for the
# walks in its serial order and applies the acceptance (counters, cap tracking), the m, k and n stop rules, the sums
# and the record to the returned modes in that order, so the result equals the serial one. The walks of a shell are
# planned from the reach of the previous shell (`_walk_plan!`, with the parameters `params_for` gives, set per shell);
# a walk asked for that was not planned is computed then, with a few more ahead. A walk's error is raised when it is
# asked for. Walks no longer wanted are waited for and dropped (`_walk_discard!`, at the end of each shell).
mutable struct _WalkPrefetch{R}
    run::R                 # (key, params) -> the walk, a Vector of (l, mode) in walk order; pure
    params_for::Any        # key -> params, set by the summation per shell (in the serial order of its bookkeeping)
    tasks::Dict{NTuple{4, Int}, Tuple{Any, Task}}
    walks::Int             # walks started
    walks_used::Int        # walks asked for
    modes::Int             # modes of the finished walks (used or not)
    modes_used::Int        # modes of the walks asked for
end
_WalkPrefetch(run) = _WalkPrefetch(run, nothing, Dict{NTuple{4, Int}, Tuple{Any, Task}}(), 0, 0, 0, 0)
# A prefetch for a resumed pass (see _with_auto_caps), counting on from the previous pass's (nothing: from zero).
_walk_prefetch_resumed(run, prev) = prev === nothing ? _WalkPrefetch(run) :
    _WalkPrefetch(run, nothing, Dict{NTuple{4, Int}, Tuple{Any, Task}}(), prev.walks, prev.walks_used, prev.modes, prev.modes_used)

function _walk_spawn!(pf::_WalkPrefetch, key::NTuple{4, Int}, params)
    haskey(pf.tasks, key) && return nothing
    pf.walks += 1
    run = pf.run
    pf.tasks[key] = (params, Threads.@spawn try
        (true, run(key, params))   # the walk's error is returned, not thrown: raised when the walk is asked for
    catch error
        (false, error)
    end)
    return nothing
end

# Start the walks of `keys` (in that order) that are not running yet.
function _walk_plan!(pf::_WalkPrefetch, keys)
    for key in keys
        haskey(pf.tasks, key) || _walk_spawn!(pf, key, pf.params_for(key))
    end
    return nothing
end

# The walk `key`: the one computed ahead, else computed now (and the walks `ahead` started).
function (pf::_WalkPrefetch)(key::NTuple{4, Int}; ahead = ())
    if !haskey(pf.tasks, key)
        _walk_spawn!(pf, key, pf.params_for(key))
        _walk_plan!(pf, ahead)
    end
    ok, value = fetch(pf.tasks[key][2])
    delete!(pf.tasks, key)
    ok || throw(value)
    pf.walks_used += 1
    pf.modes += _walk_length(value) + _walk_unused(value)
    pf.modes_used += _walk_length(value)
    return value
end

# Wait for the walks no longer wanted and drop them (a running walk cannot be cancelled).
function _walk_discard!(pf::_WalkPrefetch)
    for key in collect(keys(pf.tasks))
        ok, value = fetch(pf.tasks[key][2])
        delete!(pf.tasks, key)
        ok && (pf.modes += _walk_length(value) + _walk_unused(value))
    end
    return nothing
end

_walk_length(walk) = walk isa AbstractVector ? length(walk) : length(walk.modes)
_walk_unused(walk) = walk isa NamedTuple && haskey(walk, :unused) ? walk.unused : 0

# the summation reads only the scalars of a mode computed ahead: drop the trajectory, radial and angular objects
_strip_mode!(mode) = (mode isa AbstractDict && foreach(k -> delete!(mode, k), ("Trajectory", "YSolution", "SWSH")); mode)
