# Truncated Taylor series in binary64 whose coefficient vectors come from a reusable arena, for the closed-form
# A, B series (direct_ab_series!, horizon_closed_form_ab_series!): every operation does what the TaylorSeries method
# for Taylor1{ComplexF64} does (the same promotion, order fixing and coefficient kernels mul!, div!, sqr!, pow!, in
# the same order), so the coefficients are bitwise those of Taylor1, without allocating a vector per operation.
# Written against TaylorSeries 0.18.

import TaylorSeries

const _C = ComplexF64

# Coefficient vectors of each order, the number handed out in the current call, and the number of coefficients held.
mutable struct _TaylorArena
    pools::Dict{Int,Vector{Taylor1{_C}}}
    used::Dict{Int,Int}
    stored::Int
end

_TaylorArena() = _TaylorArena(Dict{Int,Vector{Taylor1{_C}}}(), Dict{Int,Int}(), 0)

# Coefficients an arena keeps between calls (16 MB); above this the pools are dropped at the next reset.
const _ARENA_MAX_STORED = 2^20

# A zeroed Taylor1 of `order` from the arena.
function _take!(arena::_TaylorArena, order::Int)
    pool = get!(() -> Taylor1{_C}[], arena.pools, order)
    k = get(arena.used, order, 0) + 1
    arena.used[order] = k
    k > length(pool) && (push!(pool, Taylor1(zero(_C), order)); arena.stored += order + 1)
    t = pool[k]
    fill!(t.coeffs, zero(_C))
    return t
end

function _reset!(arena::_TaylorArena)
    empty!(arena.used)
    arena.stored > _ARENA_MAX_STORED && (empty!(arena.pools); arena.stored = 0)
    return arena
end

# The arena of the current task: each task has its own (no state shared between tasks or threads), and the series
# of a call are copied out before the call returns.
_task_arena() = get!(_TaylorArena, task_local_storage(), :GSN_ISEM_TAYLOR_ARENA)::_TaylorArena

struct _ArenaSeries
    t::Taylor1{_C}
    arena::_TaylorArena
end

_order(a::_ArenaSeries) = a.t.order
TaylorSeries.getcoeff(a::_ArenaSeries, k::Integer) = a.t[k]

# Independent variable of order `order` (Taylor1(ComplexF64, order)).
function _arena_variable(arena::_TaylorArena, order::Int)
    t = _take!(arena, order)
    order >= 1 && (t[1] = one(_C))
    return _ArenaSeries(t, arena)
end

# fixorder: copies at the common order (_minorder) when the orders differ.
function _fixorder(a::_ArenaSeries, b::_ArenaSeries)
    a.t.order == b.t.order && return a, b
    order = TaylorSeries._minorder(a.t, b.t)
    return _truncated(a, order), _truncated(b, order)
end

function _truncated(a::_ArenaSeries, order::Int)
    t = _take!(a.arena, order)
    n = min(order, a.t.order)
    copyto!(t.coeffs, 1, a.t.coeffs, 1, n + 1)
    return _ArenaSeries(t, a.arena)
end

# A number promoted to a series of the order of `a` (promote, then fixorder against the order-0 constant).
function _promoted(x::Number, a::_ArenaSeries)
    t = _take!(a.arena, a.t.order)
    t[0] = convert(_C, x)
    return _ArenaSeries(t, a.arena)
end

for (f, fc) in ((:+, :add!), (:-, :subst!))
    @eval begin
        function Base.$f(a::_ArenaSeries, b::_ArenaSeries)
            a, b = _fixorder(a, b)
            c = _take!(a.arena, a.t.order)
            for k in eachindex(a.t)
                TaylorSeries.$fc(c, a.t, b.t, k)
            end
            return _ArenaSeries(c, a.arena)
        end
        # Taylor1{T} with a T: only the constant term changes; any other number is promoted first.
        function Base.$f(a::_ArenaSeries, b::_C)
            c = _take!(a.arena, a.t.order)
            copyto!(c.coeffs, a.t.coeffs)
            c[0] = $f(a.t[0], b)
            return _ArenaSeries(c, a.arena)
        end
        function Base.$f(b::_C, a::_ArenaSeries)
            c = _take!(a.arena, a.t.order)
            c.coeffs .= $f.(a.t.coeffs)
            c[0] = $f(b, a.t[0])
            return _ArenaSeries(c, a.arena)
        end
        Base.$f(a::_ArenaSeries, b::Number) = $f(a, _promoted(b, a))
        Base.$f(b::Number, a::_ArenaSeries) = $f(_promoted(b, a), a)
    end
end

function Base.:-(a::_ArenaSeries)
    c = _take!(a.arena, a.t.order)
    for k in eachindex(a.t)
        TaylorSeries.subst!(c, a.t, k)
    end
    return _ArenaSeries(c, a.arena)
end

function Base.:*(a::_ArenaSeries, b::_ArenaSeries)
    a, b = _fixorder(a, b)
    c = _take!(a.arena, a.t.order)
    _mul!(c.coeffs, a.t.coeffs, b.t.coeffs)
    return _ArenaSeries(c, a.arena)
end

# TaylorSeries.mul!(c, a, b, k) for every k (c[k] = a[0] b[k] + a[1] b[k-1] + ... + a[k] b[0], summed in this order)
# with the sum kept in a local and the bounds checks dropped.
function _mul!(c::Vector{_C}, a::Vector{_C}, b::Vector{_C})
    @inbounds for k in 0:(length(c) - 1)
        sum = a[1] * b[k + 1]
        for i in 1:k
            sum += a[i + 1] * b[k - i + 1]
        end
        c[k + 1] = sum
    end
    return c
end

# Number times series (either side): x * a[k] for every coefficient.
function Base.:*(x::Number, a::_ArenaSeries)
    c = _take!(a.arena, a.t.order)
    c.coeffs .= x .* a.t.coeffs
    return _ArenaSeries(c, a.arena)
end
Base.:*(a::_ArenaSeries, x::Number) = x * a

function Base.:/(a::_ArenaSeries, x::Number)
    c = _take!(a.arena, a.t.order)
    c.coeffs .= a.t.coeffs ./ x
    return _ArenaSeries(c, a.arena)
end

function Base.:/(a::_ArenaSeries, b::_ArenaSeries)
    iszero(a.t) && !iszero(b.t) && return _ArenaSeries(_take!(a.arena, a.t.order), a.arena)
    a, b = _fixorder(a, b)
    ordfact, cdivfact = TaylorSeries.divfactorization(a.t, b.t)
    c = _take!(a.arena, a.t.order - ordfact)
    c[0] = cdivfact
    for k in eachindex(c)
        TaylorSeries.div!(c, a.t, b.t, k)
    end
    return _ArenaSeries(c, a.arena)
end

# A number over a series: promoted to an order-0 constant, which `/` returns as such when it is zero.
function Base.:/(x::Number, a::_ArenaSeries)
    cx = convert(_C, x)
    iszero(cx) && !iszero(a.t) && return _ArenaSeries(_take!(a.arena, 0), a.arena)
    return _promoted(cx, a) / a
end

function _square(a::_ArenaSeries)
    c = _take!(a.arena, a.t.order)
    for k in eachindex(a.t)
        TaylorSeries.sqr!(c, a.t, k)
    end
    return _ArenaSeries(c, a.arena)
end

function Base.:^(a::_ArenaSeries, n::Integer)
    if n == 0
        c = _take!(a.arena, a.t.order)
        c[0] = one(_C)
        return _ArenaSeries(c, a.arena)
    end
    n == 1 && return _truncated(a, a.t.order)
    n == 2 && return _square(a)
    # _pow(a, n) = a^float(n): the real-power method
    r = float(n)
    aux = a.t[0]^zero(r)
    aa = aux * a
    return _real_pow(aa, r)
end
Base.literal_pow(::typeof(^), a::_ArenaSeries, ::Val{n}) where {n} = a^n

# TaylorSeries._pow(a::Taylor1, r::Real) for r not in (0, 1, 2, 0.5).
function _real_pow(a::_ArenaSeries, r::Float64)
    aux = one(a.t[0])^r
    l0 = findfirst(a.t)
    lnull = trunc(Int, r * l0)
    lnull > a.t.order && return _ArenaSeries(_take!(a.arena, a.t.order), a.arena)
    order = l0 == 0 ? a.t.order : min(a.t.order, trunc(Int, r * a.t.order))
    c = _take!(a.arena, order)
    aux0 = _take!(a.arena, order)
    for k in eachindex(c)
        TaylorSeries.pow!(c, a.t, aux0, r, k)
    end
    return _ArenaSeries(c, a.arena)
end

function TaylorSeries.differentiate(a::_ArenaSeries)
    c = _take!(a.arena, a.t.order - 1)
    for k in eachindex(c)
        TaylorSeries.differentiate!(c, a.t, k)
    end
    return _ArenaSeries(c, a.arena)
end
