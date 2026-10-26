# Closed-form A, B series from exact rational functions of u = r - r(x0) (binary64 points of direct_ab_series!).
# The potentials sF, sU are evaluated once with a sum-of-fractions type whose numerators and denominators are
# polynomials in u (sums with different denominators are kept apart: merging them cancels in sU), and every
# fraction is then expanded in the step variable t = x - x0 through u = lam t / (1 - t / (1 - x0)) by a Horner
# recurrence that multiplies by u in O(n), followed by one series division; B = -U / g^2 is the product of the series
# of U and of 1 / g^2. Small polynomial products, six divisions and one product replace the dozens of order-n series
# products of the Taylor1 form (30 us against 77 us at order 40). Polynomials and work series live in a task-local
# arena (bounded by one call's use); a call that would exceed the fixed sizes returns `nothing` (the caller takes
# the Taylor1 form).

const _RA_CAPACITY = 48       # coefficients per polynomial (degree < 48)
const _RA_TERMS = 4           # fractions per sum

struct _RationalOverflow <: Exception end

mutable struct _RationalArena
    polys::Vector{Vector{_C}}
    degree::Vector{Int}
    used::Int
    series::NTuple{6,Vector{_C}}   # work series of length order + 2
end

_RationalArena() = _RationalArena(Vector{_C}[], Int[], 0, ntuple(_ -> _C[], 6))

_ra_reset!(arena::_RationalArena) = (arena.used = 0; arena)

function _ra_take!(arena::_RationalArena)
    arena.used += 1
    if arena.used > length(arena.polys)
        push!(arena.polys, zeros(_C, _RA_CAPACITY))
        push!(arena.degree, 0)
    end
    return arena.used
end

_ra_task_arena() = get!(_RationalArena, task_local_storage(), :GSN_ISEM_RATIONAL_ARENA)::_RationalArena

# Sum of up to _RA_TERMS fractions num[k] / den[k] (indices of polynomials in `arena`).
struct _RatSum
    n::Int
    num::NTuple{_RA_TERMS,Int}
    den::NTuple{_RA_TERMS,Int}
    arena::_RationalArena
end

function _ra_poly!(arena, coeffs...)
    k = _ra_take!(arena)
    p = arena.polys[k]
    for (i, c) in enumerate(coeffs)
        p[i] = c
    end
    arena.degree[k] = length(coeffs) - 1
    return k
end

function _ra_mul!(arena, i, j)
    di, dj = arena.degree[i], arena.degree[j]
    di + dj < _RA_CAPACITY || throw(_RationalOverflow())
    k = _ra_take!(arena)
    p, a, b = arena.polys[k], arena.polys[i], arena.polys[j]
    @inbounds begin
        fill!(view(p, 1:(di + dj + 1)), zero(_C))
        for x in 0:di, y in 0:dj
            p[x + y + 1] += a[x + 1] * b[y + 1]
        end
    end
    arena.degree[k] = di + dj
    return k
end

function _ra_add!(arena, i, j, sign)
    di, dj = arena.degree[i], arena.degree[j]
    k = _ra_take!(arena)
    p, a, b = arena.polys[k], arena.polys[i], arena.polys[j]
    d = max(di, dj)
    @inbounds for x in 0:d
        p[x + 1] = (x <= di ? a[x + 1] : zero(_C)) + (x <= dj ? sign * b[x + 1] : zero(_C))
    end
    arena.degree[k] = d
    return k
end

function _ra_scale!(arena, i, c)
    k = _ra_take!(arena)
    d = arena.degree[i]
    p, a = arena.polys[k], arena.polys[i]
    @inbounds for x in 0:d
        p[x + 1] = c * a[x + 1]
    end
    arena.degree[k] = d
    return k
end

_ra_equal(arena, i, j) = i == j || (arena.degree[i] == arena.degree[j] &&
    view(arena.polys[i], 1:(arena.degree[i] + 1)) == view(arena.polys[j], 1:(arena.degree[j] + 1)))

# Term lists are fixed-size tuples (no allocation per operation).
const _RaIdx = NTuple{_RA_TERMS,Int}
const _RA_EMPTY = ntuple(_ -> 0, _RA_TERMS)

_ra_single_sum(arena, num, den) = _RatSum(1, Base.setindex(_RA_EMPTY, num, 1), Base.setindex(_RA_EMPTY, den, 1), arena)

# (n, nums, dens) with num/den appended, merged into a term with an equal denominator.
function _ra_push(arena, n, nums::_RaIdx, dens::_RaIdx, num, den)
    for k in 1:n
        if _ra_equal(arena, dens[k], den)
            return n, Base.setindex(nums, _ra_add!(arena, nums[k], num, 1), k), dens
        end
    end
    n < _RA_TERMS || throw(_RationalOverflow())
    return n + 1, Base.setindex(nums, num, n + 1), Base.setindex(dens, den, n + 1)
end

function Base.:+(x::_RatSum, y::_RatSum)
    arena = x.arena
    n, nums, dens = x.n, x.num, x.den
    for k in 1:y.n
        n, nums, dens = _ra_push(arena, n, nums, dens, y.num[k], y.den[k])
    end
    return _RatSum(n, nums, dens, arena)
end

function Base.:-(x::_RatSum)
    nums = x.num
    for k in 1:x.n
        nums = Base.setindex(nums, _ra_scale!(x.arena, x.num[k], -one(_C)), k)
    end
    return _RatSum(x.n, nums, x.den, x.arena)
end
Base.:-(x::_RatSum, y::_RatSum) = x + (-y)

function Base.:*(x::_RatSum, y::_RatSum)
    arena = x.arena
    n, nums, dens = 0, _RA_EMPTY, _RA_EMPTY
    for i in 1:x.n, j in 1:y.n
        n, nums, dens = _ra_push(arena, n, nums, dens, _ra_mul!(arena, x.num[i], y.num[j]),
            _ra_mul!(arena, x.den[i], y.den[j]))
    end
    return _RatSum(n, nums, dens, arena)
end

# The divisor collapsed to one fraction.
function _ra_single(y::_RatSum)
    arena = y.arena
    num, den = y.num[1], y.den[1]
    for k in 2:y.n
        if _ra_equal(arena, den, y.den[k])
            num = _ra_add!(arena, num, y.num[k], 1)
        else
            num = _ra_add!(arena, _ra_mul!(arena, num, y.den[k]), _ra_mul!(arena, y.num[k], den), 1)
            den = _ra_mul!(arena, den, y.den[k])
        end
    end
    return num, den
end

function Base.:/(x::_RatSum, y::_RatSum)
    arena = x.arena
    n, d = _ra_single(y)
    nums, dens = x.num, x.den
    for k in 1:x.n
        nums = Base.setindex(nums, _ra_mul!(arena, x.num[k], d), k)
        dens = Base.setindex(dens, _ra_mul!(arena, x.den[k], n), k)
    end
    return _RatSum(x.n, nums, dens, arena)
end

_ra_constant(arena, c) = _ra_single_sum(arena, _ra_poly!(arena, convert(_C, c)), _ra_poly!(arena, one(_C)))

for op in (:+, :-, :/)
    @eval Base.$op(x::_RatSum, c::Number) = $op(x, _ra_constant(x.arena, c))
    @eval Base.$op(c::Number, x::_RatSum) = $op(_ra_constant(x.arena, c), x)
end
function Base.:*(c::Number, x::_RatSum)
    nums = x.num
    for k in 1:x.n
        nums = Base.setindex(nums, _ra_scale!(x.arena, x.num[k], convert(_C, c)), k)
    end
    return _RatSum(x.n, nums, x.den, x.arena)
end
Base.:*(x::_RatSum, c::Number) = c * x
function Base.:^(x::_RatSum, k::Integer)
    k >= 1 || throw(_RationalOverflow())
    y = x
    for _ in 2:k
        y = y * x
    end
    return y
end
Base.literal_pow(::typeof(^), x::_RatSum, ::Val{k}) where {k} = x^k

# Series in t of q(u(t)), u = lam t / (1 - t / ga), into s[1:n+1], by Horner with (u s)_j = lam s_{j-1} + (u s)_{j-1} / ga.
function _ra_horner!(s, w, q, degree, lam, ga, n)
    @inbounds begin
        fill!(view(s, 1:(n + 1)), zero(_C))
        s[1] = q[degree + 1]
        inv_ga = inv(ga)
        for k in degree:-1:1
            # s <- u s in place: (u s)_0 = 0, (u s)_j = lam s_{j-1} + (u s)_{j-1} / ga, with s_{j-1} the old value
            old = s[1]
            s[1] = zero(_C)
            for j in 1:n
                next = s[j + 1]
                s[j + 1] = lam * old + s[j] * inv_ga
                old = next
            end
            s[1] += q[k]
        end
    end
    return s
end

# out[1:n+1] += series of num/den.
function _ra_add_series!(out, buffers, arena, num, den, lam, ga, n)
    sn, sd, w = buffers
    _ra_horner!(sn, w, arena.polys[num], arena.degree[num], lam, ga, n)
    _ra_horner!(sd, w, arena.polys[den], arena.degree[den], lam, ga, n)
    @inbounds for k in 0:n
        c = sn[k + 1]
        for j in 1:k
            c -= sd[j + 1] * w[k - j + 1]
        end
        w[k + 1] = c / sd[1]
    end
    @inbounds for k in 1:(n + 1)
        out[k] += w[k]
    end
    return out
end

# A and B of direct_ab_series! (same equation, same output layout); `nothing` when the sizes are exceeded.
function _rational_ab_series!(a_out, b_out, params, x0, order::Integer)
    s, m, a, omega, lambda = params.s, params.m, params.a, params.omega, params.lambda
    kappa = sqrt(1 - a^2)
    rplus, rminus = 1 + kappa, 1 - kappa
    ga = 1 - x0
    r0 = (rplus - rminus * x0) / ga
    lam = (rplus - rminus) / ga^2
    n = order + 1
    arena = _ra_reset!(_ra_task_arena())
    try
        one_ = _ra_poly!(arena, one(_C))
        r = _ra_single_sum(arena, _ra_poly!(arena, convert(_C, r0), one(_C)), one_)
        cached = _PowerCachedSeries(r, 7)
        F = sF(s, m, a, omega, lambda, cached)
        U = sU(s, m, a, omega, lambda, cached)
        F isa _RatSum || (F = _ra_constant(arena, F))
        U isa _RatSum || (U = _ra_constant(arena, U))
        # g = (r+ - r-) (r - r+)(r - r-) / ((r - r-)^2 (r^2 + a^2)) in factored form, dr/dx = (r - r-)^2 / (r+ - r-)
        up = _ra_poly!(arena, convert(_C, r0 - rplus), one(_C))
        um = _ra_poly!(arena, convert(_C, r0 - rminus), one(_C))
        gn = _ra_scale!(arena, _ra_mul!(arena, up, um), convert(_C, rplus - rminus))
        um2 = _ra_mul!(arena, um, um)
        gd = _ra_mul!(arena, um2, _ra_poly!(arena, convert(_C, r0^2 + a^2), convert(_C, 2r0), one(_C)))
        dgn = _ra_add!(arena, _ra_mul!(arena, _ra_derivative!(arena, gn), gd), _ra_mul!(arena, gn, _ra_derivative!(arena, gd)), -1)
        drdx = _ra_scale!(arena, um2, convert(_C, inv(rplus - rminus)))
        Aterm = _ra_single_sum(arena, _ra_mul!(arena, _ra_mul!(arena, dgn, drdx), gd),
            _ra_mul!(arena, _ra_mul!(arena, gd, gd), gn))
        ginv = _ra_single_sum(arena, gd, gn)
        A = Aterm - F * ginv
        g2 = _ra_single_sum(arena, _ra_mul!(arena, gd, gd), _ra_mul!(arena, gn, gn))
        foreach(v -> resize!(v, n + 1), arena.series)
        sn, sd, w, sa, sb, sg = arena.series
        buffers = (sn, sd, w)
        fill!(sa, zero(_C))
        fill!(sb, zero(_C))
        for k in 1:A.n
            _ra_add_series!(sa, buffers, arena, A.num[k], A.den[k], lam, ga, n)
        end
        # B = -U / g^2: the series of U and of 1/g^2 apart, then one product (shorter Horner sums than U / g^2 termwise)
        su = sb
        for k in 1:U.n
            _ra_add_series!(su, buffers, arena, U.num[k], U.den[k], lam, ga, n)
        end
        fill!(sg, zero(_C))
        _ra_add_series!(sg, buffers, arena, g2.num[1], g2.den[1], lam, ga, n)
        @inbounds for k in (n + 1):-1:1
            c = zero(_C)
            for j in 1:k
                c += su[j] * sg[k - j + 1]
            end
            sb[k] = -c
        end
        a_terms, b_terms = ab_term_counts(:O, order)
        copyto!(a_out, 1, sa, 1, a_terms)
        copyto!(b_out, 1, sb, 1, b_terms)
        return a_terms, b_terms
    catch err
        err isa _RationalOverflow && return nothing
        rethrow()
    end
end

function _ra_derivative!(arena, i)
    d = arena.degree[i]
    k = _ra_take!(arena)
    p, a = arena.polys[k], arena.polys[i]
    if d == 0
        p[1] = zero(_C)
    else
        @inbounds for x in 1:d
            p[x] = x * a[x + 1]
        end
    end
    arena.degree[k] = max(d - 1, 0)
    return k
end
