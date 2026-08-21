"""
    qnm_sequence(mode::QNMMode, spins; guess=nothing, kwargs...)

Solve a labeled QNM along the supplied spin sequence. Each accepted result is
the explicit seed for the next spin; the sequence stops at the first failure.
"""
function qnm_sequence(mode::QNMMode, spins; guess=nothing, kwargs...)
    values = collect(spins)
    isempty(values) && return LeaverResult[]
    results = LeaverResult[]
    current_guess = guess
    for spin in values
        result = qnm_frequency(mode, spin; guess=current_guess, kwargs...)
        push!(results, result)
        result.status == :accepted || break
        current_guess = result.omega
    end
    return results
end
