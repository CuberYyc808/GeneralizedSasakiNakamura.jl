module Errors

export absorbable

# Errors that no retry, fallback or skip may absorb: interrupts, programming errors and exhausted resources.
# Every `catch` that goes on with an alternative first runs `absorbable(err) || rethrow()`, so such an error reaches
# the caller instead of silently selecting another route.
const _NEVER_ABSORBED = (InterruptException, MethodError, UndefVarError, BoundsError, TypeError, AssertionError,
    OutOfMemoryError, StackOverflowError)

absorbable(err) = !any(T -> err isa T, _NEVER_ABSORBED)
absorbable(err::TaskFailedException) = absorbable(err.task.result)
absorbable(err::CompositeException) = all(absorbable, err.exceptions)

end
