"""Evaluation checkpoint scheduling for DSDM condensation."""


DEFAULT_EVALUATION_ITERATIONS = (100, 500, 1000, 2000, 3000, 5000, 7500, 10000)


def _parse_iterations(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"interval", "legacy"}:
            return None
        value = value.split(",")
    elif isinstance(value, (int, float)):
        value = [value]

    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evaluate_iterations must be a list or comma-separated integers"
        ) from exc


def resolve_evaluation_iterations(niter, evaluate_iterations=None, evaluate_iter=100):
    """Return sorted 1-based condensation iterations that run full evaluation.

    An empty/None schedule keeps compatibility with the legacy fixed-interval
    behavior. The final condensation iteration is always evaluated.
    """
    niter = int(niter)
    if niter <= 0:
        raise ValueError("niter must be positive")

    requested = _parse_iterations(evaluate_iterations)
    if requested is None or not requested:
        evaluate_iter = int(evaluate_iter)
        if evaluate_iter <= 0:
            raise ValueError("evaluate_iter must be positive")
        requested = range(evaluate_iter, niter + 1, evaluate_iter)

    invalid = [iteration for iteration in requested if iteration <= 0]
    if invalid:
        raise ValueError("evaluation iterations must be positive")

    resolved = {iteration for iteration in requested if iteration <= niter}
    resolved.add(niter)
    return sorted(resolved)
