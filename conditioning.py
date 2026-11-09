import numpy as np


def make_problem(condition, seed=17):
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(96, 8)))
    right, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    singular_values = np.geomspace(1.0, 1.0 / condition, left.shape[1])
    design = (left * singular_values) @ right.T
    coefficients = rng.normal(size=left.shape[1])
    return design, design @ coefficients, coefficients


def normal_equations(design, target):
    return np.linalg.solve(design.T @ design, design.T @ target)


def qr_solve(design, target):
    q, r = np.linalg.qr(design, mode="reduced")
    return np.linalg.solve(r, q.T @ target)


def compare(condition):
    design, target, truth = make_problem(condition)
    svd, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    estimates = {"lstsq": svd}
    for name, solver in (("normal", normal_equations), ("qr", qr_solve)):
        try:
            estimates[name] = solver(design, target)
        except np.linalg.LinAlgError:
            estimates[name] = None
    actual_condition = singular_values[0] / singular_values[-1]
    target_norm = np.linalg.norm(target)
    coefficient_norm = np.linalg.norm(truth)
    results = {}
    for name, estimate in estimates.items():
        if estimate is not None:
            results[name] = (
                np.linalg.norm(estimate - truth) / coefficient_norm,
                np.linalg.norm(design @ estimate - target) / target_norm,
            )
    return actual_condition, rank, results


if __name__ == "__main__":
    print(f"{'cond(X)':>10}  {'rank':>4}  {'solver':<7}  {'coefficient error':>17}  {'residual':>11}")
    for condition in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
        actual, rank, results = compare(condition)
        for name in ("normal", "qr", "lstsq"):
            if name in results:
                error, residual = results[name]
                print(f"{actual:10.1e}  {rank:4d}  {name:<7}  {error:17.3e}  {residual:11.3e}")
            else:
                print(f"{actual:10.1e}  {rank:4d}  {name:<7}  singular")
