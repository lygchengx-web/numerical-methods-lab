def f1(x):
    return (x - 1) * (x - 2) * (x - 3)

def f2(x):
    return x**6 - x - 1


def bisection(f, a, b, tol=1e-6, max_iter=100000):
    fa = f(a)
    fb = f(b)
    if max_iter < 1 or tol <= 0:
        raise ValueError("Expected positive tolerance and iteration limit.")
    if fa == 0:
        return 0, a
    if fb == 0:
        return 0, b

    if fa * fb > 0:
        raise ValueError("The interval does not bracket a root.")

    iteration = 0

    while iteration < max_iter:
        p = (a + b) / 2
        fp = f(p)
        iteration += 1

        if abs(fp) < tol:
            return iteration, p

        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp

    return iteration, p


def regula_falsi(f, a, b, tol=1e-6, max_iter=100000):
    fa = f(a)
    fb = f(b)
    if max_iter < 1 or tol <= 0:
        raise ValueError("Expected positive tolerance and iteration limit.")
    if fa == 0:
        return 0, a
    if fb == 0:
        return 0, b

    if fa * fb > 0:
        raise ValueError("The interval does not bracket a root.")

    iteration = 0

    while iteration < max_iter:
        p = (a * fb - b * fa) / (fb - fa)
        fp = f(p)
        iteration += 1

        if abs(fp) < tol:
            return iteration, p

        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp

    return iteration, p


def compare_methods(f, a, b, name):
    b_iter, b_root = bisection(f, a, b)
    r_iter, r_root = regula_falsi(f, a, b)

    print(f"Function: {name}")
    print(f"Interval: [{a}, {b}]")

    print("Bisection Method:")
    print("  Iterations =", b_iter)
    print("  Approximate root =", b_root)
    print("  Residual =", abs(f(b_root)))

    print("Regula Falsi Method:")
    print("  Iterations =", r_iter)
    print("  Approximate root =", r_root)
    print("  Residual =", abs(f(r_root)))

    print("-" * 50)


if __name__ == "__main__":
    compare_methods(f1, 1.75, 2.95, "(x-1)(x-2)(x-3)")

    compare_methods(f2, 1.0, 1.2, "x^6 - x - 1")
    compare_methods(f2, 1.0, 2.0, "x^6 - x - 1")
