import math

def f(x):
    return math.expm1(x) - x - x**2 / 2

def fp(x):
    return math.expm1(x) - x

def newton(x0, tol=1e-10, max_iter=100):
    x = x0

    for n in range(max_iter + 1):
        if abs(f(x)) < tol:
            return x, n

        if abs(f(x)) > 1e5:
            return x, n

        x = x - f(x) / fp(x)

    return x, max_iter


def modified_newton(x0, tol=1e-10, max_iter=100):
    x = x0
    m = 3

    for n in range(max_iter + 1):
        if abs(f(x)) < tol:
            return x, n

        if abs(f(x)) > 1e5:
            return x, n

        x = x - m * f(x) / fp(x)

    return x, max_iter


if __name__ == "__main__":
    x0 = 1

    root_newton, iter_newton = newton(x0)
    root_modified, iter_modified = modified_newton(x0)

    print("Newton's method:")
    print("iterations =", iter_newton)
    print("approximation =", format(root_newton, ".8g"))
    print("f(x) =", f(root_newton))

    print()

    print("Modified Newton's method:")
    print("iterations =", iter_modified)
    print("approximation =", format(root_modified, ".8g"))
    print("f(x) =", f(root_modified))
