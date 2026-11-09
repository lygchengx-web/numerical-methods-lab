import math

def f(x):
    return x + math.cos(x)

def df(x):
    return 1 - math.sin(x)


TOL = 1e-10
MAX_ITER = 100
DIVERGE = 1e5


def newton(p0):
    p = p0

    for i in range(1, MAX_ITER + 1):
        fp = f(p)

        if abs(fp) < TOL:
            return p, i
        if abs(fp) > DIVERGE:
            return p, "Diverged"

        p = p - fp / df(p)

    return p, MAX_ITER


def secant(p0, p1):
    for i in range(1, MAX_ITER + 1):
        f0 = f(p0)
        f1 = f(p1)

        if abs(f1) < TOL:
            return p1, i
        if abs(f1) > DIVERGE:
            return p1, "Diverged"

        denominator = f1 - f0
        p2 = p1 - f1 * (p1 - p0) / denominator

        p0, p1 = p1, p2

    return p1, MAX_ITER


# Reuse the derivative at the starting point.
# p_{n+1} = p_n - f(p_n) / f'(p0)
def modified_newton(p0):
    p = p0
    df_p0 = df(p0)   # fixed derivative at the initial point

    for i in range(1, MAX_ITER + 1):
        fp = f(p)

        if abs(fp) < TOL:
            return p, i
        if abs(fp) > DIVERGE:
            return p, "Diverged"

        p = p - fp / df_p0

    return p, MAX_ITER


# Part (a): p0 = -5, p1 = 5


if __name__ == "__main__":
    print("Starting at -5")

    p_newton_a, it_newton_a = newton(-5)
    p_secant_a, it_secant_a = secant(-5, 5)
    p_mod_a, it_mod_a = modified_newton(-5)

    print(f"Newton:           p = {p_newton_a:.10f}, iterations = {it_newton_a}")
    print(f"Secant:           p = {p_secant_a:.10f}, iterations = {it_secant_a}")

    if it_mod_a == "Diverged":
        print(f"Modified Newton:  p = {p_mod_a:.10f}, result = Diverged")
    else:
        print(f"Modified Newton:  p = {p_mod_a:.10f}, iterations = {it_mod_a}")


    print("\nStarting at -0.9")

    p_newton_b, it_newton_b = newton(-0.9)
    p_secant_b, it_secant_b = secant(-0.9, 5)
    p_mod_b, it_mod_b = modified_newton(-0.9)

    print(f"Newton:           p = {p_newton_b:.10f}, iterations = {it_newton_b}")
    print(f"Secant:           p = {p_secant_b:.10f}, iterations = {it_secant_b}")

    if it_mod_b == "Diverged":
        print(f"Modified Newton:  p = {p_mod_b:.10f}, result = Diverged")
    else:
        print(f"Modified Newton:  p = {p_mod_b:.10f}, iterations = {it_mod_b}")
