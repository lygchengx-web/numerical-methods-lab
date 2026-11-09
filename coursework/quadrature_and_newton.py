import math

def g(t):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-t**2 / 2)

def composite_trapezoid(x, N):
    h = x / N
    total = 0.5 * g(0) + 0.5 * g(x)

    for k in range(1, N):
        total += g(k * h)

    return h * total


if __name__ == "__main__":
    N = 50
    x = 0.5
    tau = 1e-5
    max_iter = 100

    for n in range(max_iter):
        f_approx = composite_trapezoid(x, N) - 0.45

        if abs(f_approx) < tau:
            break

        x = x - f_approx / g(x)

    print("Number of iterations:", n)
    print("Approximated root:", format(x, ".4f"))
    print("Residual:", abs(f_approx))
