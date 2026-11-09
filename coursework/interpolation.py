import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline

def f(x):
    return 1 / (1 + 25*x**2)

def fp(x):
    return -50*x / (1 + 25*x**2)**2


if __name__ == "__main__":
    Ns = [7, 11, 20]
    x_plot = np.linspace(-1, 1, 2000)
    y_true = f(x_plot)

    for N in Ns:
        # 1. Lagrange interpolation with equally spaced nodes
        x_equi = np.linspace(-1, 1, N)
        y_equi = f(x_equi)
        lag_equi = BarycentricInterpolator(x_equi, y_equi)
        y_lag_equi = lag_equi(x_plot)

        # 2. Lagrange interpolation with Chebyshev nodes
        k = np.arange(N)
        x_cheb = np.cos((2*k + 1) * np.pi / (2*N))
        y_cheb = f(x_cheb)
        lag_cheb = BarycentricInterpolator(x_cheb, y_cheb)
        y_lag_cheb = lag_cheb(x_plot)

        # 3. Cubic spline with clamped boundary conditions
        spline = CubicSpline(
            x_equi,
            y_equi,
            bc_type=((1, fp(x_equi[0])), (1, fp(x_equi[-1])))
        )
        y_spline = spline(x_plot)

        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, y_true, label="Original f(x)", linewidth=2)
        plt.plot(x_plot, y_lag_equi, "--", label="Lagrange, equally spaced")
        plt.plot(x_plot, y_lag_cheb, "-.", label="Lagrange, Chebyshev nodes")
        plt.plot(x_plot, y_spline, ":", label="Clamped cubic spline")
        plt.scatter(x_equi, y_equi, s=20, label="Equally spaced nodes")
        plt.title(f"Interpolation comparison, N = {N}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
