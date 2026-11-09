import numpy as np

def GENP_2x2(A, b):
    """
    Gaussian elimination without pivoting for 2x2 system.
    """
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()

    # Eliminate A[1,0] using pivot A[0,0]
    m = A[1, 0] / A[0, 0]
    A[1, 1] = A[1, 1] - m * A[0, 1]
    b[1] = b[1] - m * b[0]
    A[1, 0] = 0.0

    # Back substitution
    x = np.zeros(2, dtype=np.float64)
    x[1] = b[1] / A[1, 1]
    x[0] = (b[0] - A[0, 1] * x[1]) / A[0, 0]

    return x


def GEPP_2x2(A, b):
    """
    Gaussian elimination with partial pivoting for 2x2 system.
    """
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()

    # Partial pivoting: swap rows if needed
    if abs(A[1, 0]) > abs(A[0, 0]):
        A[[0, 1], :] = A[[1, 0], :]
        b[[0, 1]] = b[[1, 0]]

    # Eliminate A[1,0]
    m = A[1, 0] / A[0, 0]
    A[1, 1] = A[1, 1] - m * A[0, 1]
    b[1] = b[1] - m * b[0]
    A[1, 0] = 0.0

    # Back substitution
    x = np.zeros(2, dtype=np.float64)
    x[1] = b[1] / A[1, 1]
    x[0] = (b[0] - A[0, 1] * x[1]) / A[0, 0]

    return x


def relative_forward_error(x, x_star):
    return np.linalg.norm(x - x_star, 2) / np.linalg.norm(x_star, 2)


def relative_residual(A, x, b):
    return np.linalg.norm(A @ x - b, 2) / np.linalg.norm(b, 2)


if __name__ == "__main__":
    x_star = np.array([1.0, 1.0], dtype=np.float64)

    print(f"{'k':>2} {'epsilon':>12} {'FE_GENP':>14} {'RR_GENP':>14} {'FE_GEPP':>14} {'RR_GEPP':>14}")

    for k in range(1, 17):
        eps = 10.0 ** (-k)

        A = np.array([[eps, 1.0],
                      [1.0, 1.0]], dtype=np.float64)

        b = A @ x_star

        x_genp = GENP_2x2(A, b)
        x_gepp = GEPP_2x2(A, b)

        FE_genp = relative_forward_error(x_genp, x_star)
        RR_genp = relative_residual(A, x_genp, b)

        FE_gepp = relative_forward_error(x_gepp, x_star)
        RR_gepp = relative_residual(A, x_gepp, b)

        print(f"{k:2d} {eps:12.1e} {FE_genp:14.6e} {RR_genp:14.6e} {FE_gepp:14.6e} {RR_gepp:14.6e}")
