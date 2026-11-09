import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**2 * np.log(x)

# Exact derivative:
# f'(x) = 2x ln(x) + x
true_value = 4 * np.log(2) + 2

# Forward difference approximation
def forward_difference(h):
    return (f(2 + h) - f(2)) / h


if __name__ == "__main__":
    print("True value of f'(2):")
    print(true_value)

    print("\nForward difference formula:")
    print("(f(2+h)-f(2))/h")



    errors = []
    h_values = []

    for n in range(1, 16):

        h = 10**(-n)

        approx = forward_difference(h)

        error = abs(approx - true_value)

        h_values.append(h)
        errors.append(error)

        print("n =", n)
        print("h =", h)
        print("Approximation =", approx)
        print("Error =", error)
        print()

    # Find optimal h
    min_index = np.argmin(errors)

    best_h = h_values[min_index]
    best_error = errors[min_index]

    print("Optimal h =", best_h)
    print("Minimum error =", best_error)

    # Plot error curve
    plt.figure(figsize=(6,4))
    plt.loglog(h_values, errors, marker='o')
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Error vs h")
    plt.grid(True)
    plt.show()



    n_opt = min_index + 1

    fine_h = []
    fine_error = []

    for m in range(200):

        h = 10**(-n_opt - 1 + 0.01*m)

        approx = forward_difference(h)

        error = abs(approx - true_value)

        fine_h.append(h)
        fine_error.append(error)

    # Best h in fine search
    fine_index = np.argmin(fine_error)

    print("\nFine search optimal h =", fine_h[fine_index])
    print("Fine search minimum error =", fine_error[fine_index])

    # Plot fine error curve
    plt.figure(figsize=(6,4))
    plt.loglog(fine_h, fine_error)
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Fine Search Error Curve")
    plt.grid(True)
    plt.show()
