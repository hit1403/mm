import numpy as np
import matplotlib.pyplot as plt

# Standard normal distribution
def f(x):
    return (1/np.sqrt(2*np.pi) * np.exp(-x**2/2))

# Trapezoidal Rule
def trapezoidal(f, a, b, n):
    """ 
    Approximates the integral of f(x) from a to b using
    composite Trapezoidal Rule with n sub-intervals
    """

    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)

    result = (h/2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
    return result

# Simpson's Rule
def simpsons(f, a, b, n):
    """
    Approximates the integral of f(x) from a to b using
    Simpsons 1/3 Rule with n sub-intervals (n must be even)
    """

    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)

    result = (h/3) * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-2:2]) + y[-1])
    return result


# Problem setup
a, b = -10, 2
true_value = 0.977249868051821

# To run for 2001 and 4001 points
for n in [2000, 4000]:
    trapezoidal_result = trapezoidal(f, a, b, n)
    simpsons_result = simpsons(f, a, b, n if n%2==0 else n+1)

    print(f"Using {n+1} points: ")
    print("Trapezoidal =", trapezoidal_result, "Error =", abs(true_value - trapezoidal_result))
    print("Simpson's =", simpsons_result, "Error =", abs(true_value - simpsons_result))

# Plotting
x_vals = np.linspace(-4, 4, 500)

plt.plot(x_vals, f(x_vals), 'b', label="Standard Normal PDF")
plt.fill_between(x_vals, f(x_vals), where=(x_vals<=2), color='skyblue', alpha=0.4)
plt.axvline(x=2, color='r', linestyle='--', label="x=2")

plt.title("Probability P(Z < 2)")

plt.legend()
plt.show()