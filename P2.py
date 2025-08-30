import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import prettytable as pt

from scipy.stats import ttest_ind


# 1. Newton Forward Interpolation

def newton_forward(x, y, x_eval, show_table=False):
    n = len(x)
    h = x[1] = x[0]


    # Difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]

    if show_table:
        table = pt.PrettyTable()
        headers = ["x"] + [f"Order {j}" for j in range(n)]
        table.field_names = headers

        for i in range(n):
            row = [x[i]]

            for j in range(n):
                if i + j < n:
                    row.append(diff_table[i][j])
                else:
                    row.append("")
            
            table.add(row)
        
        print(table)


    # Formula :
    u = (x_eval - x[0]) / h
    result = y[0]
    u_term = 1
    fact = 1

    for j in range(1, n):
        u_term *= (u - (j - 1))
        fact *= j
        result += (u_term / fact) * diff_table[0][j]

    return result


# 2. Newton Backward Interpolation

def newton_backward(x, y, x_eval, show_table=False):
    n = len(x)
    h = x[1] = x[0]


    # Difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(j, n):
            diff_table[i][j] = diff_table[i][j-1] - diff_table[i-1][j-1]

    if show_table:
        table = pt.PrettyTable()
        headers = ["x"] + [f"Order {j}" for j in range(n)]
        table.field_names = headers

        for i in range(n):
            row = [x[i]]

            for j in range(n):
                if i - j >= 0:
                    row.append(diff_table[i][j])
                else:
                    row.append("")
            
            table.add(row)
        
        print(table)


    # Formula :
    u = (x_eval - x[-1]) / h
    result = y[-1]
    u_term = 1
    fact = 1

    for j in range(1, n):
        u_term *= (u + (j - 1))
        fact *= j
        result += (u_term / fact) * diff_table[-1][j]


# 3. Newton Divided Difference Interpolation

def divided_difference(x, y, x_eval, show_table=False):
    n = len(x)

    # Difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = ( diff_table[i+1][j-1] - diff_table[i][j-1] ) / ( x[i+j] - x[i])

    if show_table:
        table = pt.PrettyTable()
        headers = ["x"] + [f"Order {j}" for j in range(n)]
        table.field_names = headers

        for i in range(n):
            row = [x[i]]

            for j in range(n):
                if i + j < n:
                    row.append(diff_table[i][j])
                else:
                    row.append("")
            
            table.add(row)
        
        print(table)


    # Formula :
    result = y[0]
    term = 1
    
    for j in range(1, n):
        term *= (x_eval - x[j-1])
        result += term * diff_table[0][j]

    return result


# 4. Lagrange Interpolation

def lagrange(x, y, x_eval):
    n = len(x)
    result = 0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_eval - x[j] / (x[i] - x[j]))

        result += term

    return result


# 5. Cubic Spline Interpolation

def cubic_spline(x, f, x_eval):
    x = np.array(x, dtype=float)
    f = np.array(f, dtype=float)
    
    n = len(x)

    A = np.zeros((n, n))
    b = np.zeros(n)

    # Natural spline boundary conditions
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    
    b[0] = 0
    b[n - 1] = 0

    for i in range(1, n - 1):
        A[i, i - 1] = x[i] - x[i - 1]
        
        A[i, i] = 2 * (x[i + 1] - x[i - 1])
        
        A[i, i + 1] = x[i + 1] - x[i]
        
        b[i] = 6 * (
            (f[i + 1] - f[i]) / (x[i + 1] - x[i])
            - (f[i] - f[i - 1]) / (x[i] - x[i - 1])
        )

    f_double_prime = solve(A, b)

    # Interpolation
    if x_eval <= x[0]:
        i = 0
    elif x_eval >= x[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x[1:], x_eval)
    
    i = min(max(i, 0), n - 2)

    result = (
        f_double_prime[i]/(6 * (x[i+1]-x[i])) * (x[i+1]-x_eval)**3 + f_double_prime[i+1]/(6*(x[i+1]-x[i]))*(x_eval-x[i])**3 +
        ((f[i]/(x[i+1]-x[i]))-(f_double_prime[i]*(x[i+1]-x[i])/6))*(x[i+1]-x_eval) +
        ((f[i+1]/(x[i+1]-x[i]))-(f_double_prime[i+1]*(x[i+1]-x[i])/6))*(x_eval-x[i])
    )

    return result

# ----------------------------------------------------------------------------------------------------

# Main Method

if __name__ == "__main__":

    # Step 1 : Insert given table values

    x = np.array([0, 8, 16, 24, 32, 40])
    y = np.array([14.621, 11.843, 9.870, 8.418, 7.305, 6.413])

    # Step 2 : Choose Interpolation method

    METHOD = "spline"
    test_points = [5]

    method_functions = {
        "forward" : lambda X, Y, xi: newton_forward(X, Y, xi, show_table=(xi == test_points[0])),
        "backward": lambda X, Y, xi: newton_backward(X, Y, xi, show_table=(xi == test_points[0])),
        "divided": lambda X, Y, xi: divided_difference(X, Y, xi, show_table=(xi == test_points[0])),
        "lagrange": lagrange,
        "spline": cubic_spline,
    }

    interpolation_function = method_functions[METHOD]
    y_eval_points = [interpolation_function(x, y, xi) for xi in test_points]

    print(f"Interpolation Results ({METHOD.title()})")
    for xi, yi in zip(test_points, y_eval_points):
        print(f"At x = {xi}, Interpolated y = {yi:.4f}")

    # Step 3 : Error analysis table and plot
    
    DO_ERROR_ANALYSIS = True

    if DO_ERROR_ANALYSIS:
        y_fit = [interpolation_function(x, y, xi) for xi in x]
        errors = y - np.array(y_fit)

        # Table
        table = pt.PrettyTable(["x", "True y", "Interpolated y", "Error"])

        for xi, yi_true, yi_fit in zip(x, y, y_fit):
            table.add_row([xi, yi_true, yi_fit, yi_true - yi_fit])

        print(table)

        # Plot
        plt.plot(x, errors, 'ro-', label="Error")
        
        plt.axhline(0, color='k', linestyle='--')

        plt.xlabel("X")
        plt.ylabel("Error")
        plt.title("Error Plot")
        
        plt.legend()
        plt.grid(True)
        plt.show()


        # Step 4 : Goodness of fit (Independent t-test)

        t_stat, p_val = ttest_ind(y, y_fit)
        
        print(f"\nT-test result: t = {t_stat:.4f}, p = {p_val:.4e}")
        
        if p_val > 0.05:
            print("Interpolating polynomial fits data well (fail to reject H0).")
        else:
            print("Significant difference (reject H0, poor fit).")

    # Step 5 : Plot comparison

    x_dense = np.linspace(min(x), max(x), 200)
    y_dense = [interpolation_function(x, y, xi) for xi in x_dense]
    
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='black', label="Given Data (Table Points)")
    
    plt.plot(x_dense, y_dense, 'r--', label=f"{METHOD.title()} Polynomial/Spline")
    
    plt.scatter(test_points, y_eval_points, color='blue', marker='s', s=100, label="Interpolated Values")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{METHOD.title()} Interpolation with Highlighted Points")
    
    plt.legend()
    plt.grid(True)
    plt.show()