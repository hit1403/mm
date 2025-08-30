import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import prettytable as pt

# ODE definition
def f(t, y):
    return (1 + 4*t) * np.sqrt(y)

# Euler Method
def euler(f, t0, y0, h, n):
    t = [t0]
    y = [y0]

    for i in range(n):
        y_next = y[-1] + h * f(t[-1], y[-1])
        t_next = t[-1] + h
        t.append(t_next)
        y.append(y_next)

    return np.array(t), np.array(y)

# Runge-Kutta 4th Order
def rk4(f, t0, y0, h, n):
    t = [t0]
    y = [y0]
    
    for i in range(n):
        k1 = f(t[-1], y[-1])
        k2 = f(t[-1] + h/2, y[-1] + h/2 * k1)
        k3 = f(t[-1] + h/2, y[-1] + h/2 * k2)
        k4 = f(t[-1] + h, y[-1] + h * k3)
    
        y_next = y[-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_next = t[-1] + h
    
        t.append(t_next)
        y.append(y_next)
    
    return np.array(t), np.array(y)


# Analytical Solution with Sympy
x = sp.Symbol('x')
y = sp.Function('y')(x)

diffeq = sp.Eq(y.diff(x), (1+4*x)*sp.sqrt(y))
sol = sp.dsolve(diffeq, y)

C1 = sp.Symbol('C1')
eq_for_C1 = sp.Eq(sol.rhs.subs(x, 0), 1)   # initial condition y(0) = 1

C1_val = [s for s in sp.solve(eq_for_C1, C1) if s > 0][0]

particular_sol = sol.subs(C1, C1_val)
y_exact_func = sp.lambdify(x, particular_sol.rhs, 'numpy')

print("Analytical Solution:", particular_sol)


# Problem setup
t0, t_final = 0.0, 1.0
y0 = 1.0
n = 20

h = (t_final - t0)/n

# Compute
t_e, y_e = euler(f, t0, y0, h, n)
t_r, y_r = rk4(f, t0, y0, h, n)
y_exact = y_exact_func(t_e)


table = pt.PrettyTable()
table.field_names = ["i", "t", "Euler", "RK4", "Exact", "Error Euler", "Error RK4"]

for i in range(len(t_e)):
    table.add_row([
        i, 
        round(t_e[i], 3), 
        round(y_e[i], 6), 
        round(y_r[i], 6), 
        round(y_exact[i], 6),
        abs(y_e[i]-y_exact[i]), 
        abs(y_r[i]-y_exact[i])
    ])

print(table)

# Plot
plt.plot(t_e, y_exact, 'k-', label="Analytical")
plt.plot(t_e, y_e, 'ro--', label="Euler")
plt.plot(t_r, y_r, 'bs-', label="RK4")

plt.xlabel("t")
plt.ylabel("y")
plt.title("Euler vs RK4 vs Analytical Solution")

plt.legend()
plt.show()