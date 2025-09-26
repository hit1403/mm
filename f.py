import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

year = [1992, 1993, 1994, 1995]
yVal = [293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
n = len(yVal)

moving_total = [sum(yVal[i:i+4]) for i in range(n-3)]

moving_total_avg = [tot/4 for tot in moving_total]

centered_moving_avg = [(moving_total_avg[i] + moving_total_avg[i+1])/2
                       for i in range(len(moving_total_avg)-1)]

percentage = [round(yVal[i+2]/centered_moving_avg[i]*100, 3)
              for i in range(len(centered_moving_avg))]

first, second, third, fourth = [], [], [], []
i, j, k, l = 2, 3, 0, 1
while i<len(percentage) and j<len(percentage) and k<len(percentage) and l<len(percentage):
    first.append(percentage[i])
    second.append(percentage[j])
    third.append(percentage[k])
    fourth.append(percentage[l])
    i+=4
    j+=4
    k+=4
    l+=4

quarters=[first, second, third, fourth]

modified_sum = [round(sum(each)-min(each)-max(each), 3) for each in quarters]

modified_mean = [each/2 for each in modified_sum]

adj_factor = round(400/sum(modified_mean), 4)

seasonal_indices = [round(each*adj_factor, 3) for each in modified_mean]

deseasonalized_data = [round(yVal[i]/(seasonal_indices[i%4]/100), 3) for i in range(len(yVal))]

# Least square regression for trend
half = n / 2
XBy2 = [(-half + 0.5) + i for i in range(n)]
X = [x*2 for x in XBy2]
XY = [round(x*y, 3) for x, y in zip(X, deseasonalized_data)]
X2 = [x**2 for x in X]
Y_mean = sum(deseasonalized_data)/n
b = sum(XY) / sum(X2)
a = Y_mean

# Cyclic variation
Y_pred = [round(a+(b*each),4) for each in X]
cyclic_variation = [round((y/y_pred)*100, 4) for y, y_pred in zip(deseasonalized_data, Y_pred)]

# Table 1: First 4 steps
year_display = []
for each in year:
    year_display.append(each)
    year_display.extend(['-', '-', '-'])

table1 = pd.DataFrame({
    "Year(1)": year_display,
    "Quarter(2)": ['I', 'II', 'III', 'IV']*len(year),
    "Actual Value(3)": yVal,
    "Moving Total(4)": ['-']*2 + moving_total + ['-'],
    "Moving Average(5)=(4)/4": ['-']*2 + moving_total_avg + ['-'],
    "Centered Moving Average(6)": ['-']*2 + centered_moving_avg + ['-']*2,
    "Percentage of Actual to Moving Average(7)": ['-']*2 + percentage + ['-']*2
})
print("\nCalculation of the first 4 steps to compute seasonal index")
print(table1)

# Table 2: Steps 5 & 6
table2 = pd.DataFrame({
    "Year": year+["Modified Sum"]+["Modified Mean"],
    "Quarter 1": ['-']+quarters[0]+[modified_sum[0]]+[modified_mean[0]],
    "Quarter 2": ['-']+quarters[1]+[modified_sum[1]]+[modified_mean[1]],
    "Quarter 3": quarters[2]+['-']+[modified_sum[2]]+[modified_mean[2]],
    "Quarter 4": quarters[3]+['-']+[modified_sum[3]]+[modified_mean[3]]
})
print("\nSteps 5 and 6 in computing the seasonal index")
print(table2)

print("\nAdjusting Factor : ", adj_factor, "\n")

table3 = pd.DataFrame({
    "Quarter": ['I', 'II', 'III', 'IV'],
    "Indices": modified_mean,
    "Seasonal Indices": seasonal_indices
})
print(table3)
print("\nSum of seasonal indices : ", sum(seasonal_indices))

# Table 4: Deseasonalized data
table4 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Actual Value(3)": yVal,
    "Seasonal index/100 (4)": [each/100 for each in seasonal_indices]*len(year),
    "Deseasonalized data (5)": deseasonalized_data
})
print("\nCalculation of deseasonalized time series values")
print(table4)

# Table 5: Trend component
table5 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Y-Deseasonalized data (3)": deseasonalized_data,
    "Translating or Coding Time Var (4)": XBy2,
    "X (5)=(4)*2": X,
    "XY (6)=(5)*(3)": XY,
    "X**2 (7)": X2
})
print("\nIdentifying the trend component")
print(table5)
print(f"\nTrend line : y = {a:.3f} + {b:.3f}x\n")

# Table 6: Cyclic variation
table6 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Y-Deseasonalized data (3)": deseasonalized_data,
    "a + bx = Y (4)": Y_pred,
    "Percent of Trend (5)": cyclic_variation
})
print("\nIdentifying the cyclic variation")
print(table6)

# ---- Plotting ---- #
labels = []
for yr in year:
    labels += [f"{yr} Q1", f"{yr} Q2", f"{yr} Q3", f"{yr} Q4"]

plt.figure(figsize=(10, 6))
plt.plot(yVal, label="Actual data")
plt.scatter(range(len(labels)), yVal)
plt.plot(Y_pred, label="Trend line")
plt.scatter(range(len(labels)), Y_pred)
plt.plot(deseasonalized_data, label="Deseasonalized data")
plt.scatter(range(len(labels)), deseasonalized_data)
plt.plot([np.nan, np.nan] + centered_moving_avg + [np.nan, np.nan], label="Centered mov.avg")
plt.scatter(range(len(labels)), [np.nan, np.nan] + centered_moving_avg + [np.nan, np.nan])

plt.xticks(range(len(labels)), labels, rotation=45)
plt.xlabel("Year and Quarter")
plt.ylabel("Values")
plt.title("Time Series with Trend Line")
plt.legend()
plt.show()


###################################################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Given data as pandas Series ---
y = pd.Series([
    29,20,25,29,31,33,34,27,26,30,
    29,28,28,26,27,26,30,28,26,30,
    31,30,37,30,33,31,27,33,37,29,
    28,30,29,34,30,20,17,23,24,34,
    36,35,33,29,25,27,30,29,28,32
])

# Parameters
y_mean = y.mean()
time_lag = 25
n = len(y)

# --- ACF calculation ---
cov_list = []
for i in range(time_lag + 1):
    covariance = ((y - y_mean) * (y.shift(i) - y_mean)).sum() / n
    cov_list.append(covariance)

rho = [c / cov_list[0] for c in cov_list]  # ACF values

# --- PACF calculation ---
def calculate_pacf(y, lags, rho):
    pacf_vals = [1.0]  # PACF(0) = 1
    for k in range(1, lags+1):
        P_k = np.array([[rho[abs(i-j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k+1])
        phi_k = np.linalg.solve(P_k, rho_k)  # Yule-Walker
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)

pacf_vals = calculate_pacf(y, time_lag, rho)

# --- Print ACF + PACF table ---
print(" Time Lag |    Covariance    |    Rho (ACF)   |    PACF")
print("----------|------------------|----------------|----------------")
for i in range(time_lag + 1):
    print(f"{i:<9} | {cov_list[i]:<16.6f} | {rho[i]:<14.6f} | {pacf_vals[i]:<14.6f}")

# --- Significance level ---
conf_level = 2 / np.sqrt(n)

# --- ACF significance testing ---
print("\nACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'ACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

for lag, val in enumerate(rho):
    if lag == 0:
        significant, decision = "N/A", "ACF(0) = 1 (by definition)"
    else:
        if abs(val) > conf_level:
            significant, decision = "Yes", "Reject H0: Significant autocorrelation"
        else:
            significant, decision = "No", "Fail to reject H0: Not significant"
    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

# --- PACF significance testing ---
print("\nPACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'PACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

for lag, val in enumerate(pacf_vals):
    if lag == 0:
        significant, decision = "N/A", "PACF(0) = 1 (by definition)"
    else:
        if abs(val) > conf_level:
            significant, decision = "Yes", "Reject H0: Significant partial autocorrelation"
        else:
            significant, decision = "No", "Fail to reject H0: Not significant"
    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

# --- Plot ACF ---
plt.figure(figsize=(8, 6))
plt.stem(range(len(rho)-1), rho[1:])
plt.axhline(0, color='black')
plt.axhline(conf_level, color='red', linestyle='--', linewidth=0.8, label='95% CI')
plt.axhline(-conf_level, color='red', linestyle='--', linewidth=0.8)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Time Lag')
plt.ylabel('ACF (rho)')
plt.legend()
plt.show()

# --- Plot PACF ---
plt.figure(figsize=(8, 6))
plt.stem(range(len(pacf_vals)), pacf_vals)
plt.axhline(0, color='black')
plt.axhline(conf_level, color='red', linestyle='--', label='95% CI')
plt.axhline(-conf_level, color='red', linestyle='--')
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




################################################################


# EXAMPLE 1 (Exercise problem 4.8)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

# Dataset
periods = np.array(list(range(1, 25)))
yt = [
    315, 195, 310, 316, 325, 335, 318, 355, 420, 410, 485, 420,
    460, 395, 390, 450, 458, 570, 520, 400, 420, 580, 475, 560
]


def exponential_smoothing(data, lambda_, y0):
    smoothed = np.zeros(len(data))
    smoothed[0] = y0
    for t in range(1, len(data)):
        smoothed[t] = lambda_ * data[t] + (1 - lambda_) * smoothed[t - 1]
    return smoothed

def second_order_smoothing(data, lambda_, y0):
    n = len(data)
    smoothed = np.zeros(n)
    smoothed[0] = y0
    smooth_2nd = np.zeros(n)
    smooth_2nd[0] = smoothed[0]

    for t in range(1, n):
        smoothed[t] = lambda_ * data[t] + (1 - lambda_) * smoothed[t-1]
        smooth_2nd[t] = lambda_ * smoothed[t] + (1 - lambda_) * smooth_2nd[t-1]

    # Final forecast values (double smoothed)
    final = 2 * smoothed - smooth_2nd
    return smoothed, smooth_2nd, final

# Parameters
lambda_02 = 0.2
lambda_04 = 0.4

smoothed_02 = exponential_smoothing(yt, lambda_02, yt[0])
smoothed_04 = exponential_smoothing(yt, lambda_04, yt[0])

sm1_02, sm2_02, final_02 = second_order_smoothing(yt, lambda_02, yt[0])
sm1_04, sm2_04, final_04 = second_order_smoothing(yt, lambda_04, yt[0])


data = {
    'Period': periods,
    'Original': yt,
    'Smoothed (λ=0.2)': smoothed_02,
    'Smoothed (λ=0.4)': smoothed_04,
    '2nd Order Final (λ=0.2)': final_02,
    '2nd Order Final (λ=0.4)': final_04
}
df = pd.DataFrame(data)
print(df)

# First-order smoothing
plt.plot(periods, yt, label='Original Data', marker='o', color='black')
plt.plot(periods, smoothed_02, label='1st Order (λ=0.2)', marker='o', color='blue')
plt.plot(periods, smoothed_04, label='1st Order (λ=0.4)', marker='o', color='red')
plt.xlabel('Period')
plt.ylabel('yt')
plt.title('First Order Exponential Smoothing')
plt.legend()
plt.grid(True)
plt.xticks(periods)
plt.tight_layout()
plt.show()

# Second-order smoothing (Final)
plt.plot(periods, yt, label='Original Data', marker='o', color='black')
plt.plot(periods, final_02, label='2nd Order Final (λ=0.2)', marker='o', color='blue')
plt.plot(periods, final_04, label='2nd Order Final (λ=0.4)', marker='o', color='red')
plt.xlabel('Period')
plt.ylabel('yt')
plt.title('Second Order Exponential Smoothing (Final)')
plt.legend()
plt.grid(True)
plt.xticks(periods)
plt.tight_layout()
plt.show()

def perform_ttest(original, smoothed, label, alpha=0.05):
    d = original - smoothed
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    n = len(d)
    t_stat = d_mean / (d_std / np.sqrt(n))
    p_val = 2 * t.sf(np.abs(t_stat), df=n-1)

    print(f"\nTest for {label}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    if p_val < alpha:
        print(f"Reject H0 at α={alpha}: Significant difference between original and {label}.")
    else:
        print(f"Fail to Reject H0 at α={alpha}: No significant difference between original and {label}.")

# First Order Tests
perform_ttest(yt, smoothed_02, "1st Order (λ=0.2)")
perform_ttest(yt, smoothed_04, "1st Order (λ=0.4)")

# Second Order Tests (Final series)
perform_ttest(yt, final_02, "2nd Order Final (λ=0.2)")
perform_ttest(yt, final_04, "2nd Order Final (λ=0.4)")





################################################################



import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


start_date = dt.datetime(2024, 1, 1)
end_date   = dt.datetime(2024, 12, 31)

tesla = yf.Ticker("TSLA")
data = tesla.history(start=start_date, end=end_date)
data["Rate of Return"] = (data["Close"] - data["Open"]) / data["Open"] * 100

data = data.sample(n=150, random_state=42).sort_index()
data["DOY"] = data.index.dayofyear


def coded_variable(n):
    if n % 2 == 0:
        return np.arange(-n+1, n, 2)[:n].astype(float)
    else:
        return np.arange(-(n//2), n//2 + 1).astype(float)

def fit_linear(y, x):
    a = np.mean(y)
    b = np.sum(x*y) / np.sum(x**2)
    y_pred = a + b*x
    return (a, b), y_pred

def fit_quadratic(y, x):
    n = len(y)
    sum_x2 = np.sum(x**2)
    sum_x4 = np.sum(x**4)
    sum_y  = np.sum(y)
    sum_x2y = np.sum(x**2 * y)

    b = np.sum(x*y) / sum_x2
    A = np.array([[n, sum_x2],[sum_x2, sum_x4]])
    B = np.array([sum_y, sum_x2y])
    a, c = np.linalg.solve(A, B)

    y_pred = a + b*x + c*(x**2)
    return (a, b, c), y_pred

def fit_cubic(y, x):
    X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
    coeffs = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_pred = X @ coeffs
    return tuple(coeffs), y_pred

def error_analysis(y, y_pred):
    resid = y - y_pred
    rmse = np.sqrt(np.mean(resid**2))
    mape = np.mean(np.abs(resid / y)) * 100
    r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
    return rmse, mape, r2


variables = {
    "Opening Stock": data["Open"].values,
    "Closing Stock": data["Close"].values,
    "Rate of Return": data["Rate of Return"].values
}

trend_types = ["Linear", "Quadratic", "Cubic"]
results_table = []

# Plot (all three regressions together)
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle("Secular Trend Fits (All Regressions Together)", fontsize=16)

for row, (var_name, y) in enumerate(variables.items()):
    x = coded_variable(len(y))
    ax = axes[row]
    ax.plot(range(len(y)), y, marker="o", markersize=3, color="black", label="Actual")

    # Fit all trends
    for trend in trend_types:
        if trend == "Linear":
            coeffs, y_pred = fit_linear(y, x)
        elif trend == "Quadratic":
            coeffs, y_pred = fit_quadratic(y, x)
        else:
            coeffs, y_pred = fit_cubic(y, x)

        rmse, mape, r2 = error_analysis(y, y_pred)
        results_table.append({
            "Variable": var_name,
            "Trend": trend,
            "Coefficients": np.round(coeffs, 4),
            "RMSE": round(rmse, 4),
        })

        # Add regression line
        ax.plot(range(len(y)), y_pred, lw=1.5, label=f"{trend} Fit")

    ax.set_title(var_name)
    ax.set_xlabel("Observation Index")
    ax.set_ylabel(var_name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

results_df = pd.DataFrame(results_table)
print("Secular Trend Results:")
print(results_df)



#############################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coded_variable(n):
    """Generate coded variable for trend fitting."""
    if n % 2 == 0:
        return np.arange(-n+1, n, 2)[:n].astype(float)
    else:
        return np.arange(-(n//2), n//2 + 1).astype(float)

def fit_linear(y, x):
    a = np.mean(y)
    b = np.sum(x*y) / np.sum(x**2)
    y_pred = a + b*x
    return (a, b), y_pred

def fit_quadratic(y, x):
    n = len(y)
    sum_x2 = np.sum(x**2)
    sum_x4 = np.sum(x**4)
    sum_y  = np.sum(y)
    sum_x2y = np.sum(x**2 * y)
    sum_xy = np.sum(x*y)

    b = sum_xy / sum_x2
    A = np.array([[n, sum_x2],[sum_x2, sum_x4]])
    B = np.array([sum_y, sum_x2y])
    a, c = np.linalg.solve(A, B)

    y_pred = a + b*x + c*(x**2)
    return (a, b, c), y_pred

def fit_cubic(y, x):
    X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
    coeffs = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_pred = X @ coeffs
    return tuple(coeffs), y_pred

def error_analysis(y, y_pred):
    resid = y - y_pred
    rmse = np.sqrt(np.mean(resid**2))
    mape = np.mean(np.abs(resid / y)) * 100
    r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
    return rmse, mape, r2


def secular_trend_analysis(csv_file, columns, trend_types=["Linear", "Quadratic", "Cubic"]):
    """
    csv_file  : Path to CSV file
    columns   : dict, { "Variable Name": "csv_column_name" }
    trend_types: list of regressions to fit ["Linear", "Quadratic", "Cubic"]
    """

    # Load dataset
    data = pd.read_csv(csv_file)

    # Prepare variables
    variables = {k: data[v].values for k, v in columns.items()}

    results_table = []
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4*len(variables)))
    fig.suptitle("Secular Trend Fits", fontsize=16)

    if len(variables) == 1:
        axes = [axes]  # make iterable if only one variable

    for row, (var_name, y) in enumerate(variables.items()):
        x = coded_variable(len(y))
        ax = axes[row]
        ax.plot(range(len(y)), y, marker="o", markersize=3, color="black", label="Actual")

        # Fit selected trends
        for trend in trend_types:
            if trend == "Linear":
                coeffs, y_pred = fit_linear(y, x)
            elif trend == "Quadratic":
                coeffs, y_pred = fit_quadratic(y, x)
            else:
                coeffs, y_pred = fit_cubic(y, x)

            rmse, mape, r2 = error_analysis(y, y_pred)
            results_table.append({
                "Variable": var_name,
                "Trend": trend,
                "Coefficients": np.round(coeffs, 4),
                "RMSE": round(rmse, 4),
                "MAPE": round(mape, 2),
                "R²": round(r2, 4)
            })

            ax.plot(range(len(y)), y_pred, lw=1.5, label=f"{trend} Fit")

        ax.set_title(var_name)
        ax.set_xlabel("Observation Index")
        ax.set_ylabel(var_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Results table
    results_df = pd.DataFrame(results_table)
    return results_df

# Suppose your CSV has columns: "Open", "Close", "Rate of Return"
columns_to_use = {
    "Opening Stock": "Open",
    "Closing Stock": "Close",
    "Rate of Return": "Rate of Return"
}

results = secular_trend_analysis("your_dataset.csv", columns_to_use)
print("Secular Trend Results:")
print(results)