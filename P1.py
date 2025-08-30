import numpy as np
import pandas as pd
import random
from collections import Counter

# PARAMETERS

PURCHASE_COST = 0.30
SALES_PRICE = 0.45
SCRAP_PRICE = 0.05

# Probabilities for news type
NEWS_TYPE_RD_RANGES = {
    'Good': (1, 35),    # 0.35
    'Fair': (36, 80),   # 0.45
    'Poor': (81, 100)   # 0.20
}

# Demand assignment (from given cumulative tables)
DEMAND_RD_RANGES = {
    'Good': {
        40: (1, 3), 50: (4, 8), 60: (9, 23), 70: (24, 43),
        80: (44, 78), 90: (79, 93), 100: (94, 100)
    },
    'Fair': {
        40: (1, 10), 50: (11, 28), 60: (29, 68), 70: (69, 88),
        80: (89, 96), 90: (97, 100), 100: (97, 100)
    },
    'Poor': {
        40: (1, 44), 50: (45, 66), 60: (67, 82), 70: (83, 94),
        80: (95, 100), 90: (95, 100), 100: (95, 100)
    }
}


# HELPERS

def get_news_type(rd):
    for t, (low, high) in NEWS_TYPE_RD_RANGES.items():
        if low <= rd <= high:
            return t
    return None


def get_demand(news_type, rd):
    for demand, (low, high) in DEMAND_RD_RANGES[news_type].items():
        if low <= rd <= high:
            return demand
    return 0


def simulate_day(order_qty):
    news_type = get_news_type(random.randint(1, 100))
    demand = get_demand(news_type, random.randint(1, 100))

    sales = min(order_qty, demand)
    revenue = sales * SALES_PRICE
    excess_demand = max(0, demand - order_qty)
    loss_profit = excess_demand * (SALES_PRICE - PURCHASE_COST)
    unsold = max(0, order_qty - demand)
    salvage = unsold * SCRAP_PRICE
    daily_profit = revenue + salvage - (order_qty * PURCHASE_COST)

    return {
        "News_Type": news_type,
        "Demand": demand,
        "Revenue": revenue,
        "Loss_of_Profit": loss_profit,
        "Salvage": salvage,
        "Daily_Profit": daily_profit
    }


def run_simulation(days, order_qty):
    results = [simulate_day(order_qty) for _ in range(days)]
    return pd.DataFrame(results)


# MAIN SIMULATION

ORDER_QUANTITY = 70
SIM_DAYS = [200, 500, 1000, 10000]

for d in SIM_DAYS:
    df = run_simulation(d, ORDER_QUANTITY)
    print(f"\n--- Simulation for {d} days ---")
    print(df[['Revenue', 'Loss_of_Profit', 'Salvage', 'Daily_Profit']].describe())
    print(f"Average Daily Profit = {df['Daily_Profit'].mean():.4f} dollars")


# ===========================
# PART (a) - News type digits
# ===========================
print("\n(a) 100 Random digits for News Types:")
news_rds = [random.randint(1, 100) for _ in range(100)]
news_types = [get_news_type(rd) for rd in news_rds]
print("First 20 Digits:", news_rds[:20])
print("First 20 Types :", news_types[:20])
print("Counts:", Counter(news_types))


# ===========================
# PART (b) - Demand distributions
# ===========================
print("\n(b) Random demand samples:")

# Good → Exponential(mean=50)
good_samples = [int(x) for x in np.random.exponential(50, 500) if 0 <= x <= 100][:100]
print("Good (Exponential, mean=50):", good_samples[:20])

# Fair → Normal(mean=50, sd=10)
fair_samples = [int(x) for x in np.random.normal(50, 10, 500) if 0 <= x <= 100][:100]
print("Fair (Normal, mean=50, sd=10):", fair_samples[:20])

# Poor → Poisson(mean=50)
poor_samples = [int(x) for x in np.random.poisson(50, 500) if 0 <= x <= 100][:100]
print("Poor (Poisson, mean=50):", poor_samples[:20])



# ========================================================================================================================


import matplotlib.pyplot as plt
import random

# PARAMETERS

SIM_TIME = 1000                   # total simulation time
MEAN_INTERARRIVAL = 10            # mean interarrival time (exponential)
SERVICE_TIME_RANGE = (8, 12)      # service time uniform in [8, 12]

# INITIALIZATION

num_arrived = 0
num_served = 0
server_busy = 0

clock = 0.0
total_wait_time = 0.0
area_num_waiting = 0.0
area_server_busy = 0.0

next_arrival_time = random.expovariate(1.0 / MEAN_INTERARRIVAL)
next_departure_time = float('inf')

queue = []
waiting_times = []
time_points = [0.0]
queue_lengths = [0]
server_status = [server_busy]


# ==============================
# GENERATORS
# ==============================
def generate_interarrival():
    return random.expovariate(1.0 / MEAN_INTERARRIVAL)

def generate_service():
    return random.randint(SERVICE_TIME_RANGE[0], SERVICE_TIME_RANGE[1])

# ==============================
# SIMULATION LOOP
# ==============================
while clock < SIM_TIME:
    next_event_time = min(next_arrival_time, next_departure_time)

    # Update time-weighted averages
    dt = next_event_time - clock
    area_num_waiting += len(queue) * dt
    area_server_busy += server_busy * dt

    clock = next_event_time
    time_points.append(clock)
    
    queue_lengths.append(len(queue))
    server_status.append(server_busy)

    # ARRIVAL
    if next_arrival_time <= next_departure_time:
        num_arrived += 1
        next_arrival_time = clock + generate_interarrival()

        if server_busy == 0:
            server_busy = 1
            service_time = generate_service()
            next_departure_time = clock + service_time
            waiting_times.append(0)
        else:
            queue.append(clock)

    # DEPARTURE
    else:
        num_served += 1
        if len(queue) > 0:
            arrival_time = queue.pop(0)
            wait = clock - arrival_time
            waiting_times.append(wait)
            total_wait_time += wait

            service_time = generate_service()
            next_departure_time = clock + service_time
        else:
            server_busy = 0
            next_departure_time = float('inf')

# ==============================
# RESULTS
# ==============================
avg_wait = total_wait_time / num_served if num_served > 0 else 0.0
avg_num_waiting = area_num_waiting / clock
utilization = area_server_busy / clock

print(f"\nSimulation Results (up to {SIM_TIME} mins):")
print(f"Total Customers Arrived: {num_arrived}")
print(f"Total Customers Served: {num_served}")
print(f"Average Waiting Time: {avg_wait:.2f} minutes")
print(f"Average Number Waiting: {avg_num_waiting:.2f}")
print(f"Server Utilization: {utilization:.2%}")

# ==============================
# PLOTS
# ==============================
plt.figure(figsize=(12, 6))

# Queue length sample path
plt.subplot(2, 1, 1)
plt.step(time_points, queue_lengths, where='post', color='blue')

plt.xlabel("Time")
plt.ylabel("Queue Length")
plt.title("Queue Length Sample Path")

plt.grid(True)

# Server status
plt.subplot(2, 1, 2)
plt.step(time_points, server_status, where='post', color='red')

plt.xlabel("Time")
plt.ylabel("Server Status (0=Idle, 1=Busy)")
plt.title("Server Status Over Time")

plt.yticks([0, 1], ["Idle", "Busy"])

plt.grid(True)

plt.tight_layout()
plt.show()

# Histogram of waiting times
actual_waits = [wt for wt in waiting_times if wt > 0]
if actual_waits:
    plt.figure(figsize=(8, 5))
    plt.hist(actual_waits, bins=20, edgecolor='black')

    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Customer Waiting Times (>0)")

    plt.grid(True)
    plt.show()
else:
    print("\nNo customers had to wait.")


# ====================================================================================

import numpy as np

# ---- Given data ----
time_min = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # minutes
velocity_mph = np.array([0, 10, 18, 25, 29, 32, 20, 11, 5, 2, 0])  # miles/hr

# Convert time to hours (since v is in miles/hr)
time_hr = time_min / 60.0

# ---- (i) Acceleration at 4 min using central difference ----
i = 2   # index of t=4
accel = (velocity_mph[i+1] - velocity_mph[i-1]) / (time_hr[i+1] - time_hr[i-1])
print(f"Acceleration at 4 min = {accel:.2f} miles/hr^2")

# ---- (ii) Distance travelled using Simpson’s 1/3 rule ----
n = len(time_hr) - 1
h = (time_hr[-1] - time_hr[0]) / n

# Apply Simpson’s 1/3 directly to tabulated y-values
distance = (h/3) * (
    velocity_mph[0]
    + 4 * sum(velocity_mph[1:-1:2])
    + 2 * sum(velocity_mph[2:-2:2])
    + velocity_mph[-1]
)
print(f"Distance travelled in 20 min = {distance:.4f} miles")