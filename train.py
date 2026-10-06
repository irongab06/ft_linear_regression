import csv
import math
import sys
import matplotlib.pyplot as plt

# Couleurs ANSI
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

SCALE = 10000  # mileage is divided by this value so that gradient descent converges


def estimate_price(mileage, theta0, theta1):
	"""estimatePrice(mileage) = theta0 + (theta1 * mileage)"""
	return theta0 + (theta1 * mileage)

def read_data(path):
	"""Read the csv file and return two lists: km and price."""
	km, price = [], []
	with open(path) as f:
		for row in csv.DictReader(f):
			km.append(float(row['km']))
			price.append(float(row['price']))
	return km, price

def cost_function(x, y, theta0, theta1):
	"""Return the Mean Squared Error divided by 2."""
	m = len(x)
	return sum((estimate_price(x[i], theta0, theta1) - y[i]) ** 2 for i in range(m)) / (2 * m)

def gradient_descent(x, y, learning_rate, n_iterations):
	"""Perform gradient descent and return theta0, theta1 and the cost history.

	tmp_theta0 = learningRate * 1/m * sum(estimatePrice(x[i]) - y[i])
	tmp_theta1 = learningRate * 1/m * sum((estimatePrice(x[i]) - y[i]) * x[i])
	theta0 and theta1 are updated simultaneously at the end of each iteration.
	"""
	m = len(x)
	theta0, theta1 = 0.0, 0.0
	cost_history = []
	for _ in range(n_iterations):
		errors = [estimate_price(x[i], theta0, theta1) - y[i] for i in range(m)]
		tmp_theta0 = learning_rate * sum(errors) / m
		tmp_theta1 = learning_rate * sum(errors[i] * x[i] for i in range(m)) / m
		theta0 = theta0 - tmp_theta0
		theta1 = theta1 - tmp_theta1
		cost_history.append(cost_function(x, y, theta0, theta1))
	return theta0, theta1, cost_history

def coef_determination(y, pred):
	"""Compute the coefficient of determination (R²) to evaluate the prediction quality."""
	mean_y = sum(y) / len(y)
	u = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
	v = sum((yi - mean_y) ** 2 for yi in y)
	return 1 - u / v

def rmse(y, pred):
	"""Compute the Root Mean Squared Error (RMSE) between predictions and true values."""
	return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))


print("\033c", end="")


# ============================
# 1) Load data
# ============================

try:
	km, price = read_data('data.csv')
except (OSError, KeyError, ValueError) as e:
	print(f"{RED}Could not read data.csv: {e}{RESET}")
	sys.exit(1)
if len(km) == 0:
	print(f"{RED}data.csv is empty{RESET}")
	sys.exit(1)

# ============================
# 2) Preprocess
# ============================

x = [k / SCALE for k in km]  # normalize for convergence

# ============================
# 3) Train
# ============================

theta0, theta1, cost_history = gradient_descent(x, price, learning_rate=0.01, n_iterations=2000)
y_pred = [estimate_price(xi, theta0, theta1) for xi in x]

# Save theta0 and theta1 for the prediction program.
# theta1 is converted back to "price per km" so that predict.py works on raw mileage.
with open("theta.txt", "w") as f:
	f.write(f"{theta0}\n{theta1 / SCALE}\n")

# ============================
# 4) Sort for plotting
# ============================

order = sorted(zip(km, y_pred))  # sort (km, prediction) pairs for a continuous regression line
km_sorted = [p[0] for p in order]
y_pred_sorted = [p[1] for p in order]

# ============================
# 5) Metrics
# ============================

Rmse = rmse(price, y_pred)
determination = coef_determination(price, y_pred)

print(f"{CYAN}{BOLD}╔══════════════════════════════╗{RESET}")
print(f"{CYAN}{BOLD}║   Mean Squared Error (MSE)   ║{RESET}")
print(f"{CYAN}{BOLD}╚══════════════════════════════╝{RESET}\n")
print(f"{RED}❌ Before : {RESET}  {YELLOW}{int(cost_history[0] * 2)}  "
      f"{GREEN}\n✅ After : {RESET}  {YELLOW}{int(cost_history[-1] * 2)}{RESET}\n")

print(f"{GREEN}{BOLD}╔═══════════════════════════════════════╗{RESET}")
print(f"{GREEN}{BOLD}║  Root Mean Squared Error (RMSE) [€]   ║{RESET}")
print(f"{GREEN}{BOLD}╚═══════════════════════════════════════╝{RESET}\n")
print(f"{YELLOW}{int(Rmse)} Euros{RESET}\n")

print(f"{RED}{BOLD}╔═══════════════════════════════════════╗{RESET}")
print(f"{RED}{BOLD}║   Coefficient of Determination (R²)   ║{RESET}")
print(f"{RED}{BOLD}╚═══════════════════════════════════════╝{RESET}\n")
print(f"{YELLOW}{determination:.2f}{RESET}\n")

# ============================
# 6) Graphs
# ============================

plt.figure("Graph – Linear Regression", figsize=(10,9))

# Subplot 1: Data only
plt.subplot(3,1,1)
plt.scatter(km, price, c='blue', label="Data points")
plt.xlabel("Km")
plt.ylabel("Price")
plt.legend()

# Subplot 2: Data + regression line
plt.subplot(3,1,2)
plt.scatter(km, price, label="Data points")  # nuage de points
plt.plot(km_sorted, y_pred_sorted, 'r', linewidth=2, label="Regression line")
plt.xlabel("Km")
plt.ylabel("Price")
plt.legend()

# Subplot 3: Loss curve
plt.subplot(3,1,3)
plt.plot(range(len(cost_history)), [c / 1000 for c in cost_history], c='r', label="Loss curve")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE/2 ×10³)")
plt.legend()

plt.savefig('Linear_Regression.png')

plt.show()
