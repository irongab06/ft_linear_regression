import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Couleurs ANSI
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def model(X, theta) :
	"""Return predictions using the linear regression model."""
	return X.dot(theta)

def cost_function(y, X, theta) :
	"""Return a Mean Squared Error"""
	m = len(y)
	return 1 / (2*m) * np.sum((model(X, theta) - y) ** 2)

def grad(X, y, theta) :
	"""Compute the gradient of the cost function for linear regression."""
	m = len(y)
	return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations) :
	"""Perform gradient descent to optimize theta and return final parameters and cost history."""
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations) :
		theta = theta - learning_rate * grad(X, y, theta)
		cost_history[i] = cost_function(y, X, theta)
	return theta, cost_history

def coef_determination(y, pred) :
	"""Compute the coefficient of determination (R²) to evaluate the prediction quality."""
	u = ((y - pred)**2).sum()
	v = ((y - y.mean())**2).sum()
	return 1 - u/v

def rmse(y, pred):
	"""Compute the Root Mean Squared Error (RMSE) between predictions and true values."""
	return np.sqrt(np.mean((y - pred)**2))


print("\033c", end="")


# ============================
# 1) Load data
# ============================

data = pd.read_csv('data.csv')
x = data['km'].values
y = data['price'].values

# ============================
# 2) Preprocess
# ============================

x = x.reshape((x.shape[0], 1))
y = y.reshape((y.shape[0], 1))
x = x / 10000 # normalize for convergence

X = np.hstack((x, np.ones(x.shape))) # Create the design matrix X with a bias term (column of ones)
theta = np.zeros((2, 1), dtype=float) # Initialize theta (parameters) to zeros as a (2,1) float vector


# ============================
# 3) Train
# ============================

theta, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=2000)
y_pred = model(X, theta)

np.savetxt("theta.txt", theta) #Save file theta for prog prediction

# ============================
# 4) Sort for plotting
# ============================

order = np.argsort(x[:, 0])
x_sorted = x[order]      # Sort x (and corresponding predictions) for a continuous regression line
y_pred_sorted = y_pred[order]

# ============================
# 5) Metrics
# ============================

Rmse =  rmse(y, y_pred)
determination = coef_determination(y, y_pred)

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
plt.scatter((x*10000), y, c='blue', label="Data points")
plt.xlabel("Km")
plt.ylabel("Price")
plt.legend()

# Subplot 2: Data + regression line
plt.subplot(3,1,2)
plt.scatter((x* 10000), y, label="Data points")  # nuage de points
plt.plot((x_sorted * 10000), y_pred_sorted, 'r', linewidth=2, label="Regression line")
plt.xlabel("Km")
plt.ylabel("Price")
plt.legend()

# Subplot 3: Loss curve
plt.subplot(3,1,3)
plt.plot(range(len(cost_history)), (cost_history / 1000), c='r', label="Loss curve")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE/2 ×10³)")
plt.legend()

plt.savefig('Linear_Regression.png')

plt.show()
