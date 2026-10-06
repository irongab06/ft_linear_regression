# ft_linear_regression

42School Project: Implement a simple linear regression with gradient descent

## Objective 

Prediction of car price based on mileage using a linear regression

## Installation

The algorithm itself only uses the Python standard library (`csv`, `math`, `sys`): `predict.py` runs with any Python 3 and needs nothing.  
`matplotlib` is the only external dependency, used by `train.py` to draw the bonus graphs. The recommended way to install it is a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `matplotlib` is not installed, `train.py` still trains the model and saves `theta.txt`, it only skips the graphs.

## Usage

```bash
python3 train.py     # reads data.csv, trains the model, saves theta.txt and the graphs
python3 predict.py   # asks for a mileage and prints the estimated price
```

If `predict.py` is run before `train.py`, no `theta.txt` exists yet: θ₀ and θ₁ are set to 0 and the prediction is 0.

## Introduction

Linear regression is a very simple and widely used algorithm in machine learning.
Its purpose is to model the relationship between an independent variable **x** (here, the mileage of cars) and a dependent variable **y** (the price of the car) using a linear function.

## Model

F(x) = ax + b

<u>**x**</u> → independent variable (car mileage)  
<u>**F(x)**</u> → prediction (approximate price)  
<u>**a (or θ₁)**</u> → slope of the line (impact of mileage on the price)  
<u>**b (or θ₀)**</u> → intercept (approximate price when **x = 0**)  

### Function Model in python

```python
def estimate_price(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)
```

This is exactly the formula of the subject: **estimatePrice(mileage) = θ₀ + (θ₁ * mileage)**

<u>**mileage**</u> → independent variable **x** (car mileage)  
<u>**theta0**</u> → intercept **θ₀** (price when the mileage is 0)  
<u>**theta1**</u> → slope **θ₁** (price variation for one more unit of mileage)  
<u>**theta0 + (theta1 * mileage)**</u> → the linear model **ax + b**  

## Normalization

Mileages are large numbers (up to 250 000 km). With raw values the gradient of θ₁ is huge and gradient descent diverges.  
`train.py` therefore divides every mileage by `SCALE = 10000` before training:

```python
SCALE = 10000
x = [k / SCALE for k in km]
```

At the end of training, θ₁ is divided by `SCALE` before being saved, so that `predict.py` works directly on the raw mileage:

```python
with open("theta.txt", "w") as f:
	f.write(f"{theta0}\n{theta1 / SCALE}\n")
```

`theta.txt` contains **θ₀** on the first line and **θ₁** on the second line.

## Cost Function

The cost function is used to compute the sum of squared errors (Euclidean norm) in linear regression.

J(θ) = 1 / 2m * ​i=1 ∑ m ​( f( x(i) ) − y(i) )2

<u>**J(θ)**</u> -> value of the cost function (sum of squared errors)  
<u>**m**</u> -> m is the total number of samples  
<u>**​i=1 ∑ m**</u> -> sum from the first sample to the last one  
<u>**f(x(i))**</u> -> prediction of the model for sample **i (f(x)=ax+b)**
<u>**y(i)**</u> -> real (true) value for sample **i**  
<u>**1 / 2**</u> -> Normalization factor (we divide by the number of samples, and the 2 simplifies the derivatives later)  

### Function Cost in python

this function name is **Mean Squared Error** (MSE)

```python
def cost_function(x, y, theta0, theta1):
	m = len(x)
	return sum((estimate_price(x[i], theta0, theta1) - y[i]) ** 2 for i in range(m)) / (2 * m)
```

<u>**m**</u> → total number of samples (length of the list x)  
<u>**x**</u> → list containing all the (normalized) mileages  
<u>**y**</u> → list containing all real values (true prices of the samples)  
<u>**estimate_price(x[i], theta0, theta1) - y[i]**</u> → error for sample **i** (difference between predicted price and real price)  
<u>**(...)²**</u> → squared error, penalizing large deviations  
<u>**sum(... for i in range(m))**</u> → sum of all squared errors over the dataset  
<u>**/ (2 * m)**</u> → normalization factor (divide by the number of samples, and the 2 simplifies derivative calculation for the descent)  

## Algorithms of Minimization

This part is used to compute the gradient and perform gradient descent.  
It is very important for completing the linear regression process in machine learning.

### Gradient 

The gradient is the derivative of the cost function with respect to each parameter. It gives the direction in which **θ₀** and **θ₁** must move to decrease the cost.  
For a linear model, the two partial derivatives are the two formulas given in the subject:

tmpθ₀ = learningRate * 1/m * ​i=0 ∑ m-1 ( estimatePrice(mileage[i]) − price[i] )  
tmpθ₁ = learningRate * 1/m * ​i=0 ∑ m-1 ( estimatePrice(mileage[i]) − price[i] ) * mileage[i]  

<u>**estimatePrice(mileage[i]) − price[i]**</u> → error of the model for sample **i**  
<u>**1/m * ∑**</u> → mean of the errors (for **θ₀**), or mean of the errors weighted by the mileage (for **θ₁**)  
<u>**learningRate**</u> → step size of the update  

### Gradient descent 

its purpose is to update the parameters in the opposite direction of the gradient in order to minimize the value of **J(θ).**  


```python
def gradient_descent(x, y, learning_rate, n_iterations):
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
```


<u>**theta0, theta1 = 0.0, 0.0**</u> → the parameters start at zero  
<u>**errors**</u> → list of errors (predicted price − real price), computed once per iteration with the current **θ₀** and **θ₁**  
<u>**tmp_theta0 / tmp_theta1**</u> → the two update steps of the subject, stored in temporary variables  
<u>**theta0 = theta0 - tmp_theta0**</u> and <u>**theta1 = theta1 - tmp_theta1**</u> → simultaneous update: both temporaries are computed from the same old values before either parameter is changed. The subtraction moves the parameters in the opposite direction of the gradient (because the goal is to minimize the cost function, not maximize it)  
<u>**learning_rate**</u> → hyperparameter that controls the step size of the update (too large → divergence, too small → slow learning). Here **0.01**  
<u>**n_iterations**</u> → number of passes over the data. Here **2000**  
<u>**cost_history**</u> → cost after each iteration, used to draw the loss curve  

## Coefficient of determination

The coefficient of determination **(R²)** is a value between 0 and 1 that measures how well the model predicts the target variable.  

```python
def coef_determination(y, pred):
	mean_y = sum(y) / len(y)
	u = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
	v = sum((yi - mean_y) ** 2 for yi in y)
	return 1 - u / v
```

<u>**y**</u> → list of the true values (real prices).  
<u>**pred**</u> → list of the values predicted by the model.  
<u>**mean_y**</u> → mean of the real values (average price).  
<u>**(y[i] - pred[i])**</u> → residual (error between real and predicted value).  
<u>**u**</u> → **SSR** (Sum of Squared Residuals), total squared error of the model.  
<u>**v**</u> → **SST** (Total Sum of Squares), variance of the data relative to the mean.  
<u>**1 - u/v**</u> → formula of the coefficient of determination **(R²)**. It measures the proportion of the variance in y explained by the model (between **0** and **1**).  

## Root Mean Squared Error (RMSE)

It is an evaluation metric that measures the typical average deviation between the values predicted by your model and the true values.

```python
def rmse(y, pred):
	return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))
```

<u>**y**</u> → list of the true values (real prices).  
<u>**pred**</u> → list of the values predicted by the model.  
<u>**(y[i] - pred[i])²**</u> → squared residual (large errors are penalized more strongly).  
<u>**sum(...) / len(y)**</u> → **MSE** (Mean Squared Error), average squared error of the predictions.  
<u>**math.sqrt(...)**</u> → square root of the MSE, which brings the error back to the same unit as y (for example, euros).  

![linear Regression](Linear_Regression.png)

## Glossary of Abbreviations

<u>**MSE (Mean Squared Error)**</u> -> Average of the squared differences between predicted and true values.  
<u>**RMSE (Root Mean Squared Error)**</u> -> Square root of the **MSE**, brings the error back to the same unit as y.  
<u>**R² (Coefficient of Determination)**</u> -> Proportion of the variance in **y** explained by the model (between **0** and **1**).  
<u>**SSR (Sum of Squared Residuals)**</u> -> Sum of squared differences between predicted values and true values (errors of the model).  
<u>**SST (Total Sum of Squares)**</u> -> Total variance of the data relative to the mean of **y**.  
<u>**θ₀ (theta0)**</u> -> Intercept or bias (predicted value when **x = 0**).  
<u>**θ₁ (theta1)**</u> -> Slope or weight (impact of the variable **x** on **y**).  
<u>**J(θ) (Cost Function)**</u> -> Function measuring the error of the model (sum of squared errors).  
<u>**Learning rate (α)**</u> -> Hyperparameter that controls the step size in gradient descent updates.  
<u>**Gradient**</u> -> Vector of partial derivatives of the cost function with respect to the parameters (direction of steepest increase).  
<u>**Epoch / Iteration**</u> -> One complete update step of the parameters during gradient descent.  
<u>**Simultaneous update**</u> -> **θ₀** and **θ₁** are both computed from the same old values (temporary variables) before being assigned.  
<u>**Normalization (SCALE)**</u> -> Mileages are divided by 10000 during training so that gradient descent converges; **θ₁** is divided back before saving.  
<u>**Over-fitting**</u> -> When a model fits the training data too closely (including its noise) and predicts new data poorly. A prediction that falls exactly on a training value is suspicious.  

## Set of Mathematical Formulas and Programming

![linear Regression](Formule-math.jpg)![linear Regression](Formule-for-code.jpg)

## Auteur
Gabriel Cavalier  
[Mon GitHub](https://github.com/irongab06)
